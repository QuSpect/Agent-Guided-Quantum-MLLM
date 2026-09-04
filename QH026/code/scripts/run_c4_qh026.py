#!/usr/bin/env python3
"""Locked end-to-end priming and C4 evaluation for QH-026/CC-026."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM


PROJECT = Path(__file__).resolve().parents[2]
LOCK = PROJECT / "artifacts/qh026-butterfly-lock.json"
TRAIN_TOKENS = PROJECT / "datasets/processed/wikitext2_qwen38/train-tokens.pt"
EVAL_TOKENS = PROJECT / "datasets/processed/c4_validation_qwen38/validation-tokens.pt"
OUTPUT_DIR = PROJECT / "artifacts/c4_qh026"
sys.path.insert(0, str(PROJECT / "code/src"))

from quantum_qwen38.qh026_butterfly_replacement import freeze_and_replace_qh026


def load_common_runner():
    path = PROJECT / "code/scripts/run_c4_qh025.py"
    spec = importlib.util.spec_from_file_location("qh025_common_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load common QH025 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = load_common_runner()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=["base", "svd512", "qh026", "cc026"], required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_qh025_shared_initialization(replacement, candidate: str, lock: dict) -> dict:
    prefix = "qh025" if candidate == "qh026" else "cc025"
    path_key = "shared_qh025_parameters_from" if candidate == "qh026" else "shared_cc025_parameters_from"
    hash_key = f"{prefix}_sha256"
    path = Path(lock["initialization"][path_key])
    if sha256(path) != lock["initialization"][hash_key]:
        raise RuntimeError("QH025 parent checkpoint hash mismatch")
    state = torch.load(path, map_location=replacement.gamma.device, weights_only=True)
    expected = {"gamma", "down.weight", "up.weight", "core.theta"}
    if set(state) != expected or tuple(state["core.theta"].shape) != (2, 10, 3):
        raise RuntimeError(f"unexpected QH025 state: {[(key, tuple(value.shape)) for key, value in state.items()]}")
    with torch.no_grad():
        replacement.gamma.copy_(state["gamma"])
        replacement.down.weight.copy_(state["down.weight"])
        replacement.up.weight.copy_(state["up.weight"])
        replacement.core.theta[:2].copy_(state["core.theta"])
    return {
        "path": str(path),
        "sha256": sha256(path),
        "copied": ["gamma", "down.weight", "up.weight", "core.theta[0:2]"],
        "new_seeded": ["core.theta[2]"],
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("QH026 is GPU-only")
    if args.smoke and args.candidate not in {"qh026", "cc026"}:
        raise ValueError("smoke is trainable candidates only")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "locked_before_qh026_qh_cc_base_performance_results":
        raise RuntimeError("invalid QH026 lock")
    COMMON.seed_everything(int(lock["seed"]))
    train_tokens = torch.load(TRAIN_TOKENS, map_location="cpu", weights_only=True)
    eval_tokens = None if args.smoke else torch.load(EVAL_TOKENS, map_location="cpu", weights_only=True)
    model_dir = Path((PROJECT / "records/active_model_path.txt").read_text().strip())
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    replacement = None
    initialization = None
    if args.candidate == "base":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        injection = {
            "candidate": "frozen-dense-base",
            "base_model_parameters": base_parameters,
            "deployed_model_parameters": base_parameters,
            "net_model_parameter_reduction": 0,
            "trainable_parameter_count": 0,
        }
        training = {"steps": 0}
    else:
        core_kind = "classical" if args.candidate == "cc026" else "quantum"
        injection = freeze_and_replace_qh026(model, core_kind=core_kind)
        injection["candidate"] = {
            "svd512": "SVD-512-QH026-control",
            "qh026": "QH-026",
            "cc026": "CC-026",
        }[args.candidate]
        replacement = model.model.language_model.layers[7].self_attn.v_proj
        if args.candidate == "svd512":
            replacement.set_residual_enabled(False)
            training = {"steps": 0, "role": "zero-residual compression control"}
        else:
            initialization = load_qh025_shared_initialization(replacement, args.candidate, lock)
            training = COMMON.distill_end_to_end(model, replacement, train_tokens, lock, smoke=args.smoke)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        payload = {
            "status": "ok",
            "mode": "one-step train-only QH026 integration smoke",
            "candidate": args.candidate,
            "lock_sha256": sha256(LOCK),
            "training": training,
            "injection": injection,
            "initialization": initialization,
            "holdout_rows_inspected": 0,
            "test_rows_inspected": 0,
            "gpu_peak_allocated_bytes": [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path = OUTPUT_DIR / f"{args.candidate}-integration-smoke.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    checkpoint_path = OUTPUT_DIR / f"c4-{args.candidate}-s{lock['seed']}-replacement.pt"
    if replacement is not None and args.candidate in {"qh026", "cc026"}:
        torch.save(
            {name: parameter.detach().cpu() for name, parameter in replacement.named_parameters() if parameter.requires_grad},
            checkpoint_path,
        )
    if replacement is not None:
        replacement.strip_teacher()
    evaluation = COMMON.evaluate(model, eval_tokens, [int(value) for value in lock["eval_indices"]], int(lock["sequence_length"]))
    evaluation_zero = None
    calls_before_zero = calls_after_zero = None
    if replacement is not None and args.candidate in {"qh026", "cc026"}:
        calls_before_zero = replacement.core.circuit_calls
        replacement.set_residual_enabled(False)
        evaluation_zero = COMMON.evaluate(model, eval_tokens, [int(value) for value in lock["eval_indices"]], int(lock["sequence_length"]))
        calls_after_zero = replacement.core.circuit_calls
        if calls_after_zero != calls_before_zero:
            raise AssertionError("zero branch executed residual core")
        replacement.set_residual_enabled(True)
    run_id = f"c4-{args.candidate}-s{lock['seed']}-n{lock['train_count']}-e{lock['eval_count']}"
    payload = {
        "status": "ok",
        "run_id": run_id,
        "candidate": args.candidate,
        "dataset_lock": str(LOCK),
        "dataset_lock_sha256": sha256(LOCK),
        "dataset": {
            "training": lock["training_dataset"],
            "evaluation": lock["evaluation_dataset"],
            "sequence_length": lock["sequence_length"],
            "train_indices": lock["train_indices"] if training["steps"] else [],
            "eval_indices": lock["eval_indices"],
            "train_tokens_sha256": sha256(TRAIN_TOKENS),
            "eval_tokens_sha256": sha256(EVAL_TOKENS),
            "test_rows_inspected_or_used_for_fitness": 0,
        },
        "injection": injection,
        "initialization": initialization,
        "training": training,
        "evaluation_active": evaluation,
        "evaluation_zero_residual": evaluation_zero,
        "simulator_calls_before_zero_evaluation": calls_before_zero,
        "simulator_calls_after_zero_evaluation": calls_after_zero,
        "replacement_audit_after_teacher_strip": replacement.audit() if replacement else None,
        "checkpoint": str(checkpoint_path) if checkpoint_path.exists() else None,
        "gpu_peak_allocated_bytes": [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = OUTPUT_DIR / f"{run_id}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "run_id": run_id,
        "net_model_parameter_reduction": injection["net_model_parameter_reduction"],
        "trainable_parameters": injection["trainable_parameter_count"],
        "training_summary": {key: value for key, value in training.items() if key != "losses"},
        "evaluation_active": evaluation["metrics"],
        "evaluation_zero_residual": evaluation_zero["metrics"] if evaluation_zero else None,
        "record": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
