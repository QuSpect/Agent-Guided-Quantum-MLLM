#!/usr/bin/env python3
"""Locked end-to-end hidden-state distillation and C4 evaluation for QH-025."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM


PROJECT = Path(__file__).resolve().parents[2]
LOCK = PROJECT / "artifacts/wikitext-qh025-end-to-end-lock.json"
TRAIN_TOKENS = PROJECT / "datasets/processed/wikitext2_qwen38/train-tokens.pt"
EVAL_TOKENS = PROJECT / "datasets/processed/c4_validation_qwen38/validation-tokens.pt"
QH024_DIR = PROJECT / "artifacts/wikitext_qh024"
OUTPUT_DIR = PROJECT / "artifacts/c4_qh025"
sys.path.insert(0, str(PROJECT / "code/src"))

from quantum_qwen38.hyqut_replacement import QH024Config, freeze_and_replace_qh024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=["base", "svd512", "qh025", "cc025"], required=True)
    parser.add_argument("--smoke", action="store_true", help="one train step, no holdout evaluation")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def token_block(tokens: torch.Tensor, index: int, length: int, device: torch.device) -> torch.Tensor:
    start = index * length
    return tokens[start : start + length].to(device=device, dtype=torch.long).unsqueeze(0)


def load_qh024_initialization(replacement, candidate: str, lock: dict) -> dict:
    key = "qh025_from" if candidate == "qh025" else "cc025_from"
    hash_key = f"{key}_sha256"
    path = Path(lock["initialization"][key])
    if sha256(path) != lock["initialization"][hash_key]:
        raise RuntimeError(f"{candidate} initialization hash mismatch")
    state = torch.load(path, map_location=replacement.gamma.device, weights_only=True)
    incompatible = replacement.load_state_dict(state, strict=False)
    allowed_missing = {"svd_down.weight", "svd_up.weight"}
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected initialization keys: {incompatible}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "missing_frozen_svd_keys_recomputed_from_pinned_base": sorted(allowed_missing),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def distill_end_to_end(model, replacement, tokens: torch.Tensor, lock: dict, *, smoke: bool) -> dict:
    protocol = lock["training_protocol"]
    parameters = [parameter for parameter in replacement.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    model.eval()
    model.config.use_cache = False
    backbone = model.model.language_model
    indices = [int(value) for value in lock["train_indices"]]
    if smoke:
        indices = indices[:1]
    losses, gradient_norms, seconds = [], [], []
    simulator_before = int(replacement.core.circuit_calls)
    for index in indices:
        input_ids = token_block(tokens, index, int(lock["sequence_length"]), model.device)
        replacement.set_capture_teacher_only(True)
        with torch.inference_mode():
            teacher_output = backbone(input_ids=input_ids, use_cache=False, return_dict=True)
            teacher_hidden = teacher_output.last_hidden_state.detach()
        captured_input, captured_projection = replacement.take_capture()
        del captured_input, captured_projection, teacher_output
        replacement.set_capture_teacher_only(False)

        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        student_output = backbone(input_ids=input_ids, use_cache=False, return_dict=True)
        student_hidden = student_output.last_hidden_state
        denominator = teacher_hidden.float().square().mean().clamp_min(1.0e-8)
        loss = (student_hidden.float() - teacher_hidden.float()).square().mean() / denominator
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(protocol["gradient_clip_norm"])
        )
        optimizer.step()
        torch.cuda.synchronize(replacement.gamma.device)
        losses.append(float(loss.detach()))
        gradient_norms.append(float(gradient_norm.detach()))
        seconds.append(time.perf_counter() - started)
        if not smoke and len(losses) % 64 == 0:
            print(json.dumps({
                "progress_step": len(losses),
                "progress_total": len(indices),
                "recent_end_to_end_loss_mean": float(np.mean(losses[-64:])),
                "recent_student_seconds_mean": float(np.mean(seconds[-64:])),
            }), flush=True)
        del (
            input_ids, teacher_hidden, student_output, student_hidden,
            denominator, loss, gradient_norm,
        )
    simulator_after = int(replacement.core.circuit_calls)
    expected_calls = len(indices) if replacement.inference_simulator_required else 0
    if simulator_after - simulator_before != expected_calls:
        raise AssertionError("unexpected end-to-end simulator call count")
    edge = min(32, len(losses))
    return {
        "steps": len(losses),
        "objective": protocol["objective"],
        "learning_rate": float(protocol["learning_rate"]),
        "loss_first_window_mean": float(np.mean(losses[:edge])),
        "loss_last_window_mean": float(np.mean(losses[-edge:])),
        "loss_min": float(np.min(losses)),
        "gradient_norm_mean": float(np.mean(gradient_norms)),
        "mean_student_backbone_step_seconds": float(np.mean(seconds)),
        "total_student_backbone_step_seconds": float(np.sum(seconds)),
        "simulator_calls_during_student_training": simulator_after - simulator_before,
        "frozen_parameter_grad_tensor_count": sum(
            parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad
        ),
        "losses": losses,
    }


@torch.inference_mode()
def evaluate(model, tokens: torch.Tensor, indices: list[int], length: int) -> dict:
    model.eval()
    rows, seconds = [], []
    for index in indices:
        input_ids = token_block(tokens, index, length, model.device)
        started = time.perf_counter()
        loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        elapsed = time.perf_counter() - started
        rows.append({
            "block_index": index,
            "mean_token_nll": float(loss),
            "predicted_tokens": length - 1,
            "seconds": elapsed,
        })
        seconds.append(elapsed)
    nll = float(np.mean([row["mean_token_nll"] for row in rows]))
    return {
        "metrics": {
            "sequence_count": len(rows),
            "predicted_token_count": len(rows) * (length - 1),
            "mean_token_nll": nll,
            "token_perplexity": math.exp(min(50.0, nll)),
            "mean_sequence_seconds": float(np.mean(seconds)),
            "p95_sequence_seconds": float(np.quantile(seconds, 0.95)),
        },
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("QH-025 is GPU-only")
    if args.smoke and args.candidate not in {"qh025", "cc025"}:
        raise ValueError("smoke mode is only for trainable candidates")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "locked_before_qh025_qh_cc_base_performance_results":
        raise RuntimeError("invalid QH-025 lock")
    if lock["test_rows_inspected_or_used_for_fitness"] != 0:
        raise RuntimeError("test leakage disclosure is nonzero")
    seed_everything(int(lock["seed"]))
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
        core_kind = "classical" if args.candidate == "cc025" else "quantum"
        injection = freeze_and_replace_qh024(model, QH024Config(), core_kind=core_kind)
        injection["candidate"] = {
            "svd512": "SVD-512-C4-control",
            "qh025": "QH-025",
            "cc025": "CC-025",
        }[args.candidate]
        replacement = model.model.language_model.layers[7].self_attn.v_proj
        if args.candidate == "svd512":
            replacement.set_residual_enabled(False)
            training = {"steps": 0, "role": "causal zero-residual compression control"}
        else:
            initialization = load_qh024_initialization(replacement, args.candidate, lock)
            training = distill_end_to_end(
                model, replacement, train_tokens, lock, smoke=args.smoke
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        payload = {
            "status": "ok",
            "mode": "one-step training-data-only smoke",
            "candidate": args.candidate,
            "lock_sha256": sha256(LOCK),
            "training": training,
            "injection": injection,
            "initialization": initialization,
            "holdout_rows_inspected": 0,
            "test_rows_inspected": 0,
            "gpu_peak_allocated_bytes": [
                torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path = OUTPUT_DIR / f"{args.candidate}-end-to-end-smoke.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    checkpoint_path = OUTPUT_DIR / f"c4-{args.candidate}-s{lock['seed']}-replacement.pt"
    if replacement is not None and args.candidate in {"qh025", "cc025"}:
        torch.save(
            {
                name: parameter.detach().cpu()
                for name, parameter in replacement.named_parameters()
                if parameter.requires_grad
            },
            checkpoint_path,
        )
    if replacement is not None:
        replacement.strip_teacher()
    evaluation = evaluate(
        model,
        eval_tokens,
        [int(value) for value in lock["eval_indices"]],
        int(lock["sequence_length"]),
    )
    evaluation_zero = None
    calls_before_zero = calls_after_zero = None
    if replacement is not None and args.candidate in {"qh025", "cc025"}:
        calls_before_zero = replacement.core.circuit_calls
        replacement.set_residual_enabled(False)
        evaluation_zero = evaluate(
            model,
            eval_tokens,
            [int(value) for value in lock["eval_indices"]],
            int(lock["sequence_length"]),
        )
        calls_after_zero = replacement.core.circuit_calls
        if calls_after_zero != calls_before_zero:
            raise AssertionError("zero branch executed residual core")
        replacement.set_residual_enabled(True)
    run_id = f"c4-{args.candidate}-s{lock['seed']}-n{lock['train_count']}-e{lock['eval_count']}"
    payload = {
        "status": "ok",
        "run_id": run_id,
        "candidate": args.candidate,
        "model_dir": str(model_dir),
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
        "gpu_peak_allocated_bytes": [
            torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())
        ],
        "checked_at": datetime.now(timezone.utc).isoformat(),
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
