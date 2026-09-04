#!/usr/bin/env python3
"""Locked WikiText distillation/evaluation for QH-024 and controls."""

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
LOCK = PROJECT / "artifacts/wikitext-qh024-distillation-lock.json"
TRAIN_TOKENS = PROJECT / "datasets/processed/wikitext2_qwen38/train-tokens.pt"
VALIDATION_TOKENS = PROJECT / "datasets/processed/wikitext2_qwen38/validation-tokens.pt"
OUTPUT_DIR = PROJECT / "artifacts/wikitext_qh024"
sys.path.insert(0, str(PROJECT / "code/src"))

from quantum_qwen38.hyqut_replacement import QH024Config, freeze_and_replace_qh024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=["base", "svd512", "qh024", "cc024"], required=True)
    parser.add_argument("--timing-context", choices=["exclusive", "shared"], default="shared")
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


def distill(model, replacement, tokens: torch.Tensor, lock: dict) -> dict:
    protocol = lock["training_protocol"]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in replacement.parameters() if parameter.requires_grad],
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    losses, gradient_norms, seconds = [], [], []
    model.eval()
    model.config.use_cache = False
    order = list(int(value) for value in lock["train_indices"])
    random.Random(int(lock["seed"])).shuffle(order)
    for index in order:
        input_ids = token_block(tokens, index, int(lock["sequence_length"]), model.device)
        replacement.set_capture_teacher_only(True)
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
        hidden_states, teacher_target = replacement.take_capture()
        replacement.set_capture_teacher_only(False)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        student = replacement.student_forward(hidden_states)
        denominator = teacher_target.float().square().mean().clamp_min(1.0e-8)
        loss = (student.float() - teacher_target.float()).square().mean() / denominator
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in replacement.parameters() if parameter.requires_grad],
            float(protocol["gradient_clip_norm"]),
        )
        optimizer.step()
        torch.cuda.synchronize(replacement.gamma.device)
        losses.append(float(loss.detach()))
        gradient_norms.append(float(gradient_norm))
        seconds.append(time.perf_counter() - started)
        del input_ids, hidden_states, teacher_target, student, denominator, loss
    return {
        "steps": len(losses),
        "objective": protocol["objective"],
        "learning_rate": float(protocol["learning_rate"]),
        "loss_first_16_mean": float(np.mean(losses[:16])),
        "loss_last_16_mean": float(np.mean(losses[-16:])),
        "loss_min": float(np.min(losses)),
        "gradient_norm_mean": float(np.mean(gradient_norms)),
        "mean_residual_step_seconds_excluding_teacher_forward": float(np.mean(seconds)),
        "total_residual_step_seconds_excluding_teacher_forward": float(np.sum(seconds)),
        "losses": losses,
        "frozen_parameter_grad_tensor_count": sum(
            parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad
        ),
    }


@torch.inference_mode()
def evaluate(model, tokens: torch.Tensor, indices: list[int], length: int) -> dict:
    model.eval()
    rows, seconds = [], []
    for index in indices:
        input_ids = token_block(tokens, index, length, model.device)
        started = time.perf_counter()
        loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        value = float(loss)
        elapsed = time.perf_counter() - started
        rows.append({
            "block_index": index,
            "mean_token_nll": value,
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
        raise RuntimeError("QH-024 formal runner is GPU-only")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "locked_before_qh024_qh_cc_base_performance_results":
        raise RuntimeError("invalid QH-024 lock")
    if lock["test_rows_inspected_or_used_for_fitness"] != 0:
        raise RuntimeError("test leakage disclosure is nonzero")
    seed_everything(int(lock["seed"]))
    train_tokens = torch.load(TRAIN_TOKENS, map_location="cpu", weights_only=True)
    validation_tokens = torch.load(VALIDATION_TOKENS, map_location="cpu", weights_only=True)
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
        kind = "classical" if args.candidate == "cc024" else "quantum"
        injection = freeze_and_replace_qh024(model, QH024Config(), core_kind=kind)
        replacement = model.model.language_model.layers[7].self_attn.v_proj
        if args.candidate == "svd512":
            replacement.set_residual_enabled(False)
            training = {"steps": 0, "role": "untrained causal zero-residual control"}
        else:
            training = distill(model, replacement, train_tokens, lock)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / f"wikitext-{args.candidate}-s{lock['seed']}-replacement.pt"
    if replacement is not None and args.candidate in {"qh024", "cc024"}:
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
    eval_indices = [int(value) for value in lock["eval_indices"]]
    evaluation = evaluate(model, validation_tokens, eval_indices, int(lock["sequence_length"]))
    evaluation_zero = None
    simulator_calls_before_zero = None
    simulator_calls_after_zero = None
    if replacement is not None and args.candidate in {"qh024", "cc024"}:
        simulator_calls_before_zero = replacement.core.circuit_calls
        replacement.set_residual_enabled(False)
        evaluation_zero = evaluate(model, validation_tokens, eval_indices, int(lock["sequence_length"]))
        simulator_calls_after_zero = replacement.core.circuit_calls
        replacement.set_residual_enabled(True)
        if simulator_calls_after_zero != simulator_calls_before_zero:
            raise AssertionError("zero branch unexpectedly executed the residual core")
    run_id = f"wikitext-{args.candidate}-s{lock['seed']}-n{lock['train_count']}-e{lock['eval_count']}"
    payload = {
        "status": "ok",
        "run_id": run_id,
        "candidate": args.candidate,
        "model_dir": str(model_dir),
        "timing_context": args.timing_context,
        "dataset_lock": str(LOCK),
        "dataset_lock_sha256": sha256(LOCK),
        "dataset": {
            "name": lock["dataset"],
            "source_revision": lock["source_revision"],
            "license": lock["license"],
            "sequence_length": lock["sequence_length"],
            "train_indices": lock["train_indices"] if training["steps"] else [],
            "eval_indices": eval_indices,
            "train_tokens_sha256": sha256(TRAIN_TOKENS),
            "validation_tokens_sha256": sha256(VALIDATION_TOKENS),
            "test_rows_inspected_or_used_for_fitness": 0,
        },
        "injection": injection,
        "training": training,
        "evaluation_active": evaluation,
        "evaluation_zero_residual": evaluation_zero,
        "simulator_calls_before_zero_evaluation": simulator_calls_before_zero,
        "simulator_calls_after_zero_evaluation": simulator_calls_after_zero,
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
