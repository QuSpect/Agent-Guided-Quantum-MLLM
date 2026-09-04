#!/usr/bin/env python3
"""Locked QH031 shared-basis training and paired C4 evaluation."""

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
LOCK = PROJECT / "artifacts/qh031-shared-basis-lock.json"
TRAIN_TOKENS = PROJECT / "datasets/processed/wikitext2_qwen38/train-tokens.pt"
EVAL_TOKENS = PROJECT / "datasets/processed/c4_validation_qwen38/validation-tokens.pt"
OUTPUT_DIR = PROJECT / "artifacts/c4_qh031"
sys.path.insert(0, str(PROJECT / "code/src"))

from quantum_qwen38.qh031_shared_basis_replacement import (  # noqa: E402
    DequantizedAuditCore,
    ExactQuantumCore,
    QH031Config,
    install_qh031,
    unique_trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        choices=["base", "sharedsvd1024", "qh031", "cc031"],
        required=True,
    )
    parser.add_argument("--smoke", action="store_true")
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


def token_block(tokens: torch.Tensor, index: int, length: int, device) -> torch.Tensor:
    start = index * length
    return tokens[start : start + length].to(device=device, dtype=torch.long).unsqueeze(0)


def set_projections(model, modules) -> None:
    for layer_index, module in modules.items():
        model.model.language_model.layers[layer_index].self_attn.v_proj = module


def distill_end_to_end(
    model,
    replacements,
    originals,
    tokens: torch.Tensor,
    lock: dict,
    *,
    smoke: bool,
) -> dict:
    protocol = lock["training_protocol"]
    parameters = list(unique_trainable_parameters(replacements))
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
    core = replacements[int(lock["architecture"]["layer_indices"][0])].core
    simulator_before = int(core.circuit_calls.item())
    losses, gradient_norms, theta_gradient_norms, gamma_gradient_norms, seconds = (
        [], [], [], [], []
    )
    for index in indices:
        input_ids = token_block(
            tokens, index, int(lock["sequence_length"]), model.device
        )
        set_projections(model, originals)
        with torch.inference_mode():
            teacher_hidden = backbone(
                input_ids=input_ids, use_cache=False, return_dict=True
            ).last_hidden_state.detach()
        set_projections(model, replacements)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        student_hidden = backbone(
            input_ids=input_ids, use_cache=False, return_dict=True
        ).last_hidden_state
        denominator = teacher_hidden.float().square().mean().clamp_min(1.0e-8)
        loss = (
            (student_hidden.float() - teacher_hidden.float()).square().mean()
            / denominator
        )
        loss.backward()
        theta_gradient_norms.append(float(core.theta.grad.float().norm()))
        gamma_gradient_norms.append(float(torch.stack([
            replacement.gamma.grad.float().abs()
            for replacement in replacements.values()
        ]).norm()))
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(protocol["gradient_clip_norm"])
        )
        optimizer.step()
        torch.cuda.synchronize(core.theta.device)
        losses.append(float(loss.detach()))
        gradient_norms.append(float(gradient_norm.detach()))
        seconds.append(time.perf_counter() - started)
        if not smoke and len(losses) % 64 == 0:
            print(json.dumps({
                "candidate_progress": len(losses),
                "total": len(indices),
                "recent_loss_mean": float(np.mean(losses[-64:])),
                "recent_step_seconds": float(np.mean(seconds[-64:])),
                "theta_norm": float(core.theta.detach().float().norm()),
            }), flush=True)
        del input_ids, teacher_hidden, student_hidden, denominator, loss, gradient_norm
    simulator_after = int(core.circuit_calls.item())
    expected_calls = (
        len(indices) * len(replacements) if isinstance(core, ExactQuantumCore) else 0
    )
    if simulator_after - simulator_before != expected_calls:
        raise AssertionError(
            f"simulator call mismatch: observed={simulator_after - simulator_before}, "
            f"expected={expected_calls}"
        )
    frozen_grad_tensors = sum(
        int(parameter.grad is not None)
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    if frozen_grad_tensors:
        raise AssertionError("a frozen model/shared-basis parameter received gradients")
    edge = min(32, len(losses))
    return {
        "steps": len(losses),
        "objective": protocol["objective"],
        "learning_rate": float(protocol["learning_rate"]),
        "loss_first_window_mean": float(np.mean(losses[:edge])),
        "loss_last_window_mean": float(np.mean(losses[-edge:])),
        "loss_min": float(np.min(losses)),
        "gradient_norm_mean": float(np.mean(gradient_norms)),
        "theta_gradient_norm_mean": float(np.mean(theta_gradient_norms)),
        "gamma_gradient_norm_mean": float(np.mean(gamma_gradient_norms)),
        "mean_student_backbone_step_seconds": float(np.mean(seconds)),
        "total_student_backbone_step_seconds": float(np.sum(seconds)),
        "simulator_calls_during_student_training": simulator_after - simulator_before,
        "frozen_parameter_grad_tensor_count": frozen_grad_tensors,
        "final_theta_l2_norm": float(core.theta.detach().float().norm()),
        "final_gammas": {
            str(index): float(replacement.gamma.detach())
            for index, replacement in replacements.items()
        },
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


def unique_checkpoint(replacements) -> dict:
    first_index = sorted(replacements)[0]
    return {
        "core.theta": replacements[first_index].core.theta.detach().cpu(),
        **{
            f"gamma.{index}": replacement.gamma.detach().cpu()
            for index, replacement in replacements.items()
        },
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("QH031 is GPU-only")
    if args.smoke and args.candidate not in {"qh031", "cc031"}:
        raise ValueError("smoke is only supported for trainable candidates")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "locked_before_qh031_qh_cc_base_performance_results":
        raise RuntimeError("invalid QH031 lock")
    if lock["test_rows_inspected_or_used_for_fitness"] != 0:
        raise RuntimeError("test leakage disclosure is nonzero")
    seed_everything(int(lock["seed"]))
    train_tokens = torch.load(TRAIN_TOKENS, map_location="cpu", weights_only=True)
    eval_tokens = None if args.smoke else torch.load(
        EVAL_TOKENS, map_location="cpu", weights_only=True
    )
    model_dir = Path((PROJECT / "records/active_model_path.txt").read_text().strip())
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = False
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    replacements = originals = None
    if args.candidate == "base":
        model.requires_grad_(False)
        injection = {
            "candidate": "frozen-dense-base",
            "base_model_parameters": base_parameters,
            "deployed_model_parameters": base_parameters,
            "net_model_parameter_reduction": 0,
            "trainable_parameter_count": 0,
        }
        training = {"steps": 0}
    else:
        branch = "classical" if args.candidate == "cc031" else "quantum"
        config = QH031Config()
        replacements, originals, injection = install_qh031(
            model, config, branch=branch
        )
        injection["candidate"] = {
            "sharedsvd1024": "shared-SVD-1024-control",
            "qh031": "QH-031",
            "cc031": "CC-031",
        }[args.candidate]
        if args.candidate == "sharedsvd1024":
            for replacement in replacements.values():
                replacement.branch_enabled = False
                replacement.gamma.requires_grad_(False)
                replacement.core.theta.requires_grad_(False)
            training = {"steps": 0, "role": "shared-SVD zero-branch control"}
            injection["observed_trainable_parameter_count"] = 0
            injection["trainable_parameter_count"] = 0
        else:
            training = distill_end_to_end(
                model,
                replacements,
                originals,
                train_tokens,
                lock,
                smoke=args.smoke,
            )
        injection.setdefault(
            "trainable_parameter_count", injection["observed_trainable_parameter_count"]
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        payload = {
            "status": "ok",
            "mode": "fresh-lock one-step train-only integration smoke",
            "candidate": args.candidate,
            "lock_sha256": sha256(LOCK),
            "training": training,
            "injection": injection,
            "holdout_rows_inspected": 0,
            "test_rows_inspected": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path = OUTPUT_DIR / f"{args.candidate}-fresh-lock-smoke.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    checkpoint_path = OUTPUT_DIR / (
        f"c4-{args.candidate}-s{lock['seed']}-replacement.pt"
    )
    if replacements is not None and args.candidate in {"qh031", "cc031"}:
        torch.save(unique_checkpoint(replacements), checkpoint_path)
    eval_indices = [int(value) for value in lock["eval_indices"]]
    length = int(lock["sequence_length"])
    evaluation_active = evaluate(model, eval_tokens, eval_indices, length)
    evaluation_zero = evaluation_no_ent = evaluation_dq = None
    intervention_calls = {}
    if replacements is not None and args.candidate in {"qh031", "cc031"}:
        core = replacements[config.layer_indices[0]].core
        before = int(core.circuit_calls.item())
        for replacement in replacements.values():
            replacement.branch_enabled = False
        evaluation_zero = evaluate(model, eval_tokens, eval_indices, length)
        after = int(core.circuit_calls.item())
        if after != before:
            raise AssertionError("own-zero evaluation executed the active core")
        intervention_calls["own_zero"] = after - before
        for replacement in replacements.values():
            replacement.branch_enabled = True

        if args.candidate == "qh031":
            before = int(core.circuit_calls.item())
            core.use_entanglers = False
            evaluation_no_ent = evaluate(model, eval_tokens, eval_indices, length)
            core.use_entanglers = True
            after = int(core.circuit_calls.item())
            intervention_calls["no_entanglement"] = after - before
            expected = len(eval_indices) * len(replacements)
            if after - before != expected:
                raise AssertionError("no-entanglement simulator call count mismatch")

            dq_core = DequantizedAuditCore(config).to(core.theta.device)
            dq_core.load_state_dict(core.state_dict(), strict=True)
            for replacement in replacements.values():
                replacement.core = dq_core
            evaluation_dq = evaluate(model, eval_tokens, eval_indices, length)
            dq_calls = int(dq_core.circuit_calls.item())
            intervention_calls["dequantized_audit"] = dq_calls
            if dq_calls != expected:
                raise AssertionError("dequantized audit call count mismatch")
            for replacement in replacements.values():
                replacement.core = core

    run_id = (
        f"c4-{args.candidate}-s{lock['seed']}-"
        f"n{lock['train_count']}-e{lock['eval_count']}"
    )
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
            "sequence_length": length,
            "train_indices": lock["train_indices"] if training["steps"] else [],
            "eval_indices": eval_indices,
            "train_tokens_sha256": sha256(TRAIN_TOKENS),
            "eval_tokens_sha256": sha256(EVAL_TOKENS),
            "test_rows_inspected_or_used_for_fitness": 0,
        },
        "injection": injection,
        "training": training,
        "evaluation_active": evaluation_active,
        "evaluation_zero_shared_svd": evaluation_zero,
        "evaluation_no_entanglement": evaluation_no_ent,
        "evaluation_dequantized_audit": evaluation_dq,
        "intervention_simulator_calls": intervention_calls,
        "checkpoint": str(checkpoint_path) if checkpoint_path.exists() else None,
        "teacher_dense_modules_retained_in_deployed_model": False,
        "gpu_peak_allocated_bytes": [
            torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
        ],
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
        "evaluation_active": evaluation_active["metrics"],
        "evaluation_zero": evaluation_zero["metrics"] if evaluation_zero else None,
        "evaluation_no_entanglement": evaluation_no_ent["metrics"] if evaluation_no_ent else None,
        "evaluation_dequantized": evaluation_dq["metrics"] if evaluation_dq else None,
        "record": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()

