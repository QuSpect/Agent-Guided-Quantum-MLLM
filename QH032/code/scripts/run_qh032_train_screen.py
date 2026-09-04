#!/usr/bin/env python3
"""Run one arm of the locked WikiText-train QH032 screen."""

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
LOCK = PROJECT / "artifacts/qh032-train-screen-lock.json"
TOKENS = PROJECT / "datasets/processed/wikitext2_qwen38/train-tokens.pt"
OUTPUT_DIR = PROJECT / "artifacts/qh032_train_screen"
sys.path.insert(0, str(PROJECT / "code/src"))
from quantum_qwen38.qh032_pauli_observable_replacement import (  # noqa: E402
    ExactPauliObservableCore, QH032Config, install_qh032, unique_trainable_parameters,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block(tokens, index, length, device):
    start = index * length
    return tokens[start : start + length].to(device=device, dtype=torch.long).unsqueeze(0)


def set_projections(model, modules) -> None:
    for index, module in modules.items():
        model.model.language_model.layers[index].self_attn.v_proj = module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=["base", "sharedsvd1024", "qh032", "cc032"], required=True)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model, tokens, indices, length):
    rows, seconds = [], []
    for index in indices:
        inputs = block(tokens, index, length, model.device)
        started = time.perf_counter()
        loss = model(input_ids=inputs, labels=inputs, use_cache=False).loss
        rows.append({"block_index": index, "mean_token_nll": float(loss), "seconds": time.perf_counter() - started})
        seconds.append(rows[-1]["seconds"])
    nll = float(np.mean([row["mean_token_nll"] for row in rows]))
    return {
        "metrics": {"sequence_count": len(rows), "mean_token_nll": nll, "token_perplexity": math.exp(nll),
                    "mean_sequence_seconds": float(np.mean(seconds))},
        "rows": rows,
    }


def train(model, replacements, originals, tokens, lock):
    parameters = list(unique_trainable_parameters(replacements))
    protocol = lock["training_protocol"]
    optimizer = torch.optim.AdamW(parameters, lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
    backbone = model.model.language_model
    config = QH032Config()
    core = replacements[config.layer_indices[0]].core
    calls_before = int(core.circuit_calls.item())
    losses, seconds, theta_grads, alpha_grads = [], [], [], []
    for position, index in enumerate(lock["train_indices"], start=1):
        inputs = block(tokens, int(index), int(lock["sequence_length"]), model.device)
        set_projections(model, originals)
        with torch.inference_mode():
            teacher = backbone(input_ids=inputs, use_cache=False, return_dict=True).last_hidden_state.detach()
        set_projections(model, replacements)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        student = backbone(input_ids=inputs, use_cache=False, return_dict=True).last_hidden_state
        denominator = teacher.float().square().mean().clamp_min(1.0e-8)
        loss = (student.float() - teacher.float()).square().mean() / denominator
        loss.backward()
        theta_grads.append(float(core.theta.grad.float().norm()))
        alpha_grads.append(float(core.alpha.grad.float().norm()))
        torch.nn.utils.clip_grad_norm_(parameters, protocol["gradient_clip_norm"])
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        seconds.append(time.perf_counter() - started)
        if position % 64 == 0:
            print(json.dumps({"trained": position, "recent_loss": float(np.mean(losses[-64:])), "recent_seconds": float(np.mean(seconds[-64:]))}), flush=True)
    calls = int(core.circuit_calls.item()) - calls_before
    expected = len(lock["train_indices"]) * len(config.layer_indices) if isinstance(core, ExactPauliObservableCore) else 0
    if calls != expected:
        raise AssertionError(f"training simulator calls {calls} != {expected}")
    frozen_grads = sum(int(p.grad is not None) for p in model.parameters() if not p.requires_grad)
    if frozen_grads:
        raise AssertionError("frozen parameter received gradient")
    edge = 32
    return {
        "steps": len(losses), "loss_first_window_mean": float(np.mean(losses[:edge])),
        "loss_last_window_mean": float(np.mean(losses[-edge:])), "loss_min": float(np.min(losses)),
        "theta_gradient_norm_mean": float(np.mean(theta_grads)), "alpha_gradient_norm_mean": float(np.mean(alpha_grads)),
        "mean_student_step_seconds": float(np.mean(seconds)), "simulator_calls": calls,
        "frozen_gradient_tensor_count": frozen_grads, "final_theta_l2_norm": float(core.theta.detach().norm()),
        "final_alpha_l2_norm": float(core.alpha.detach().norm()),
        "final_gammas": {str(i): float(m.gamma.detach()) for i, m in replacements.items()},
        "losses": losses,
    }


def checkpoint(replacements):
    first = sorted(replacements)[0]
    return {
        "core.theta": replacements[first].core.theta.detach().cpu(),
        "core.alpha": replacements[first].core.alpha.detach().cpu(),
        **{f"gamma.{index}": module.gamma.detach().cpu() for index, module in replacements.items()},
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("QH032 screen is GPU-only")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "locked_before_qh032_train_only_screen_results":
        raise RuntimeError("invalid QH032 train-screen lock")
    random.seed(lock["seed"]); np.random.seed(lock["seed"]); torch.manual_seed(lock["seed"]); torch.cuda.manual_seed_all(lock["seed"])
    tokens = torch.load(TOKENS, map_location="cpu", weights_only=True)
    model_dir = Path((PROJECT / "records/active_model_path.txt").read_text().strip())
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16, device_map="balanced",
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    )
    model.eval(); model.config.use_cache = False
    replacements = originals = None
    if args.candidate == "base":
        model.requires_grad_(False)
        audit = {"candidate": "base", "net_model_parameter_reduction": 0, "trainable_parameter_count": 0}
        training = {"steps": 0}
    else:
        branch = "classical" if args.candidate == "cc032" else "quantum"
        replacements, originals, audit = install_qh032(model, QH032Config(), branch=branch)
        if args.candidate == "sharedsvd1024":
            for module in replacements.values():
                module.branch_enabled = False; module.requires_grad_(False)
            training = {"steps": 0}
        else:
            training = train(model, replacements, originals, tokens, lock)
    evaluation = evaluate(model, tokens, [int(i) for i in lock["eval_indices"]], int(lock["sequence_length"]))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if replacements is not None and args.candidate in {"qh032", "cc032"}:
        torch.save(checkpoint(replacements), OUTPUT_DIR / f"{args.candidate}-screen-checkpoint.pt")
    payload = {
        "status": "ok", "candidate": args.candidate, "dataset_lock": str(LOCK), "dataset_lock_sha256": sha256(LOCK),
        "dataset": {"name": lock["dataset"], "train_indices": lock["train_indices"] if training["steps"] else [],
                    "eval_indices": lock["eval_indices"], "validation_rows_inspected": 0, "test_rows_inspected": 0},
        "injection": audit, "training": training, "evaluation": evaluation,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = OUTPUT_DIR / f"screen-{args.candidate}-s{lock['seed']}-n{lock['train_count']}-e{lock['eval_count']}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "candidate": args.candidate, "nll": evaluation["metrics"]["mean_token_nll"],
                      "output": str(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()

