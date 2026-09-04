#!/usr/bin/env python3
"""Evaluate frozen QH031/CC031 checkpoints on the locked confirmation blocks."""

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
LOCK = PROJECT / "artifacts/qh031-confirmation-lock.json"
TOKENS = PROJECT / "datasets/processed/c4_validation_qwen38/validation-tokens.pt"
OUTPUT_DIR = PROJECT / "artifacts/c4_qh031_confirmation"
sys.path.insert(0, str(PROJECT / "code/src"))

from quantum_qwen38.qh031_shared_basis_replacement import (  # noqa: E402
    QH031Config,
    install_qh031,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", choices=["base", "sharedsvd1024", "qh031", "cc031"], required=True
    )
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


@torch.inference_mode()
def evaluate(model, tokens: torch.Tensor, indices: list[int], length: int) -> dict:
    model.eval()
    rows, seconds = [], []
    for position, index in enumerate(indices, start=1):
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
        if position % 64 == 0:
            print(json.dumps({
                "evaluated": position,
                "total": len(indices),
                "running_mean_nll": float(np.mean([row["mean_token_nll"] for row in rows])),
                "recent_mean_seconds": float(np.mean(seconds[-64:])),
            }), flush=True)
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


def load_checkpoint(replacements, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    first = sorted(replacements)[0]
    with torch.no_grad():
        replacements[first].core.theta.copy_(
            state["core.theta"].to(replacements[first].core.theta.device)
        )
        for index, replacement in replacements.items():
            replacement.gamma.copy_(state[f"gamma.{index}"].to(replacement.gamma.device))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("QH031 confirmation is GPU-only")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "locked_before_qh031_confirmation_results":
        raise RuntimeError("invalid confirmation lock")
    if lock["test_rows_inspected_or_used_for_fitness"] != 0:
        raise RuntimeError("test leakage disclosure is nonzero")
    seed_everything(int(lock["seed"]))
    tokens = torch.load(TOKENS, map_location="cpu", weights_only=True)
    if sha256(TOKENS) != lock["evaluation_dataset"]["tokens_sha256"]:
        raise RuntimeError("C4 token file changed after lock")
    model_dir = Path((PROJECT / "records/active_model_path.txt").read_text().strip())
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16,
        device_map="balanced", low_cpu_mem_usage=True, attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = False
    model.requires_grad_(False)
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    replacements = None
    injection = {
        "candidate": args.candidate,
        "base_model_parameters": base_parameters,
        "deployed_model_parameters": base_parameters,
        "net_model_parameter_reduction": 0,
        "trainable_parameter_count_during_confirmation": 0,
    }
    if args.candidate != "base":
        branch = "classical" if args.candidate == "cc031" else "quantum"
        config = QH031Config()
        replacements, _, injection = install_qh031(model, config, branch=branch)
        for replacement in replacements.values():
            replacement.requires_grad_(False)
        if args.candidate == "sharedsvd1024":
            for replacement in replacements.values():
                replacement.branch_enabled = False
        else:
            checkpoint_key = f"{args.candidate}_checkpoint"
            checkpoint = Path(lock["frozen_inputs"][checkpoint_key])
            if sha256(checkpoint) != lock["frozen_inputs"][f"{checkpoint_key}_sha256"]:
                raise RuntimeError(f"{args.candidate} checkpoint changed after lock")
            load_checkpoint(replacements, checkpoint)
        injection["trainable_parameter_count_during_confirmation"] = 0

    indices = [int(value) for value in lock["eval_indices"]]
    length = int(lock["sequence_length"])
    evaluation_active = evaluate(model, tokens, indices, length)
    evaluation_no_ent = None
    simulator_calls = 0
    if args.candidate == "qh031":
        core = replacements[config.layer_indices[0]].core
        simulator_calls = int(core.circuit_calls.item())
        expected = len(indices) * len(replacements)
        if simulator_calls != expected:
            raise AssertionError(f"active simulator calls {simulator_calls} != {expected}")
        core.use_entanglers = False
        before = int(core.circuit_calls.item())
        evaluation_no_ent = evaluate(model, tokens, indices, length)
        after = int(core.circuit_calls.item())
        if after - before != expected:
            raise AssertionError("no-entanglement simulator call count mismatch")
        simulator_calls = after
    elif args.candidate == "cc031":
        core = replacements[config.layer_indices[0]].core
        if int(core.circuit_calls.item()) != 0:
            raise AssertionError("classical control executed simulator")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok",
        "candidate": args.candidate,
        "mode": "frozen-checkpoint disjoint C4 confirmation; zero training steps",
        "dataset_lock": str(LOCK),
        "dataset_lock_sha256": sha256(LOCK),
        "dataset": {
            "evaluation": lock["evaluation_dataset"],
            "sequence_length": length,
            "eval_indices": indices,
            "test_rows_inspected_or_used_for_fitness": 0,
        },
        "injection": injection,
        "training": {"steps": 0, "hyperparameter_changes": 0},
        "evaluation_active": evaluation_active,
        "evaluation_no_entanglement": evaluation_no_ent,
        "simulator_calls": simulator_calls,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = OUTPUT_DIR / f"confirmation-{args.candidate}-s{lock['seed']}-e{lock['eval_count']}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "candidate": args.candidate,
        "mean_token_nll": evaluation_active["metrics"]["mean_token_nll"],
        "mean_seconds": evaluation_active["metrics"]["mean_sequence_seconds"],
        "output": str(output),
        "sha256": sha256(output),
    }, indent=2))


if __name__ == "__main__":
    main()

