#!/usr/bin/env python3
"""GPU-only language-modeling screen for layer-level quantum adapters."""

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


PROJECT_DIR = Path(__file__).resolve().parents[2]
RECORD_DIR = PROJECT_DIR / "records"
TOKEN_DIR = PROJECT_DIR / "datasets/processed/wikitext2_qwen38"
PREPARATION = PROJECT_DIR / "artifacts/wikitext2-preparation.json"
ARTIFACT_DIR = PROJECT_DIR / "artifacts/wikitext_stage_a"
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.cayley_two_qubit import freeze_and_inject_qh010
from quantum_qwen38.prefill_parameter_generator import freeze_and_inject_qh022


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        choices=[
            "base", "qh010", "dq010", "cc010", "qh010_noent",
            "qh022", "cc022", "qh022_noent",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--train-sequences", type=int, default=256)
    parser.add_argument("--eval-sequences", type=int, default=64)
    parser.add_argument("--eval-indices-file", type=Path, default=None)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--theta-l2", type=float, default=0.0)
    parser.add_argument("--timing-context", choices=["exclusive", "shared"], default="shared")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
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


def block_indices(token_count: int, sequence_length: int, count: int, seed: int) -> list[int]:
    complete_blocks = token_count // sequence_length
    generator = np.random.default_rng(seed)
    return generator.permutation(complete_blocks)[: min(count, complete_blocks)].tolist()


def block(token_ids: torch.Tensor, index: int, sequence_length: int, device: torch.device) -> torch.Tensor:
    start = index * sequence_length
    return token_ids[start : start + sequence_length].to(device=device, dtype=torch.long).unsqueeze(0)


def inject(model, candidate: str) -> dict:
    if candidate == "base":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return {"candidate": "base", "trainable_parameter_count": 0}
    if candidate in {"qh022", "cc022", "qh022_noent"}:
        core_kind = {
            "qh022": "quantum",
            "cc022": "classical",
            "qh022_noent": "no_entanglement",
        }[candidate]
        return freeze_and_inject_qh022(model, core_kind=core_kind)
    core_kind = {
        "qh010": "quantum",
        "dq010": "dequantized",
        "cc010": "classical",
        "qh010_noent": "no_entanglement",
    }[candidate]
    return freeze_and_inject_qh010(model, core_kind=core_kind)


def train(model, token_ids: torch.Tensor, indices: list[int], args: argparse.Namespace) -> dict:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        return {"steps": 0, "losses": [], "learning_rate": None}
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    losses, task_losses, grad_norms, seconds = [], [], [], []
    for epoch in range(args.epochs):
        order = list(indices)
        random.Random(args.seed + 1009 * epoch).shuffle(order)
        for index in order:
            input_ids = block(token_ids, index, args.sequence_length, model.device)
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            task_loss = model(input_ids=input_ids, labels=input_ids).loss
            regularizer = args.theta_l2 * sum(parameter.float().square().mean() for parameter in trainable)
            loss = task_loss + regularizer
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            task_losses.append(float(task_loss.detach()))
            losses.append(float(loss.detach()))
            grad_norms.append(float(grad_norm))
            seconds.append(time.perf_counter() - started)
    return {
        "steps": len(losses),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "theta_l2": args.theta_l2,
        "losses": losses,
        "task_loss_first_16_mean": float(np.mean(task_losses[:16])),
        "task_loss_last_16_mean": float(np.mean(task_losses[-16:])),
        "gradient_norm_mean": float(np.mean(grad_norms)),
        "mean_step_seconds": float(np.mean(seconds)),
        "total_step_seconds": float(np.sum(seconds)),
        "frozen_parameter_grad_tensor_count": sum(
            parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad
        ),
    }


@torch.inference_mode()
def evaluate(model, token_ids: torch.Tensor, indices: list[int], sequence_length: int) -> dict:
    model.eval()
    rows, total_nll, total_tokens, seconds = [], 0.0, 0, []
    for index in indices:
        input_ids = block(token_ids, index, sequence_length, model.device)
        started = time.perf_counter()
        loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        predicted_tokens = sequence_length - 1
        nll = float(loss) * predicted_tokens
        rows.append({
            "block_index": index,
            "predicted_tokens": predicted_tokens,
            "mean_token_nll": float(loss),
            "total_nll": nll,
            "seconds": elapsed,
        })
        total_nll += nll
        total_tokens += predicted_tokens
        seconds.append(elapsed)
    mean_nll = total_nll / total_tokens
    return {
        "metrics": {
            "sequence_count": len(rows),
            "predicted_token_count": total_tokens,
            "mean_token_nll": mean_nll,
            "token_perplexity": math.exp(min(50.0, mean_nll)),
            "mean_sequence_seconds": float(np.mean(seconds)),
            "p95_sequence_seconds": float(np.quantile(seconds, 0.95)),
        },
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("WikiText Stage-A is GPU-only")
    preparation = json.loads(PREPARATION.read_text(encoding="utf-8"))
    if preparation["protocol_disclosure"]["test_rows_inspected_or_used_for_fitness"] != 0:
        raise RuntimeError("WikiText test leakage disclosure is nonzero")
    seed_everything(args.seed)
    train_path = TOKEN_DIR / "train-tokens.pt"
    validation_path = TOKEN_DIR / "validation-tokens.pt"
    train_tokens = torch.load(train_path, map_location="cpu", weights_only=True)
    validation_tokens = torch.load(validation_path, map_location="cpu", weights_only=True)
    train_indices = block_indices(len(train_tokens), args.sequence_length, args.train_sequences, args.seed)
    eval_indices_file_sha256 = None
    if args.eval_indices_file is None:
        eval_indices = block_indices(
            len(validation_tokens), args.sequence_length, args.eval_sequences, args.seed + 1
        )
    else:
        locked = json.loads(args.eval_indices_file.read_text(encoding="utf-8"))
        if int(locked["sequence_length"]) != args.sequence_length:
            raise ValueError("locked eval sequence length does not match the run")
        eval_indices = [int(index) for index in locked["indices"][: args.eval_sequences]]
        if len(eval_indices) != args.eval_sequences or len(set(eval_indices)) != len(eval_indices):
            raise ValueError("locked eval indices are insufficient or duplicated")
        complete = len(validation_tokens) // args.sequence_length
        if any(index < 0 or index >= complete for index in eval_indices):
            raise ValueError("locked eval index is out of range")
        eval_indices_file_sha256 = sha256_file(args.eval_indices_file)
    model_dir = Path((RECORD_DIR / "active_model_path.txt").read_text(encoding="utf-8").strip())
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    base_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    audit = inject(model, args.candidate)
    trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
    training = train(model, train_tokens, train_indices, args)
    evaluation = evaluate(model, validation_tokens, eval_indices, args.sequence_length)
    suffix = f"-{args.run_suffix}" if args.run_suffix else ""
    run_id = (
        f"wikitext-{args.candidate}{suffix}-s{args.seed}-n{len(train_indices)}-l{args.sequence_length}"
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ARTIFACT_DIR / f"{run_id}-adapter.pt"
    if trainable:
        torch.save(
            {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad},
            checkpoint_path,
        )
    payload = {
        "status": "ok",
        "run_id": run_id,
        "candidate": args.candidate,
        "seed": args.seed,
        "model_dir": str(model_dir),
        "base_parameter_count": base_parameter_count,
        "timing_context": args.timing_context,
        "candidate_audit": audit,
        "trainable_parameters": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
        "dataset": {
            "name": preparation["dataset"],
            "config": preparation["config"],
            "source_revision": preparation["source_revision"],
            "license": preparation["license"],
            "train_token_sha256": sha256_file(train_path),
            "validation_token_sha256": sha256_file(validation_path),
            "train_indices": train_indices,
            "eval_indices": eval_indices,
            "eval_indices_file": str(args.eval_indices_file) if args.eval_indices_file else None,
            "eval_indices_file_sha256": eval_indices_file_sha256,
            "sequence_length": args.sequence_length,
            "test_rows_used_for_training_or_evaluation": 0,
            "protocol_disclosure": preparation["protocol_disclosure"],
        },
        "training": training,
        "evaluation_after": evaluation,
        "adapter_checkpoint": str(checkpoint_path) if trainable else None,
        "gpu_peak_allocated_bytes": [
            torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())
        ],
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    output = ARTIFACT_DIR / f"{run_id}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "run_id": run_id,
        "trainable_parameters": payload["trainable_parameters"],
        "training": training,
        "evaluation_after_metrics": evaluation["metrics"],
        "record": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
