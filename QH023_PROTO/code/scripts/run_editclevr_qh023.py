#!/usr/bin/env python3
"""GPU-only Qwen3.8-27B Stage-A for QH-023 on EditCLEVR development data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor


PROJECT = Path(__file__).resolve().parents[2]
RECORDS = PROJECT / "records"
DATA = PROJECT / "datasets/processed/editclevr-dev"
IMAGE_ROOT = DATA / "images"
DEFAULT_LOCK = PROJECT / "artifacts/editclevr/qh023-stage-a-lock.json"
ARTIFACTS = PROJECT / "artifacts/editclevr"
sys.path.insert(0, str(PROJECT / "code/src"))

from quantum_qwen38.pauli_stiefel_adapter import (
    ClassicalStiefelAdapter,
    DequantizedPauliStiefelAdapter,
    PauliStiefelAdapter,
    ProjectionInputPauliStiefel,
    QH023Config,
)


FACTORS = {"color", "shape", "size", "material"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        choices=["base", "qh023", "dq023", "cc023", "qh023_noent"],
        required=True,
    )
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--seed", type=int, default=20260836)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--reverse-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--run-suffix", default="stage-a")
    parser.add_argument("--timing-context", choices=["exclusive", "shared"], default="shared")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def open_pair(row: dict, reverse: bool = False) -> tuple[Image.Image, Image.Image]:
    paths = [IMAGE_ROOT / row["before_image"], IMAGE_ROOT / row["after_image"]]
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return (images[1], images[0]) if reverse else (images[0], images[1])


def render_inputs(processor, images: tuple[Image.Image, Image.Image], row: dict, answer=None):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "BEFORE image:"},
            {"type": "image"},
            {"type": "text", "text": "AFTER image:"},
            {"type": "image"},
            {"type": "text", "text": row["prompt"]},
        ],
    }]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=answer is None,
        enable_thinking=False,
    )
    return processor(text=[text], images=list(images), return_tensors="pt")


def supervised_inputs(processor, row: dict, device):
    images = open_pair(row)
    prompt = render_inputs(processor, images, row)
    full = render_inputs(processor, images, row, row["answer"]).to(device)
    labels = full["input_ids"].clone()
    labels[:, : int(prompt["input_ids"].shape[-1])] = -100
    full["labels"] = labels
    return full


def parse_answer(raw: str) -> tuple[str | None, str | None, str | None]:
    normalized = raw.lower().strip()
    normalized = normalized.replace("→", "|").replace("->", "|")
    tokens = [token.strip(" `\"'.,:;[]{}()") for token in normalized.split("|")]
    if len(tokens) >= 3:
        factor = next((token for token in tokens if token in FACTORS), None)
        if factor is not None:
            position = tokens.index(factor)
            if position + 2 < len(tokens):
                return factor, tokens[position + 1], tokens[position + 2]
    match = re.search(
        r"(color|shape|size|material)\s*[:;,|\-]+\s*([a-z0-9_]+)\s*(?:to|[:;,|\-]+)\s*([a-z0-9_]+)",
        normalized,
    )
    return (None, None, None) if match is None else match.groups()


def inject(model, candidate: str) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if candidate == "base":
        return {"candidate": "base", "trainable_parameter_count": 0}
    config = QH023Config()
    layer = model.model.language_model.layers[config.layer_index]
    if getattr(layer, "block_type", None) != "full_attention":
        raise RuntimeError("QH-023 placement is not full-attention")
    projection = getattr(layer.self_attn, config.target_projection)
    reference = next(projection.parameters())
    if candidate == "qh023":
        adapter = PauliStiefelAdapter(config).to(reference.device)
    elif candidate == "dq023":
        adapter = DequantizedPauliStiefelAdapter(config).to(reference.device)
    elif candidate == "cc023":
        adapter = ClassicalStiefelAdapter(config).to(reference.device)
    else:
        adapter = PauliStiefelAdapter(config, entangle=False).to(reference.device)
    setattr(layer.self_attn, config.target_projection, ProjectionInputPauliStiefel(projection, adapter))
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    return {
        "candidate": adapter.audit()["candidate"],
        "injection_path": "model.model.language_model.layers.7.self_attn.v_proj.input",
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }


def train(model, processor, rows: list[dict], indices: list[int], args) -> dict:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        return {"steps": 0, "learning_rate": None, "losses": []}
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    losses, gradients, seconds = [], [], []
    for epoch in range(args.epochs):
        order = list(indices)
        random.Random(args.seed + 10007 * epoch).shuffle(order)
        for index in order:
            inputs = supervised_inputs(processor, rows[index], model.device)
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            loss = model(**inputs).loss
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            losses.append(float(loss.detach()))
            gradients.append(float(gradient))
            seconds.append(time.perf_counter() - started)
    return {
        "steps": len(losses),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "losses": losses,
        "loss_first_16_mean": float(np.mean(losses[:16])),
        "loss_last_16_mean": float(np.mean(losses[-16:])),
        "gradient_norm_mean": float(np.mean(gradients)),
        "mean_step_seconds": float(np.mean(seconds)),
        "total_step_seconds": float(np.sum(seconds)),
        "frozen_parameter_grad_tensor_count": sum(
            parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad
        ),
    }


@torch.inference_mode()
def predict(model, processor, row: dict, reverse: bool, max_new_tokens: int) -> dict:
    images = open_pair(row, reverse=reverse)
    inputs = render_inputs(processor, images, row).to(model.device)
    input_length = int(inputs["input_ids"].shape[-1])
    started = time.perf_counter()
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
    )
    torch.cuda.synchronize()
    raw = processor.decode(output[0, input_length:], skip_special_tokens=True).strip()
    parsed = parse_answer(raw)
    target = (
        row["edit_factor"], row["new_value"], row["old_value"]
    ) if reverse else (
        row["edit_factor"], row["old_value"], row["new_value"]
    )
    return {
        "raw": raw,
        "parsed": list(parsed),
        "target": list(target),
        "exact": parsed == target,
        "factor_exact": parsed[0] == target[0],
        "old_exact": parsed[1] == target[1],
        "new_exact": parsed[2] == target[2],
        "seconds": time.perf_counter() - started,
    }


def evaluate(model, processor, rows, indices, reverse_indices, max_new_tokens) -> dict:
    model.eval()
    predictions = []
    reverse_set = set(reverse_indices)
    for index in indices:
        row = rows[index]
        original = predict(model, processor, row, False, max_new_tokens)
        record = {
            "val_index": index,
            "pair_id": row["pair_id"],
            "base_scene_seed": row["base_scene_seed"],
            "edit_factor": row["edit_factor"],
            **{f"original_{key}": value for key, value in original.items()},
        }
        if index in reverse_set:
            reverse = predict(model, processor, row, True, max_new_tokens)
            original_target = tuple(original["target"])
            reverse_parsed = tuple(reverse["parsed"])
            record.update({f"reverse_{key}": value for key, value in reverse.items()})
            record["reverse_matches_original_target"] = reverse_parsed == original_target
            record["prediction_changes_on_swap"] = reverse_parsed != tuple(original["parsed"])
        predictions.append(record)
    exact = [float(row["original_exact"]) for row in predictions]
    by_factor: dict[str, list[float]] = defaultdict(list)
    for row in predictions:
        by_factor[row["edit_factor"]].append(float(row["original_exact"]))
    reverse_rows = [row for row in predictions if "reverse_exact" in row]
    metrics = {
        "n": len(predictions),
        "three_component_exact_accuracy": float(np.mean(exact)),
        "factor_accuracy": float(np.mean([row["original_factor_exact"] for row in predictions])),
        "old_value_accuracy": float(np.mean([row["original_old_exact"] for row in predictions])),
        "new_value_accuracy": float(np.mean([row["original_new_exact"] for row in predictions])),
        "macro_factor_exact": float(np.mean([np.mean(values) for values in by_factor.values()])),
        "per_factor_exact": {key: float(np.mean(value)) for key, value in sorted(by_factor.items())},
        "mean_generation_seconds": float(np.mean([row["original_seconds"] for row in predictions])),
        "reverse_n": len(reverse_rows),
    }
    if reverse_rows:
        reverse_exact = float(np.mean([row["reverse_exact"] for row in reverse_rows]))
        reverse_wrong_direction = float(
            np.mean([row["reverse_matches_original_target"] for row in reverse_rows])
        )
        metrics.update({
            "reverse_direction_exact": reverse_exact,
            "reverse_matches_original_direction": reverse_wrong_direction,
            "pair_dependency_gain": reverse_exact - reverse_wrong_direction,
            "prediction_change_on_swap": float(
                np.mean([row["prediction_changes_on_swap"] for row in reverse_rows])
            ),
        })
    return {"metrics": metrics, "predictions": predictions}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EditCLEVR QH-023 screen is GPU-only")
    seed_everything(args.seed)
    lock = json.loads(args.lock_file.read_text(encoding="utf-8"))
    if lock["status"] != "locked_before_any_qh023_task_prediction":
        raise RuntimeError("invalid preregistration lock")
    train_rows = load_jsonl(DATA / "train.jsonl")
    val_rows = load_jsonl(DATA / "val.jsonl")
    train_indices = [int(i) for i in lock["train_indices"][: args.train_samples]]
    eval_indices = [int(i) for i in lock["val_indices"][: args.eval_samples]]
    reverse_allowed = set(lock["reverse_counterfactual_val_indices"])
    reverse_indices = [i for i in eval_indices if i in reverse_allowed][: args.reverse_samples]
    model_dir = Path((RECORDS / "active_model_path.txt").read_text().strip())
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
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
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    training = train(model, processor, train_rows, train_indices, args)
    evaluation = evaluate(
        model, processor, val_rows, eval_indices, reverse_indices, args.max_new_tokens
    )
    run_id = (
        f"editclevr-{args.candidate}-{args.run_suffix}-s{args.seed}-"
        f"n{len(train_indices)}-e{len(eval_indices)}"
    )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    checkpoint = ARTIFACTS / f"{run_id}-adapter.pt"
    if trainable:
        torch.save(
            {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad},
            checkpoint,
        )
    result = {
        "status": "ok",
        "run_id": run_id,
        "candidate": args.candidate,
        "seed": args.seed,
        "model_dir": str(model_dir),
        "base_parameter_count": base_parameter_count,
        "candidate_audit": audit,
        "trainable_parameters": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
        "timing_context": args.timing_context,
        "dataset": {
            "name": "EditCLEVR Phase 1 atomic_id development",
            "license": "CC BY 4.0",
            "source_revision": lock["source"]["repo_revision"],
            "lock_file": str(args.lock_file),
            "lock_sha256": sha256_file(args.lock_file),
            "train_jsonl_sha256": sha256_file(DATA / "train.jsonl"),
            "val_jsonl_sha256": sha256_file(DATA / "val.jsonl"),
            "train_indices": train_indices,
            "eval_indices": eval_indices,
            "reverse_indices": reverse_indices,
            "test_rows_used_for_fitness": 0,
            "test_payloads_materialized": 0,
        },
        "training": training,
        "evaluation_after": evaluation,
        "adapter_checkpoint": str(checkpoint) if trainable else None,
        "gpu_peak_allocated_bytes": [
            torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
        ],
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    output = ARTIFACTS / f"{run_id}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "run_id": run_id,
        "trainable_parameters": result["trainable_parameters"],
        "training_summary": {key: value for key, value in training.items() if key != "losses"},
        "metrics": evaluation["metrics"],
        "record": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
