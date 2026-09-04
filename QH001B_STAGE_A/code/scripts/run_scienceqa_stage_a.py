#!/usr/bin/env python3
"""Real-data Stage-A training/evaluation for Qwen3.8 quantum candidates.

The script deliberately evaluates original, shuffled, and blank images.  This
separates answer accuracy from actual image dependence and keeps all quantum
simulation and model execution on CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor


PROJECT_DIR = Path(__file__).resolve().parents[2]
RECORD_DIR = PROJECT_DIR / "records"
ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "scienceqa_stage_a"
TRAIN_FILE = (
    PROJECT_DIR
    / "datasets/raw/scienceqa/ScienceQA-IMG/train-00000-of-00001.parquet"
)
VAL_FILE = (
    PROJECT_DIR
    / "datasets/raw/scienceqa/ScienceQA-IMG/validation-00000-of-00001.parquet"
)
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.amplitude_unitary import freeze_and_inject_qh006
from quantum_qwen38.classical_control import freeze_and_inject_cc001
from quantum_qwen38.data_reupload_correlator import freeze_and_inject_qh007
from quantum_qwen38.quantum_no_entanglement import freeze_and_inject_no_entanglement
from quantum_qwen38.quantum_gated_low_rank import freeze_and_inject_qh008
from quantum_qwen38.quantum_residual_bf16 import (
    QH001BF16Config,
    freeze_and_inject_qh001b,
)


LETTERS = string.ascii_uppercase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        choices=[
            "base", "qh006e", "qh006d", "qh001b", "qh007", "qh008",
            "cc001", "cc008", "noent", "qh008_noent",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--train-indices-json", type=Path, default=None)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--counterfactual-samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--eval-before", action="store_true")
    parser.add_argument(
        "--timing-context", choices=["exclusive", "shared", "unspecified"], default="unspecified"
    )
    parser.add_argument("--counterfactual-train-weight", type=float, default=0.0)
    parser.add_argument("--counterfactual-margin", type=float, default=0.5)
    parser.add_argument(
        "--loss-mode",
        choices=["sequence", "answer_token"],
        default="sequence",
        help=(
            "sequence uses the chat-template answer span; answer_token optimizes "
            "only the first answer token and applies counterfactual ranking there"
        ),
    )
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


def question_text(sample: dict) -> str:
    choices = "\n".join(
        f"{LETTERS[index]}. {choice}" for index, choice in enumerate(sample["choices"])
    )
    hint = sample.get("hint") or ""
    hint_text = f"\nContext: {hint}" if hint.strip() else ""
    valid_letters = ", ".join(LETTERS[: len(sample["choices"])])
    return (
        f"Question: {sample['question']}{hint_text}\nChoices:\n{choices}\n"
        f"Answer with exactly one uppercase letter from: {valid_letters}."
    )


def render_inputs(processor, image: Image.Image, sample: dict, answer: str | None = None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question_text(sample)},
            ],
        }
    ]
    if answer is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=answer is None,
        enable_thinking=False,
    )
    return processor(text=[rendered], images=[image.convert("RGB")], return_tensors="pt")


def supervised_inputs(processor, image: Image.Image, sample: dict, answer: str, device):
    prompt = render_inputs(processor, image, sample)
    full = render_inputs(processor, image, sample, answer).to(device)
    prompt_length = int(prompt["input_ids"].shape[-1])
    if prompt_length >= int(full["input_ids"].shape[-1]):
        raise ValueError("assistant answer token is missing from the rendered sequence")
    labels = full["input_ids"].clone()
    labels[:, :prompt_length] = -100
    full["labels"] = labels
    return full, prompt_length


def parse_letter(text: str, number_of_choices: int) -> str | None:
    normalized = text.strip().upper()
    valid = set(LETTERS[:number_of_choices])
    match = re.search(r"(?<![A-Z])([A-Z])(?![A-Z])", normalized)
    if match and match.group(1) in valid:
        return match.group(1)
    for char in normalized:
        if char in valid:
            return char
    return None


def select_indices(length: int, count: int, seed: int) -> list[int]:
    generator = np.random.default_rng(seed)
    return generator.permutation(length)[: min(count, length)].tolist()


def inject_candidate(model, candidate: str) -> dict:
    if candidate == "base":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return {"candidate": "base", "trainable_parameter_count": 0}
    if candidate == "qh006e":
        return freeze_and_inject_qh006(model, mode="post")
    if candidate == "qh006d":
        return freeze_and_inject_qh006(model, mode="sandwich")
    if candidate == "qh001b":
        return freeze_and_inject_qh001b(model)
    if candidate == "qh007":
        return freeze_and_inject_qh007(model)
    if candidate == "cc001":
        return freeze_and_inject_cc001(model)
    if candidate == "qh008":
        return freeze_and_inject_qh008(model, core_kind="quantum")
    if candidate == "cc008":
        return freeze_and_inject_qh008(model, core_kind="classical")
    if candidate == "qh008_noent":
        return freeze_and_inject_qh008(model, core_kind="no_entanglement")
    if candidate == "noent":
        # Gain fixed from an independent calibration batch in
        # qh001b-no-entanglement-control.json.
        config = QH001BF16Config(scale_init=0.11888859421014786)
        return freeze_and_inject_no_entanglement(model, config)
    raise ValueError(candidate)


@torch.inference_mode()
def predict(model, processor, sample: dict, image: Image.Image, max_new_tokens: int) -> dict:
    inputs = render_inputs(processor, image, sample).to(model.device)
    input_length = int(inputs["input_ids"].shape[-1])
    started = time.perf_counter()
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated = output[0, input_length:]
    text = processor.decode(generated, skip_special_tokens=True)
    return {
        "letter": parse_letter(text, len(sample["choices"])),
        "raw": text,
        "seconds": elapsed,
    }


def evaluate(
    model,
    processor,
    dataset: Dataset,
    indices: list[int],
    counterfactual_count: int,
    max_new_tokens: int,
) -> dict:
    model.eval()
    rows = []
    total_seconds = 0.0
    cf_count = min(counterfactual_count, len(indices))
    for position, index in enumerate(indices):
        sample = dataset[index]
        original_image = sample["image"].convert("RGB")
        original = predict(model, processor, sample, original_image, max_new_tokens)
        total_seconds += original["seconds"]
        answer = LETTERS[int(sample["answer"])]
        row = {
            "dataset_index": index,
            "answer": answer,
            "original_prediction": original["letter"],
            "original_raw": original["raw"],
            "original_correct": original["letter"] == answer,
            "original_seconds": original["seconds"],
        }
        if position < cf_count:
            shuffled_index = indices[(position + max(1, cf_count // 2)) % len(indices)]
            shuffled_image = dataset[shuffled_index]["image"].convert("RGB")
            blank_image = Image.new("RGB", original_image.size, "white")
            shuffled = predict(model, processor, sample, shuffled_image, max_new_tokens)
            blank = predict(model, processor, sample, blank_image, max_new_tokens)
            total_seconds += shuffled["seconds"] + blank["seconds"]
            row.update(
                {
                    "shuffled_image_index": shuffled_index,
                    "shuffled_prediction": shuffled["letter"],
                    "shuffled_correct": shuffled["letter"] == answer,
                    "shuffled_seconds": shuffled["seconds"],
                    "blank_prediction": blank["letter"],
                    "blank_correct": blank["letter"] == answer,
                    "blank_seconds": blank["seconds"],
                }
            )
        rows.append(row)

    original_accuracy = float(np.mean([row["original_correct"] for row in rows]))
    cf_rows = rows[:cf_count]
    parsed_rate = float(np.mean([row["original_prediction"] is not None for row in rows]))
    metrics = {
        "n": len(rows),
        "counterfactual_n": cf_count,
        "accuracy": original_accuracy,
        "parsed_prediction_rate": parsed_rate,
        "total_generation_seconds": total_seconds,
        "mean_generation_seconds_per_call": total_seconds / (len(rows) + 2 * cf_count),
    }
    if cf_rows:
        shuffled_accuracy = float(np.mean([row["shuffled_correct"] for row in cf_rows]))
        blank_accuracy = float(np.mean([row["blank_correct"] for row in cf_rows]))
        original_cf_accuracy = float(np.mean([row["original_correct"] for row in cf_rows]))
        metrics.update(
            {
                "counterfactual_original_accuracy": original_cf_accuracy,
                "shuffled_accuracy": shuffled_accuracy,
                "blank_accuracy": blank_accuracy,
                "original_minus_shuffled": original_cf_accuracy - shuffled_accuracy,
                "multimodal_gain_original_minus_blank": original_cf_accuracy - blank_accuracy,
                "visual_rescue_rate": float(
                    np.mean([row["original_correct"] and not row["blank_correct"] for row in cf_rows])
                ),
                "prediction_change_on_shuffle": float(
                    np.mean(
                        [row["original_prediction"] != row["shuffled_prediction"] for row in cf_rows]
                    )
                ),
                "prediction_change_on_blank": float(
                    np.mean(
                        [row["original_prediction"] != row["blank_prediction"] for row in cf_rows]
                    )
                ),
            }
        )
    return {"metrics": metrics, "predictions": rows}


def train(
    model,
    processor,
    dataset: Dataset,
    indices: list[int],
    epochs: int,
    lr: float,
    counterfactual_weight: float,
    counterfactual_margin: float,
    loss_mode: str,
) -> dict:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        return {"steps": 0, "losses": [], "learning_rate": None}
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    losses = []
    original_losses = []
    counterfactual_losses = []
    ranking_losses = []
    counterfactual_modes = []
    grad_norms = []
    step_seconds = []
    for epoch in range(epochs):
        order = list(indices)
        random.Random(10_000 * epoch + 17).shuffle(order)
        for position, index in enumerate(order):
            sample = dataset[index]
            answer = LETTERS[int(sample["answer"])]
            full, prompt_length = supervised_inputs(
                processor, sample["image"], sample, answer, model.device
            )
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            output = model(**full)
            if loss_mode == "answer_token":
                target_token = full["input_ids"][:, prompt_length]
                original_loss = F.cross_entropy(
                    output.logits[:, prompt_length - 1, :].float(), target_token
                )
            else:
                original_loss = output.loss
            counterfactual_loss = None
            ranking_loss = original_loss.new_zeros(())
            counterfactual_mode = None
            if counterfactual_weight > 0.0:
                if position % 2 == 0:
                    counterfactual_image = Image.new(
                        "RGB", sample["image"].size, "white"
                    )
                    counterfactual_mode = "blank"
                else:
                    offset = max(1, len(order) // 2)
                    other_index = order[(position + offset) % len(order)]
                    counterfactual_image = dataset[other_index]["image"]
                    counterfactual_mode = "shuffled"
                cf_full, cf_prompt_length = supervised_inputs(
                    processor,
                    counterfactual_image,
                    sample,
                    answer,
                    model.device,
                )
                cf_output = model(**cf_full)
                if loss_mode == "answer_token":
                    cf_target_token = cf_full["input_ids"][:, cf_prompt_length]
                    counterfactual_loss = F.cross_entropy(
                        cf_output.logits[:, cf_prompt_length - 1, :].float(),
                        cf_target_token,
                    )
                else:
                    counterfactual_loss = cf_output.loss
                ranking_loss = torch.relu(
                    original_loss.new_tensor(counterfactual_margin)
                    + original_loss
                    - counterfactual_loss
                )
            loss = original_loss + counterfactual_weight * ranking_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            losses.append(float(loss.detach()))
            original_losses.append(float(original_loss.detach()))
            if counterfactual_loss is not None:
                counterfactual_losses.append(float(counterfactual_loss.detach()))
                ranking_losses.append(float(ranking_loss.detach()))
                counterfactual_modes.append(counterfactual_mode)
            grad_norms.append(float(grad_norm))
            step_seconds.append(time.perf_counter() - started)
    frozen_grad_count = sum(
        parameter.grad is not None
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    return {
        "steps": len(losses),
        "epochs": epochs,
        "loss_mode": loss_mode,
        "learning_rate": lr,
        "counterfactual_train_weight": counterfactual_weight,
        "counterfactual_margin": counterfactual_margin,
        "losses": losses,
        "original_loss_first_16_mean": float(np.mean(original_losses[:16])),
        "original_loss_last_16_mean": float(np.mean(original_losses[-16:])),
        "counterfactual_loss_mean": None
        if not counterfactual_losses
        else float(np.mean(counterfactual_losses)),
        "ranking_loss_mean": None if not ranking_losses else float(np.mean(ranking_losses)),
        "counterfactual_mode_counts": {
            mode: counterfactual_modes.count(mode) for mode in sorted(set(counterfactual_modes))
        },
        "loss_first_16_mean": float(np.mean(losses[:16])),
        "loss_last_16_mean": float(np.mean(losses[-16:])),
        "gradient_norm_mean": float(np.mean(grad_norms)),
        "mean_step_seconds": float(np.mean(step_seconds)),
        "total_step_seconds": float(np.sum(step_seconds)),
        "frozen_parameter_grad_tensor_count": frozen_grad_count,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-A is GPU-only")
    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        raise FileNotFoundError("ScienceQA train/validation parquet is incomplete")
    seed_everything(args.seed)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = Path((RECORD_DIR / "active_model_path.txt").read_text(encoding="utf-8").strip())
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    load_started = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    load_seconds = time.perf_counter() - load_started
    base_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    audit = inject_candidate(model, args.candidate)
    trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
    train_data = Dataset.from_parquet(str(TRAIN_FILE))
    val_data = Dataset.from_parquet(str(VAL_FILE))
    train_index_source = None
    if args.train_indices_json is None:
        train_indices = select_indices(len(train_data), args.train_samples, args.seed)
    else:
        index_record = json.loads(args.train_indices_json.read_text(encoding="utf-8"))
        available_indices = index_record["wrong_indices"]
        train_indices = [int(index) for index in available_indices[: args.train_samples]]
        if len(train_indices) < args.train_samples:
            raise ValueError(
                f"hard-mining record has only {len(train_indices)} wrong rows, "
                f"requested {args.train_samples}"
            )
        train_index_source = {
            "path": str(args.train_indices_json),
            "sha256": sha256_file(args.train_indices_json),
            "source_run_id": index_record.get("run_id"),
        }
    eval_indices = select_indices(len(val_data), args.eval_samples, args.seed + 1)

    eval_before = None
    if args.eval_before or args.candidate == "base":
        eval_before = evaluate(
            model,
            processor,
            val_data,
            eval_indices,
            args.counterfactual_samples,
            args.max_new_tokens,
        )
    default_lr = 5.0e-3 if args.candidate in {"qh006e", "qh006d"} else 3.0e-4
    learning_rate = args.learning_rate if args.learning_rate is not None else default_lr
    training = train(
        model,
        processor,
        train_data,
        train_indices,
        args.epochs,
        learning_rate,
        args.counterfactual_train_weight,
        args.counterfactual_margin,
        args.loss_mode,
    )
    eval_after = eval_before if args.candidate == "base" else evaluate(
        model,
        processor,
        val_data,
        eval_indices,
        args.counterfactual_samples,
        args.max_new_tokens,
    )

    data_suffix = "" if train_index_source is None else "-hard"
    objective_suffix = (
        ""
        if args.counterfactual_train_weight == 0.0
        else f"-cfw{args.counterfactual_train_weight:g}-m{args.counterfactual_margin:g}"
    )
    if args.loss_mode == "answer_token":
        objective_suffix += "-answer-token"
    run_id = (
        f"scienceqa-{args.candidate}-s{args.seed}-n{len(train_indices)}"
        f"{data_suffix}{objective_suffix}"
    )
    checkpoint_path = ARTIFACT_DIR / f"{run_id}-adapter.pt"
    if trainable:
        torch.save(
            {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad},
            checkpoint_path,
        )
    result = {
        "status": "ok",
        "run_id": run_id,
        "candidate": args.candidate,
        "seed": args.seed,
        "model_dir": str(model_dir),
        "base_parameter_count": base_parameter_count,
        "load_seconds": load_seconds,
        "timing_context": args.timing_context,
        "candidate_audit": audit,
        "trainable_parameters": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
        "dataset": {
            "name": "ScienceQA-IMG",
            "source": "lmms-lab/ScienceQA (formatted from lupantech/ScienceQA)",
            "license": "CC BY-NC-SA 4.0; academic/non-commercial track only",
            "train_rows": len(train_data),
            "validation_rows": len(val_data),
            "train_sha256": sha256_file(TRAIN_FILE),
            "validation_sha256": sha256_file(VAL_FILE),
            "train_indices": train_indices,
            "train_index_source": train_index_source,
            "eval_indices": eval_indices,
        },
        "evaluation_before": eval_before,
        "training": training,
        "evaluation_after": eval_after,
        "adapter_checkpoint": str(checkpoint_path) if trainable else None,
        "gpu_peak_allocated_bytes": [
            torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())
        ],
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = ARTIFACT_DIR / f"{run_id}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "run_id": run_id,
        "trainable_parameters": result["trainable_parameters"],
        "training": training,
        "evaluation_before_metrics": None if eval_before is None else eval_before["metrics"],
        "evaluation_after_metrics": eval_after["metrics"],
        "record": str(output_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
