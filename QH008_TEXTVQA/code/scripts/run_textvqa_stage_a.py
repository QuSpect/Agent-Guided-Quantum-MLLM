#!/usr/bin/env python3
"""GPU-only TextVQA pilot for quantum/classical Qwen3.8 adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor


PROJECT_DIR = Path(__file__).resolve().parents[2]
RECORD_DIR = PROJECT_DIR / "records"
ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "textvqa_stage_a"
TRAIN_FILE = PROJECT_DIR / "datasets/raw/textvqa/textvqa/train/0000.parquet"
VAL_FILE = PROJECT_DIR / "datasets/raw/textvqa/textvqa/validation/0000.parquet"
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.amplitude_unitary import freeze_and_inject_qh006
from quantum_qwen38.brickwork_four_qubit import freeze_and_inject_qh012
from quantum_qwen38.sparse_routed_correlator import freeze_and_inject_qh013
from quantum_qwen38.question_conditioned_anchor import freeze_and_inject_qh014
from quantum_qwen38.question_conditioned_film import freeze_and_inject_qh015
from quantum_qwen38.question_routed_relations import freeze_and_inject_qh016
from quantum_qwen38.classical_control import freeze_and_inject_cc001
from quantum_qwen38.cayley_two_qubit import freeze_and_inject_qh010, freeze_and_inject_qh011
from quantum_qwen38.correlation_gated_low_rank import freeze_and_inject_qh009
from quantum_qwen38.data_reupload_correlator import freeze_and_inject_qh007
from quantum_qwen38.quantum_no_entanglement import freeze_and_inject_no_entanglement
from quantum_qwen38.quantum_gated_low_rank import freeze_and_inject_qh008
from quantum_qwen38.quantum_residual_bf16 import QH001BF16Config, freeze_and_inject_qh001b
from quantum_qwen38.vqa_metric import EvalAIAnswerProcessor, vqa_soft_accuracy


ANSWER_PROCESSOR = EvalAIAnswerProcessor()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        choices=[
            "base", "qh006e", "qh006d", "qh001b", "qh007", "qh008",
            "cc001", "cc008", "noent", "qh008_noent",
            "qh009", "cc009", "qh009_noent",
            "qh010", "dq010", "cc010", "qh010_noent",
            "qh011", "dq011", "cc011", "qh011_noent",
            "qh012", "dq012", "cc012", "qh012_noent",
            "qh013", "cc013", "qh013_noent",
            "qh014", "cc014", "qh014_noent",
            "qh015", "cc015", "qh015_noent",
            "qh016", "cc016", "qh016_noent",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--counterfactual-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--min-consensus", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--train-indices-file", type=Path, default=None)
    parser.add_argument("--curriculum-tag", type=str, default=None)
    parser.add_argument("--load-adapter-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--anchor-control",
        choices=["normal", "shuffled_question"],
        default="normal",
    )
    parser.add_argument("--run-suffix", type=str, default=None)
    parser.add_argument(
        "--timing-context", choices=["exclusive", "shared", "unspecified"], default="unspecified"
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
    return (
        f"Question: {sample['question']}\n"
        "Answer the question using a single word or short phrase."
    )


def render_inputs(processor, image: Image.Image, sample: dict, answer: str | None = None):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question_text(sample)},
        ],
    }]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=answer is None,
        enable_thinking=False,
    )
    return processor(text=[rendered], images=[image.convert("RGB")], return_tensors="pt")


def supervised_inputs(processor, sample: dict, answer: str, device):
    prompt = render_inputs(processor, sample["image"], sample)
    full = render_inputs(processor, sample["image"], sample, answer).to(device)
    prompt_length = int(prompt["input_ids"].shape[-1])
    labels = full["input_ids"].clone()
    labels[:, :prompt_length] = -100
    full["labels"] = labels
    return full, prompt["input_ids"]


def set_external_text_anchor(model, prompt_input_ids: torch.Tensor) -> None:
    adapters = [
        module for module in model.modules()
        if callable(getattr(module, "set_text_anchor", None))
    ]
    if not adapters:
        return
    embedding = model.get_input_embeddings()
    input_ids = prompt_input_ids.to(embedding.weight.device)
    with torch.no_grad():
        token_embeddings = embedding(input_ids).float()
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        for name in ("image_token_id", "video_token_id"):
            token_id = getattr(model.config, name, None)
            if token_id is not None:
                mask &= input_ids != int(token_id)
        weights = mask.unsqueeze(-1).to(token_embeddings.dtype)
        anchor = (token_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    for adapter in adapters:
        adapter.set_text_anchor(anchor)


def normalized_answer_counts(sample: dict) -> Counter:
    return Counter(ANSWER_PROCESSOR(answer) for answer in sample["answers"])


def training_answer(sample: dict) -> str:
    counts = normalized_answer_counts(sample)
    winner = counts.most_common(1)[0][0]
    return winner


def select_train_indices(dataset: Dataset, count: int, seed: int, min_consensus: int) -> tuple[list[int], int]:
    # Avoid decoding every image while filtering only by the answer annotations.
    answer_only = dataset.select_columns(["answers"])
    eligible = [
        index
        for index in range(len(answer_only))
        if max(normalized_answer_counts(answer_only[index]).values()) >= min_consensus
    ]
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(eligible))[: min(count, len(eligible))]
    return [eligible[int(position)] for position in order], len(eligible)


def load_train_indices(
    dataset: Dataset,
    path: Path,
    count: int,
    min_consensus: int,
) -> tuple[list[int], int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("selected_indices", payload.get("wrong_indices"))
    if not isinstance(source, list) or not source:
        raise ValueError("train-indices file must contain selected_indices or wrong_indices")
    indices = [int(index) for index in source]
    if len(indices) != len(set(indices)):
        raise ValueError("train-indices file contains duplicate indices")
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError("train-indices file contains out-of-range indices")
    answer_only = dataset.select_columns(["answers"])
    ineligible = [
        index for index in indices
        if max(normalized_answer_counts(answer_only[index]).values()) < min_consensus
    ]
    if ineligible:
        raise ValueError(f"train-indices file violates min_consensus: {ineligible[:8]}")
    return indices[: min(count, len(indices))], len(indices), sha256_file(path)


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
    if candidate == "qh009":
        return freeze_and_inject_qh009(model, core_kind="quantum")
    if candidate == "cc009":
        return freeze_and_inject_qh009(model, core_kind="classical")
    if candidate == "qh009_noent":
        return freeze_and_inject_qh009(model, core_kind="no_entanglement")
    if candidate == "qh010":
        return freeze_and_inject_qh010(model, core_kind="quantum")
    if candidate == "dq010":
        return freeze_and_inject_qh010(model, core_kind="dequantized")
    if candidate == "cc010":
        return freeze_and_inject_qh010(model, core_kind="classical")
    if candidate == "qh010_noent":
        return freeze_and_inject_qh010(model, core_kind="no_entanglement")
    if candidate == "qh011":
        return freeze_and_inject_qh011(model, core_kind="quantum")
    if candidate == "dq011":
        return freeze_and_inject_qh011(model, core_kind="dequantized")
    if candidate == "cc011":
        return freeze_and_inject_qh011(model, core_kind="classical")
    if candidate == "qh011_noent":
        return freeze_and_inject_qh011(model, core_kind="no_entanglement")
    if candidate == "qh012":
        return freeze_and_inject_qh012(model, core_kind="quantum")
    if candidate == "dq012":
        return freeze_and_inject_qh012(model, core_kind="dequantized")
    if candidate == "cc012":
        return freeze_and_inject_qh012(model, core_kind="classical")
    if candidate == "qh012_noent":
        return freeze_and_inject_qh012(model, core_kind="no_entanglement")
    if candidate == "qh013":
        return freeze_and_inject_qh013(model, core_kind="quantum")
    if candidate == "cc013":
        return freeze_and_inject_qh013(model, core_kind="classical")
    if candidate == "qh013_noent":
        return freeze_and_inject_qh013(model, core_kind="no_entanglement")
    if candidate == "qh014":
        return freeze_and_inject_qh014(model, core_kind="quantum")
    if candidate == "cc014":
        return freeze_and_inject_qh014(model, core_kind="classical")
    if candidate == "qh014_noent":
        return freeze_and_inject_qh014(model, core_kind="no_entanglement")
    if candidate == "qh015":
        return freeze_and_inject_qh015(model, core_kind="quantum")
    if candidate == "cc015":
        return freeze_and_inject_qh015(model, core_kind="classical")
    if candidate == "qh015_noent":
        return freeze_and_inject_qh015(model, core_kind="no_entanglement")
    if candidate == "qh016":
        return freeze_and_inject_qh016(model, core_kind="quantum")
    if candidate == "cc016":
        return freeze_and_inject_qh016(model, core_kind="classical")
    if candidate == "qh016_noent":
        return freeze_and_inject_qh016(model, core_kind="no_entanglement")
    if candidate == "noent":
        return freeze_and_inject_no_entanglement(
            model, QH001BF16Config(scale_init=0.11888859421014786)
        )
    raise ValueError(candidate)


@torch.inference_mode()
def predict(
    model,
    processor,
    sample: dict,
    image: Image.Image,
    max_new_tokens: int,
    anchor_sample: dict | None = None,
) -> dict:
    inputs = render_inputs(processor, image, sample).to(model.device)
    if anchor_sample is None:
        anchor_input_ids = inputs["input_ids"]
    else:
        anchor_input_ids = render_inputs(processor, image, anchor_sample)["input_ids"]
    set_external_text_anchor(model, anchor_input_ids)
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
    raw = processor.decode(output[0, input_length:], skip_special_tokens=True).strip()
    score, normalized = vqa_soft_accuracy(raw, list(sample["answers"]))
    normalized_answers = {ANSWER_PROCESSOR(answer) for answer in sample["answers"]}
    return {
        "raw": raw,
        "normalized": normalized,
        "soft_score": score,
        "exact_any": normalized in normalized_answers,
        "seconds": elapsed,
    }


def evaluate(
    model,
    processor,
    dataset: Dataset,
    indices: list[int],
    counterfactual_count: int,
    max_new_tokens: int,
    anchor_control: str = "normal",
) -> dict:
    model.eval()
    rows = []
    total_seconds = 0.0
    cf_count = min(counterfactual_count, len(indices))
    for position, index in enumerate(indices):
        sample = dataset[index]
        anchor_index = index
        anchor_sample = None
        if anchor_control == "shuffled_question":
            anchor_index = indices[(position + max(1, len(indices) // 2)) % len(indices)]
            anchor_sample = dataset[anchor_index]
        original_image = sample["image"].convert("RGB")
        original = predict(
            model, processor, sample, original_image, max_new_tokens, anchor_sample
        )
        total_seconds += original["seconds"]
        row = {
            "dataset_index": index,
            "question_id": int(sample["question_id"]),
            "answers": list(sample["answers"]),
            "original_raw": original["raw"],
            "original_normalized": original["normalized"],
            "original_soft_score": original["soft_score"],
            "original_exact_any": original["exact_any"],
            "original_seconds": original["seconds"],
            "anchor_control": anchor_control,
            "anchor_question_index": anchor_index,
        }
        if position < cf_count:
            shuffled_index = indices[(position + max(1, cf_count // 2)) % len(indices)]
            shuffled_image = dataset[shuffled_index]["image"].convert("RGB")
            blank_image = Image.new("RGB", original_image.size, "white")
            shuffled = predict(
                model, processor, sample, shuffled_image, max_new_tokens, anchor_sample
            )
            blank = predict(
                model, processor, sample, blank_image, max_new_tokens, anchor_sample
            )
            total_seconds += shuffled["seconds"] + blank["seconds"]
            row.update({
                "shuffled_image_index": shuffled_index,
                "shuffled_normalized": shuffled["normalized"],
                "shuffled_soft_score": shuffled["soft_score"],
                "shuffled_seconds": shuffled["seconds"],
                "blank_normalized": blank["normalized"],
                "blank_soft_score": blank["soft_score"],
                "blank_seconds": blank["seconds"],
            })
        rows.append(row)
    cf_rows = rows[:cf_count]
    metrics = {
        "n": len(rows),
        "counterfactual_n": cf_count,
        "vqa_soft_accuracy": float(np.mean([row["original_soft_score"] for row in rows])),
        "exact_any_accuracy": float(np.mean([row["original_exact_any"] for row in rows])),
        "mean_generation_seconds_per_call": total_seconds / (len(rows) + 2 * cf_count),
    }
    if cf_rows:
        original_cf = float(np.mean([row["original_soft_score"] for row in cf_rows]))
        shuffled = float(np.mean([row["shuffled_soft_score"] for row in cf_rows]))
        blank = float(np.mean([row["blank_soft_score"] for row in cf_rows]))
        metrics.update({
            "counterfactual_original_vqa_soft_accuracy": original_cf,
            "shuffled_vqa_soft_accuracy": shuffled,
            "blank_vqa_soft_accuracy": blank,
            "original_minus_shuffled": original_cf - shuffled,
            "multimodal_gain_original_minus_blank": original_cf - blank,
            "mean_positive_visual_rescue": float(np.mean([
                max(0.0, row["original_soft_score"] - row["blank_soft_score"])
                for row in cf_rows
            ])),
            "prediction_change_on_blank": float(np.mean([
                row["original_normalized"] != row["blank_normalized"] for row in cf_rows
            ])),
        })
    return {"metrics": metrics, "predictions": rows}


def load_adapter_checkpoint(model, checkpoint_path: Path) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    named_parameters = dict(model.named_parameters())
    expected = {
        name for name, parameter in named_parameters.items() if parameter.requires_grad
    }
    received = set(payload)
    if received != expected:
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        raise RuntimeError(
            f"adapter checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, tensor in payload.items():
            target = named_parameters[name]
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
    return {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "tensor_count": len(payload),
        "parameter_count": sum(tensor.numel() for tensor in payload.values()),
    }


def train(model, processor, dataset: Dataset, indices: list[int], epochs: int, lr: float) -> dict:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        return {"steps": 0, "losses": [], "learning_rate": None}
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    losses, grad_norms, step_seconds = [], [], []
    for epoch in range(epochs):
        order = list(indices)
        random.Random(10_000 * epoch + 29).shuffle(order)
        for index in order:
            sample = dataset[index]
            inputs, prompt_input_ids = supervised_inputs(
                processor, sample, training_answer(sample), model.device
            )
            set_external_text_anchor(model, prompt_input_ids)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs).loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            losses.append(float(loss.detach()))
            grad_norms.append(float(grad_norm))
            step_seconds.append(time.perf_counter() - started)
    return {
        "steps": len(losses),
        "epochs": epochs,
        "learning_rate": lr,
        "losses": losses,
        "loss_first_16_mean": float(np.mean(losses[:16])),
        "loss_last_16_mean": float(np.mean(losses[-16:])),
        "gradient_norm_mean": float(np.mean(grad_norms)),
        "mean_step_seconds": float(np.mean(step_seconds)),
        "total_step_seconds": float(np.sum(step_seconds)),
        "frozen_parameter_grad_tensor_count": sum(
            parameter.grad is not None
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TextVQA Stage-A is GPU-only")
    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        raise FileNotFoundError("TextVQA pilot train/validation shards are incomplete")
    seed_everything(args.seed)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = Path((RECORD_DIR / "active_model_path.txt").read_text(encoding="utf-8").strip())
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
    audit = inject_candidate(model, args.candidate)
    trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
    loaded_adapter = None
    if args.load_adapter_checkpoint is not None:
        if not trainable:
            raise RuntimeError("cannot load an adapter checkpoint into a frozen-base run")
        loaded_adapter = load_adapter_checkpoint(model, args.load_adapter_checkpoint)
    train_data = Dataset.from_parquet(str(TRAIN_FILE))
    val_data = Dataset.from_parquet(str(VAL_FILE))
    train_indices_source = None
    train_indices_sha256 = None
    if args.train_indices_file is None:
        train_indices, eligible_train_rows = select_train_indices(
            train_data, args.train_samples, args.seed, args.min_consensus
        )
    else:
        train_indices, eligible_train_rows, train_indices_sha256 = load_train_indices(
            train_data, args.train_indices_file, args.train_samples, args.min_consensus
        )
        train_indices_source = str(args.train_indices_file)
    eval_indices = select_indices(len(val_data), args.eval_samples, args.seed + 1)
    default_lr = 5.0e-3 if args.candidate in {"qh006e", "qh006d"} else 3.0e-4
    lr = args.learning_rate if args.learning_rate is not None else default_lr
    if loaded_adapter is None:
        training = train(model, processor, train_data, train_indices, args.epochs, lr)
    else:
        training = {
            "steps": 0,
            "epochs": 0,
            "learning_rate": None,
            "loaded_adapter_checkpoint": loaded_adapter,
        }
    evaluation = evaluate(
        model,
        processor,
        val_data,
        eval_indices,
        args.counterfactual_samples,
        args.max_new_tokens,
        anchor_control=args.anchor_control,
    )
    tag = "" if args.curriculum_tag is None else f"-{args.curriculum_tag}"
    control_tag = "" if args.anchor_control == "normal" else "-shuffled-question-anchor"
    suffix = "" if args.run_suffix is None else f"-{args.run_suffix}"
    run_id = (
        f"textvqa-{args.candidate}{tag}{control_tag}{suffix}"
        f"-s{args.seed}-n{len(train_indices)}"
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
        "timing_context": args.timing_context,
        "candidate_audit": audit,
        "anchor_control": args.anchor_control,
        "loaded_adapter": loaded_adapter,
        "trainable_parameters": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
        "dataset": {
            "name": "TextVQA pilot shards",
            "source": "facebook/textvqa refs/convert/parquet",
            "source_revision": "adeddaaa993f9f09c7ba460762c03628f37aa298",
            "license": "CC BY 4.0",
            "train_rows": len(train_data),
            "validation_rows": len(val_data),
            "eligible_train_rows": eligible_train_rows,
            "min_consensus": args.min_consensus,
            "train_sha256": sha256_file(TRAIN_FILE),
            "validation_sha256": sha256_file(VAL_FILE),
            "train_indices": train_indices,
            "train_indices_source": train_indices_source,
            "train_indices_sha256": train_indices_sha256,
            "curriculum_tag": args.curriculum_tag,
            "eval_indices": eval_indices,
            "test_rows_read": 0,
        },
        "training": training,
        "evaluation_after": evaluation,
        "adapter_checkpoint": str(checkpoint_path) if trainable else None,
        "gpu_peak_allocated_bytes": [
            torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())
        ],
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = ARTIFACT_DIR / f"{run_id}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "run_id": run_id,
        "trainable_parameters": result["trainable_parameters"],
        "training": training,
        "evaluation_after_metrics": evaluation["metrics"],
        "record": str(output_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
