#!/usr/bin/env python3
"""Mine TextVQA train-only failures of frozen Qwen3.8 without touching validation/test."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForMultimodalLM, AutoProcessor

from run_textvqa_stage_a import (
    ANSWER_PROCESSOR,
    RECORD_DIR,
    TRAIN_FILE,
    normalized_answer_counts,
    predict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=1900)
    parser.add_argument("--min-consensus", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/textvqa_stage_a/textvqa-hard-mine-s20260829.json"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TextVQA hard mining requires CUDA")
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
    model.eval()
    dataset = Dataset.from_parquet(str(TRAIN_FILE))
    answer_only = dataset.select_columns(["answers"])
    eligible = [
        index for index in range(min(args.max_rows, len(answer_only)))
        if max(normalized_answer_counts(answer_only[index]).values()) >= args.min_consensus
    ]
    rows: list[dict] = []
    for position, index in enumerate(eligible, start=1):
        sample = dataset[index]
        result = predict(
            model, processor, sample, sample["image"].convert("RGB"), args.max_new_tokens
        )
        rows.append({
            "dataset_index": index,
            "question_id": int(sample["question_id"]),
            "prediction": result["raw"],
            "normalized_prediction": result["normalized"],
            "soft_score": result["soft_score"],
            "exact_any": result["exact_any"],
            "seconds": result["seconds"],
        })
        if position % args.checkpoint_every == 0 or position == len(eligible):
            selected = [
                row["dataset_index"]
                for row in sorted(rows, key=lambda row: (row["soft_score"], row["dataset_index"]))
                if row["soft_score"] < 0.6
            ]
            payload = {
                "status": "complete" if position == len(eligible) else "checkpoint",
                "dataset": "facebook/textvqa train/0000.parquet only",
                "source_revision": "adeddaaa993f9f09c7ba460762c03628f37aa298",
                "train_sha256": sha256_file(TRAIN_FILE),
                "model_dir": str(model_dir),
                "rows_scanned": position,
                "eligible_rows": len(eligible),
                "min_consensus": args.min_consensus,
                "selection_rule": "frozen-base VQA soft score < 0.6, sorted by score then index",
                "selected_indices": selected,
                "soft_accuracy_scanned": float(np.mean([row["soft_score"] for row in rows])),
                "exact_any_accuracy_scanned": float(np.mean([row["exact_any"] for row in rows])),
                "validation_rows_read": 0,
                "test_rows_read": 0,
                "predictions": rows,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            save(payload, args.output)
    print(json.dumps({
        "status": "complete",
        "rows_scanned": len(rows),
        "selected_rows": len(payload["selected_indices"]),
        "soft_accuracy": payload["soft_accuracy_scanned"],
        "exact_any_accuracy": payload["exact_any_accuracy_scanned"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
