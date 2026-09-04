#!/usr/bin/env python3
"""Mine Qwen3.8 errors from ScienceQA train only; never touches validation/test."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForMultimodalLM, AutoProcessor


PROJECT_DIR = Path(__file__).resolve().parents[2]
RECORD_DIR = PROJECT_DIR / "records"
ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "scienceqa_stage_a"
TRAIN_FILE = PROJECT_DIR / "datasets/raw/scienceqa/ScienceQA-IMG/train-00000-of-00001.parquet"
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_scienceqa_stage_a import LETTERS, predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_record(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("hard mining is GPU-only")
    torch.manual_seed(args.seed)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = Path((RECORD_DIR / "active_model_path.txt").read_text().strip())
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
    data = Dataset.from_parquet(str(TRAIN_FILE))
    generator = np.random.default_rng(args.seed)
    scan_indices = generator.permutation(len(data))[: min(args.scan_samples, len(data))].tolist()
    run_id = f"scienceqa-hard-mine-s{args.seed}-scan{len(scan_indices)}"
    output_path = ARTIFACT_DIR / f"{run_id}.json"
    wrong_indices = []
    rows = []
    category_wrong = Counter()
    total_generation_seconds = 0.0
    previous_wall_seconds = 0.0
    start_position = 0
    if args.resume_from is not None:
        previous = json.loads(args.resume_from.read_text(encoding="utf-8"))
        if previous["seed"] != args.seed:
            raise ValueError("resume seed differs")
        if previous["source_sha256"] != sha256_file(TRAIN_FILE):
            raise ValueError("resume source hash differs")
        start_position = int(previous["processed_rows"])
        if previous["processed_indices"] != scan_indices[:start_position]:
            raise ValueError("resume indices are not the prefix of this deterministic scan")
        wrong_indices = [int(index) for index in previous["wrong_indices"]]
        rows = list(previous["wrong_rows"])
        category_wrong = Counter(previous["category_wrong_counts"])
        total_generation_seconds = float(previous["total_generation_seconds"])
        previous_wall_seconds = float(previous["wall_seconds"])
    started = time.perf_counter()
    for position, index in enumerate(
        scan_indices[start_position:], start=start_position + 1
    ):
        sample = data[index]
        prediction = predict(
            model, processor, sample, sample["image"].convert("RGB"), args.max_new_tokens
        )
        answer = LETTERS[int(sample["answer"])]
        correct = prediction["letter"] == answer
        total_generation_seconds += prediction["seconds"]
        if not correct:
            wrong_indices.append(index)
            category_wrong[str(sample.get("category", "unknown"))] += 1
            rows.append(
                {
                    "dataset_index": index,
                    "answer": answer,
                    "prediction": prediction["letter"],
                    "raw": prediction["raw"],
                    "subject": sample.get("subject"),
                    "category": sample.get("category"),
                    "topic": sample.get("topic"),
                    "skill": sample.get("skill"),
                }
            )
        if position % args.checkpoint_every == 0 or position == len(scan_indices):
            record = {
                "status": "complete" if position == len(scan_indices) else "running",
                "run_id": run_id,
                "source_split": "ScienceQA-IMG/train only",
                "source_sha256": sha256_file(TRAIN_FILE),
                "seed": args.seed,
                "planned_scan_rows": len(scan_indices),
                "processed_rows": position,
                "processed_indices": scan_indices[:position],
                "wrong_indices": wrong_indices,
                "wrong_rows": rows,
                "wrong_count": len(wrong_indices),
                "accuracy_on_scanned_train": 1.0 - len(wrong_indices) / position,
                "category_wrong_counts": dict(category_wrong),
                "total_generation_seconds": total_generation_seconds,
                "wall_seconds": previous_wall_seconds + time.perf_counter() - started,
                "validation_or_test_rows_read": 0,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
            write_record(output_path, record)
            print(
                json.dumps(
                    {
                        "processed": position,
                        "wrong": len(wrong_indices),
                        "accuracy": record["accuracy_on_scanned_train"],
                        "record": str(output_path),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
