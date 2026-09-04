#!/usr/bin/env python3
"""Audit train/validation identity overlap without decoding image payloads."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[2]
TRAIN_FILE = PROJECT_DIR / "datasets/raw/textvqa/textvqa/train/0000.parquet"
VAL_FILE = PROJECT_DIR / "datasets/raw/textvqa/textvqa/validation/0000.parquet"


def normalize_question(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def hash_values(values: set[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_metadata(path: Path) -> dict[str, set[str]]:
    table = pq.read_table(path, columns=["question_id", "image_id", "question"])
    question_ids = {str(value) for value in table["question_id"].to_pylist()}
    image_ids = {str(value) for value in table["image_id"].to_pylist()}
    questions = {normalize_question(str(value)) for value in table["question"].to_pylist()}
    return {
        "question_ids": question_ids,
        "image_ids": image_ids,
        "normalized_questions": questions,
    }


def main() -> None:
    train = load_metadata(TRAIN_FILE)
    validation = load_metadata(VAL_FILE)
    result = {
        "status": "pass",
        "scope": "TextVQA train/0000.parquet vs validation/0000.parquet",
        "images_decoded": 0,
        "test_rows_read": 0,
        "train_rows": pq.read_metadata(TRAIN_FILE).num_rows,
        "validation_rows": pq.read_metadata(VAL_FILE).num_rows,
        "unique_counts": {
            "train": {name: len(values) for name, values in train.items()},
            "validation": {name: len(values) for name, values in validation.items()},
        },
        "overlap_counts": {
            name: len(train[name] & validation[name]) for name in train
        },
        "set_sha256": {
            "train": {name: hash_values(values) for name, values in train.items()},
            "validation": {
                name: hash_values(values) for name, values in validation.items()
            },
        },
    }
    if result["overlap_counts"]["question_ids"] != 0:
        result["status"] = "fail"
    path = PROJECT_DIR / "artifacts/textvqa_stage_a/textvqa-pilot-overlap-audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
