#!/usr/bin/env python3
"""Prepare deterministic train/validation relation pools from official CLEVR."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=(Path(__file__).resolve().parents[2] / "datasets/raw/clevr/CLEVR_v1.0.zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./datasets/clevr_relations"),
    )
    parser.add_argument("--train-pool", type=int, default=1024)
    parser.add_argument("--validation-pool", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_questions(archive: zipfile.ZipFile, split: str) -> list[dict]:
    member = f"CLEVR_v1.0/questions/CLEVR_{split}_questions.json"
    with archive.open(member) as handle:
        payload = json.load(handle)
    return payload["questions"]


def has_relation_program(sample: dict) -> bool:
    return any(step.get("function") == "relate" for step in sample.get("program", []))


def choose_pool(rows: list[dict], count: int, seed: int) -> list[dict]:
    eligible = [row for row in rows if has_relation_program(row)]
    order = list(range(len(eligible)))
    random.Random(seed).shuffle(order)
    return [eligible[index] for index in order[: min(count, len(order))]]


def materialize(
    archive: zipfile.ZipFile,
    rows: list[dict],
    split: str,
    output_dir: Path,
) -> list[dict]:
    image_dir = output_dir / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for row in rows:
        filename = row["image_filename"]
        target = image_dir / filename
        if not target.exists():
            member = f"CLEVR_v1.0/images/{split}/{filename}"
            with archive.open(member) as source, target.open("wb") as sink:
                while chunk := source.read(1 << 20):
                    sink.write(chunk)
        functions = [step.get("function") for step in row.get("program", [])]
        prepared.append(
            {
                "split": split,
                "question_index": int(row["question_index"]),
                "image_index": int(row["image_index"]),
                "question_family_index": int(row["question_family_index"]),
                "question": row["question"],
                "answer": str(row["answer"]),
                "image_filename": filename,
                "image_path": str(target),
                "program_functions": functions,
                "relation_count": functions.count("relate"),
            }
        )
    return prepared


def summarize(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "unique_images": len({row["image_index"] for row in rows}),
        "answer_distribution": dict(Counter(row["answer"] for row in rows)),
        "relation_count_distribution": dict(
            Counter(str(row["relation_count"]) for row in rows)
        ),
        "question_family_count": len(
            {row["question_family_index"] for row in rows}
        ),
    }


def main() -> None:
    args = parse_args()
    if not args.archive.exists():
        raise FileNotFoundError(args.archive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        train_all = read_questions(archive, "train")
        validation_all = read_questions(archive, "val")
        train_selected = choose_pool(train_all, args.train_pool, args.seed)
        validation_selected = choose_pool(
            validation_all, args.validation_pool, args.seed + 1
        )
        train_rows = materialize(archive, train_selected, "train", args.output_dir)
        validation_rows = materialize(
            archive, validation_selected, "val", args.output_dir
        )

    train_path = args.output_dir / "train_relation_pool.json"
    validation_path = args.output_dir / "validation_relation_pool.json"
    train_path.write_text(
        json.dumps(train_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_path.write_text(
        json.dumps(validation_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "dataset": "CLEVR v1.0",
        "source": "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
        "license": "CC BY 4.0",
        "archive": str(args.archive),
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": sha256_file(args.archive),
        "selection": "program contains at least one relate operation",
        "seed": args.seed,
        "train": summarize(train_rows),
        "validation": summarize(validation_rows),
        "test_rows_read": 0,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = args.output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
