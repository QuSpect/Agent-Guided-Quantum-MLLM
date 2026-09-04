#!/usr/bin/env python3
"""Extract only EditCLEVR train/val image members and lock VQA-style rows."""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "datasets/raw/editclevr-dev"
PROCESSED = PROJECT / "datasets/processed/editclevr-dev"
METADATA = PROCESSED / "metadata/splits.json"
ARCHIVE = RAW / "editclevr_atomic_id.tar.gz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_answer(row: dict) -> str:
    return f"{row['edit_factor']}|{row['old_value']}|{row['new_value']}"


def locked_row(row: dict, split: str) -> dict:
    edited = row["objects_before"][int(row["edited_object_id"])]
    stable = {
        key: edited[key]
        for key in ("shape", "color", "material", "size")
        if key != row["edit_factor"]
    }
    return {
        "pair_id": row["pair_id"],
        "split": split,
        "before_image": row["before_image"],
        "after_image": row["after_image"],
        "prompt": (
            "Compare the BEFORE and AFTER images. Exactly one object property changed. "
            "Answer only as factor|old_value|new_value using lowercase words."
        ),
        "answer": canonical_answer(row),
        "edit_factor": row["edit_factor"],
        "old_value": row["old_value"],
        "new_value": row["new_value"],
        "edited_object_stable_attributes": stable,
        "edited_object_id_for_scoring_only": int(row["edited_object_id"]),
        "num_objects": int(row["difficulty"]["num_objects"]),
        "occlusion_level": row["difficulty"]["occlusion_level"],
        "cogent_condition": row["difficulty"]["cogent_condition"],
        "base_scene_seed": int(row["generation"]["base_scene_seed"]),
    }


def main() -> None:
    all_splits = json.loads(METADATA.read_text(encoding="utf-8"))
    rows = {
        split: [locked_row(row, split) for row in all_splits[split]]
        for split in ("train", "val")
    }
    train_ids = {row["pair_id"] for row in rows["train"]}
    val_ids = {row["pair_id"] for row in rows["val"]}
    train_seeds = {row["base_scene_seed"] for row in rows["train"]}
    val_seeds = {row["base_scene_seed"] for row in rows["val"]}
    if train_ids & val_ids or train_seeds & val_seeds:
        raise RuntimeError("train/val pair or base-scene leakage")
    required_images = {
        row[key]
        for split_rows in rows.values()
        for row in split_rows
        for key in ("before_image", "after_image")
    }
    image_root = PROCESSED / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    # Sequential tar traversal sees member headers for the combined atomic
    # archive, but writes and decodes only exact train/val image paths.
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:
            if member.name not in required_images:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"missing payload for {member.name}")
            destination = image_root / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    handle.write(chunk)
            found.add(member.name)
    missing = sorted(required_images - found)
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} development images")
    image_records = []
    for relative in sorted(required_images):
        path = image_root / relative
        image_records.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    for split, split_rows in rows.items():
        path = PROCESSED / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    image_manifest = PROCESSED / "development-image-manifest.json"
    image_manifest.write_text(json.dumps(image_records, indent=2), encoding="utf-8")
    summary = {
        "status": "development_only_prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": "734efa9b0164ba742cd2c39b9c2945c494497c23",
        "source_archive_sha256": sha256_file(ARCHIVE),
        "splits_sha256": sha256_file(METADATA),
        "counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "image_count": len(image_records),
        "train_val_pair_overlap": len(train_ids & val_ids),
        "train_val_base_scene_seed_overlap": len(train_seeds & val_seeds),
        "factor_counts": {
            split: dict(Counter(row["edit_factor"] for row in split_rows))
            for split, split_rows in rows.items()
        },
        "answer_format": "factor|old_value|new_value",
        "image_manifest_sha256": sha256_file(image_manifest),
        "train_jsonl_sha256": sha256_file(PROCESSED / "train.jsonl"),
        "val_jsonl_sha256": sha256_file(PROCESSED / "val.jsonl"),
        "protocol_disclosure": {
            "combined_atomic_archive_headers_streamed": True,
            "test_member_payloads_materialized": 0,
            "test_images_decoded": 0,
            "test_rows_used_for_fitness": 0,
            "separate_test_noop_hard_cogent_archives_downloaded": False,
        },
    }
    output = PROJECT / "artifacts/editclevr-dev-preparation.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
