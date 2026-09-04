#!/usr/bin/env python3
"""Prelock balanced EditCLEVR train/val indices before QH-023 predictions."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "datasets/processed/editclevr-dev"
OUTPUT = PROJECT / "artifacts/editclevr/qh023-stage-a-lock.json"
SEED = 20260836


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def balanced_indices(rows: list[dict], per_factor: int, seed: int) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["edit_factor"]].append(index)
    selected: list[int] = []
    for offset, factor in enumerate(sorted(groups)):
        generator = random.Random(seed + 1009 * offset)
        selected.extend(generator.sample(groups[factor], per_factor))
    random.Random(seed + 7919).shuffle(selected)
    return selected


def main() -> None:
    train_path = DATA / "train.jsonl"
    val_path = DATA / "val.jsonl"
    train = load_jsonl(train_path)
    val = load_jsonl(val_path)
    train_indices = balanced_indices(train, 128, SEED)
    val_indices = balanced_indices(val, 64, SEED + 1)
    reverse_indices = val_indices[:128]
    train_seeds = {train[index]["base_scene_seed"] for index in train_indices}
    val_seeds = {val[index]["base_scene_seed"] for index in val_indices}
    if train_seeds & val_seeds:
        raise RuntimeError("locked train/val base-scene overlap")
    payload = {
        "status": "locked_before_any_qh023_task_prediction",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_family": "QH-023/CC-023/DQ-023",
        "seed": SEED,
        "source": {
            "repo_revision": "734efa9b0164ba742cd2c39b9c2945c494497c23",
            "train_jsonl_sha256": sha256_file(train_path),
            "val_jsonl_sha256": sha256_file(val_path),
        },
        "train_indices": train_indices,
        "val_indices": val_indices,
        "reverse_counterfactual_val_indices": reverse_indices,
        "counts": {
            "train": len(train_indices),
            "val": len(val_indices),
            "reverse_counterfactual": len(reverse_indices),
        },
        "factor_counts": {
            "train": dict(Counter(train[index]["edit_factor"] for index in train_indices)),
            "val": dict(Counter(val[index]["edit_factor"] for index in val_indices)),
        },
        "train_val_base_scene_overlap": 0,
        "primary_metric": "three_component_exact_accuracy",
        "secondary_metrics": [
            "factor_accuracy", "old_value_accuracy", "new_value_accuracy",
            "macro_factor_exact", "reverse_direction_exact", "pair_dependency_gain"
        ],
        "single_seed_gate": {
            "candidate_must_exceed": ["frozen_base", "equal_parameter_CC023"],
            "paired_bootstrap_ci_lower_must_exceed": 0.0,
            "pair_dependency_must_not_regress": True,
        },
        "forbidden_before_gate_pass": [
            "test_id", "test_noop", "test_hard", "test_cogent",
            "more_seeds", "no_entanglement_performance", "frozen_random_performance"
        ],
        "test_rows_used_for_fitness": 0,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
