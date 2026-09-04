#!/usr/bin/env python3
"""Paired effect estimates for Stage-A ScienceQA candidate records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def paired_bootstrap(delta: np.ndarray, samples: int, seed: int) -> dict:
    generator = np.random.default_rng(seed)
    n = len(delta)
    values = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        width = min(1000, samples - start)
        indices = generator.integers(0, n, size=(width, n))
        values[start : start + width] = delta[indices].mean(axis=1)
    return {
        "estimate": float(delta.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "bootstrap_probability_gt_zero": float(np.mean(values > 0.0)),
    }


def exact_mcnemar(base_correct: np.ndarray, candidate_correct: np.ndarray) -> dict:
    base_only = int(np.sum(base_correct & ~candidate_correct))
    candidate_only = int(np.sum(~base_correct & candidate_correct))
    discordant = base_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(base_only, candidate_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "base_only_correct": base_only,
        "candidate_only_correct": candidate_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def indexed_predictions(record: dict) -> dict[int, dict]:
    evaluation = record["evaluation_after"]
    return {int(row["dataset_index"]): row for row in evaluation["predictions"]}


def main() -> None:
    args = parse_args()
    base_record = json.loads(args.base.read_text(encoding="utf-8"))
    candidate_record = json.loads(args.candidate.read_text(encoding="utf-8"))
    if base_record["dataset"]["validation_sha256"] != candidate_record["dataset"]["validation_sha256"]:
        raise ValueError("validation dataset hashes differ")
    base = indexed_predictions(base_record)
    candidate = indexed_predictions(candidate_record)
    shared = sorted(set(base) & set(candidate))
    if shared != sorted(base) or shared != sorted(candidate):
        raise ValueError("paired comparison requires exactly matching evaluation row IDs")

    base_correct = np.array([base[index]["original_correct"] for index in shared], dtype=bool)
    candidate_correct = np.array(
        [candidate[index]["original_correct"] for index in shared], dtype=bool
    )
    accuracy_delta = candidate_correct.astype(float) - base_correct.astype(float)
    primary = paired_bootstrap(accuracy_delta, args.bootstrap_samples, args.seed)
    primary["base_accuracy"] = float(base_correct.mean())
    primary["candidate_accuracy"] = float(candidate_correct.mean())
    primary["delta_percentage_points"] = 100.0 * primary["estimate"]
    primary["ci95_percentage_points"] = [100.0 * x for x in primary["ci95"]]

    cf_shared = [
        index
        for index in shared
        if "blank_correct" in base[index] and "blank_correct" in candidate[index]
    ]
    base_gain = np.array(
        [float(base[index]["original_correct"]) - float(base[index]["blank_correct"]) for index in cf_shared]
    )
    candidate_gain = np.array(
        [
            float(candidate[index]["original_correct"])
            - float(candidate[index]["blank_correct"])
            for index in cf_shared
        ]
    )
    multimodal = paired_bootstrap(
        candidate_gain - base_gain, args.bootstrap_samples, args.seed + 1
    )
    multimodal.update(
        {
            "base_multimodal_gain": float(base_gain.mean()),
            "candidate_multimodal_gain": float(candidate_gain.mean()),
            "delta_percentage_points": 100.0 * multimodal["estimate"],
            "ci95_percentage_points": [100.0 * x for x in multimodal["ci95"]],
        }
    )

    result = {
        "status": "ok",
        "base_run_id": base_record["run_id"],
        "candidate_run_id": candidate_record["run_id"],
        "paired_n": len(shared),
        "counterfactual_paired_n": len(cf_shared),
        "accuracy": primary,
        "mcnemar": exact_mcnemar(base_correct, candidate_correct),
        "multimodal_gain_original_minus_blank": multimodal,
        "interpretation": {
            "accuracy_ci_excludes_zero": primary["ci95"][0] > 0.0 or primary["ci95"][1] < 0.0,
            "multimodal_gain_ci_excludes_zero": multimodal["ci95"][0] > 0.0
            or multimodal["ci95"][1] < 0.0,
            "confirmatory_claim_allowed": False,
            "reason": "single-seed Stage-A remains exploratory",
        },
    }
    output = args.output or args.candidate.with_name(
        f"compare-{candidate_record['candidate']}-vs-base-s{candidate_record['seed']}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
