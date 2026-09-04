#!/usr/bin/env python3
"""Paired statistical analysis for TextVQA Stage-A records."""

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
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aligned_rows(record: dict) -> dict[int, dict]:
    return {
        int(row["dataset_index"]): row
        for row in record["evaluation_after"]["predictions"]
    }


def paired_bootstrap(delta: np.ndarray, samples: int, rng) -> dict:
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(delta), size=len(delta))
        draws[index] = float(delta[selected].mean())
    return {
        "estimate": float(delta.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_probability_gt_zero": float(np.mean(draws > 0.0)),
    }


def exact_mcnemar(base: np.ndarray, candidate: np.ndarray) -> dict:
    base_only = int(np.sum(base & ~candidate))
    candidate_only = int(np.sum(candidate & ~base))
    discordant = base_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(base_only, candidate_only)
        tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "base_only_exact_any": base_only,
        "candidate_only_exact_any": candidate_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def main() -> None:
    args = parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    base_rows = aligned_rows(base)
    candidate_rows = aligned_rows(candidate)
    if set(base_rows) != set(candidate_rows):
        raise ValueError("candidate and base evaluation rows are not identical")
    indices = sorted(base_rows)
    base_quality = np.asarray(
        [base_rows[index]["original_soft_score"] for index in indices], dtype=np.float64
    )
    candidate_quality = np.asarray(
        [candidate_rows[index]["original_soft_score"] for index in indices], dtype=np.float64
    )
    cf_indices = [index for index in indices if "blank_soft_score" in base_rows[index]]
    if set(cf_indices) != {
        index for index in indices if "blank_soft_score" in candidate_rows[index]
    }:
        raise ValueError("candidate and base counterfactual rows are not identical")
    base_visual = np.asarray([
        base_rows[index]["original_soft_score"] - base_rows[index]["blank_soft_score"]
        for index in cf_indices
    ], dtype=np.float64)
    candidate_visual = np.asarray([
        candidate_rows[index]["original_soft_score"]
        - candidate_rows[index]["blank_soft_score"]
        for index in cf_indices
    ], dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    quality = paired_bootstrap(
        candidate_quality - base_quality, args.bootstrap_samples, rng
    )
    visual = paired_bootstrap(
        candidate_visual - base_visual, args.bootstrap_samples, rng
    )
    base_exact = np.asarray(
        [base_rows[index]["original_exact_any"] for index in indices], dtype=bool
    )
    candidate_exact = np.asarray(
        [candidate_rows[index]["original_exact_any"] for index in indices], dtype=bool
    )
    base_p95 = None
    candidate_p95 = None
    if all("original_seconds" in base_rows[index] for index in indices) and all(
        "original_seconds" in candidate_rows[index] for index in indices
    ):
        base_p95 = float(
            np.quantile([base_rows[index]["original_seconds"] for index in indices], 0.95)
        )
        candidate_p95 = float(
            np.quantile(
                [candidate_rows[index]["original_seconds"] for index in indices], 0.95
            )
        )
    latency_claim_valid = (
        base.get("timing_context") == "exclusive"
        and candidate.get("timing_context") == "exclusive"
    )
    p95_increase = (
        None
        if base_p95 in (None, 0.0) or candidate_p95 is None
        else candidate_p95 / base_p95 - 1.0
    )
    result = {
        "status": "ok",
        "base_run_id": base["run_id"],
        "candidate_run_id": candidate["run_id"],
        "paired_n": len(indices),
        "counterfactual_paired_n": len(cf_indices),
        "vqa_soft_accuracy": {
            **quality,
            "base": float(base_quality.mean()),
            "candidate": float(candidate_quality.mean()),
            "delta_percentage_points": 100.0 * quality["estimate"],
            "ci95_percentage_points": [100.0 * value for value in quality["ci95"]],
        },
        "multimodal_gain_original_minus_blank": {
            **visual,
            "base": float(base_visual.mean()),
            "candidate": float(candidate_visual.mean()),
            "delta_percentage_points": 100.0 * visual["estimate"],
            "ci95_percentage_points": [100.0 * value for value in visual["ci95"]],
        },
        "exact_any_mcnemar": exact_mcnemar(base_exact, candidate_exact),
        "latency": {
            "base_timing_context": base.get("timing_context"),
            "candidate_timing_context": candidate.get("timing_context"),
            "valid_for_claim": latency_claim_valid,
            "base_p95_seconds": base_p95,
            "candidate_p95_seconds": candidate_p95,
            "p95_increase_fraction_debug": p95_increase,
            "p95_increase_fraction_for_claim": (
                p95_increase if latency_claim_valid else None
            ),
        },
        "interpretation": {
            "single_seed_exploratory": True,
            "quality_ci_excludes_zero": quality["ci95"][0] > 0.0,
            "multimodal_gain_not_degraded": visual["estimate"] >= 0.0,
            "confirmatory_claim_allowed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
