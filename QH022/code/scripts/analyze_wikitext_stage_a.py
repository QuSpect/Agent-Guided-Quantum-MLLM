#!/usr/bin/env python3
"""Paired sequence bootstrap for WikiText token-NLL artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    for key in ("source_revision", "validation_token_sha256", "eval_indices", "sequence_length"):
        if base["dataset"][key] != candidate["dataset"][key]:
            raise ValueError(f"mismatched dataset field: {key}")
    base_rows = {row["block_index"]: row for row in base["evaluation_after"]["rows"]}
    candidate_rows = {row["block_index"]: row for row in candidate["evaluation_after"]["rows"]}
    indices = base["dataset"]["eval_indices"]
    if set(indices) != set(candidate_rows):
        raise ValueError("paired evaluation rows differ")
    # Positive means candidate lowers NLL and is better.
    deltas = np.array([
        base_rows[index]["mean_token_nll"] - candidate_rows[index]["mean_token_nll"]
        for index in indices
    ])
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(deltas), size=(args.bootstrap_samples, len(deltas)))
    bootstrap = deltas[draws].mean(axis=1)
    base_nll = base["evaluation_after"]["metrics"]["mean_token_nll"]
    candidate_nll = candidate["evaluation_after"]["metrics"]["mean_token_nll"]
    payload = {
        "status": "ok",
        "base_run_id": base["run_id"],
        "candidate_run_id": candidate["run_id"],
        "paired_sequences": len(deltas),
        "mean_token_nll_reduction": {
            "estimate": float(deltas.mean()),
            "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "probability_gt_zero": float(np.mean(bootstrap > 0.0)),
            "base": base_nll,
            "candidate": candidate_nll,
        },
        "token_perplexity": {
            "base": base["evaluation_after"]["metrics"]["token_perplexity"],
            "candidate": candidate["evaluation_after"]["metrics"]["token_perplexity"],
            "relative_reduction": 1.0 - (
                candidate["evaluation_after"]["metrics"]["token_perplexity"]
                / base["evaluation_after"]["metrics"]["token_perplexity"]
            ),
        },
        "single_seed_exploratory": True,
        "confirmatory_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
