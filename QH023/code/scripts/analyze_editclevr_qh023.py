#!/usr/bin/env python3
"""Paired bootstrap and preregistered gate for EditCLEVR QH-023 triplet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--quantum", type=Path, required=True)
    parser.add_argument("--classical", type=Path, required=True)
    parser.add_argument("--static-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260836)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_id(record: dict) -> dict[str, dict]:
    return {row["pair_id"]: row for row in record["evaluation_after"]["predictions"]}


def paired(values_a: np.ndarray, values_b: np.ndarray, rng, samples: int) -> dict:
    differences = values_a.astype(float) - values_b.astype(float)
    n = len(differences)
    draws = differences[rng.integers(0, n, size=(samples, n))].mean(axis=1)
    return {
        "estimate": float(differences.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "probability_gt_zero": float(np.mean(draws > 0)),
        "wins": int(np.sum(differences > 0)),
        "losses": int(np.sum(differences < 0)),
        "ties": int(np.sum(differences == 0)),
    }


def metric_vector(rows: dict[str, dict], ids: list[str], key: str) -> np.ndarray:
    return np.asarray([bool(rows[pair_id][key]) for pair_id in ids], dtype=float)


def main() -> None:
    args = parse_args()
    records = {name: load(path) for name, path in (
        ("base", args.base), ("quantum", args.quantum), ("classical", args.classical)
    )}
    for record in records.values():
        if record["dataset"]["test_rows_used_for_fitness"] != 0:
            raise RuntimeError("test leakage disclosure is nonzero")
    indexed = {name: rows_by_id(record) for name, record in records.items()}
    id_sets = [set(rows) for rows in indexed.values()]
    if not all(value == id_sets[0] for value in id_sets[1:]):
        raise RuntimeError("triplet pair IDs differ")
    ids = [row["pair_id"] for row in records["base"]["evaluation_after"]["predictions"]]
    rng = np.random.default_rng(args.seed)
    vectors = {
        name: metric_vector(rows, ids, "original_exact") for name, rows in indexed.items()
    }
    comparisons = {
        "quantum_minus_base": paired(vectors["quantum"], vectors["base"], rng, args.bootstrap_samples),
        "classical_minus_base": paired(vectors["classical"], vectors["base"], rng, args.bootstrap_samples),
        "quantum_minus_classical": paired(vectors["quantum"], vectors["classical"], rng, args.bootstrap_samples),
    }
    reverse_ids = [pair_id for pair_id in ids if "reverse_exact" in indexed["base"][pair_id]]
    reverse = {
        name: metric_vector(rows, reverse_ids, "reverse_exact") for name, rows in indexed.items()
    }
    reverse_comparisons = {
        "quantum_minus_base": paired(reverse["quantum"], reverse["base"], rng, args.bootstrap_samples),
        "quantum_minus_classical": paired(reverse["quantum"], reverse["classical"], rng, args.bootstrap_samples),
    }
    static = load(args.static_audit)
    qh_dq = static["qh_dq_equivalence"]
    gate = {
        "quantum_exceeds_base": comparisons["quantum_minus_base"]["estimate"] > 0,
        "quantum_exceeds_classical": comparisons["quantum_minus_classical"]["estimate"] > 0,
        "base_ci_lower_gt_zero": comparisons["quantum_minus_base"]["ci95"][0] > 0,
        "classical_ci_lower_gt_zero": comparisons["quantum_minus_classical"]["ci95"][0] > 0,
        "pair_dependency_not_regressed_vs_base": (
            records["quantum"]["evaluation_after"]["metrics"]["pair_dependency_gain"]
            >= records["base"]["evaluation_after"]["metrics"]["pair_dependency_gain"]
        ),
        "exactly_dequantized": (
            qh_dq["forward_max_absolute_error"] < 2e-5
            and qh_dq["parameter_gradient_max_absolute_error"] < 1e-4
        ),
    }
    gate["passed"] = all([
        gate["quantum_exceeds_base"],
        gate["quantum_exceeds_classical"],
        gate["base_ci_lower_gt_zero"],
        gate["classical_ci_lower_gt_zero"],
        gate["pair_dependency_not_regressed_vs_base"],
    ])
    payload = {
        "status": "ok",
        "candidate": "QH-023",
        "paired_n": len(ids),
        "reverse_paired_n": len(reverse_ids),
        "metrics": {
            name: record["evaluation_after"]["metrics"] for name, record in records.items()
        },
        "comparisons": comparisons,
        "reverse_comparisons": reverse_comparisons,
        "gate": gate,
        "decision": "promote_to_causal_controls" if gate["passed"] else "reject_current_configuration",
        "claim_ceiling": "quantum_executable_structure_exactly_dequantizable",
        "training_resources": {
            name: {
                "trainable_parameters": record["trainable_parameters"],
                "total_step_seconds": record["training"].get("total_step_seconds"),
                "mean_step_seconds": record["training"].get("mean_step_seconds"),
                "mean_generation_seconds": record["evaluation_after"]["metrics"]["mean_generation_seconds"],
                "gpu_peak_allocated_bytes": record["gpu_peak_allocated_bytes"],
            }
            for name, record in records.items()
        },
        "immutable_inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in (args.base, args.quantum, args.classical, args.static_audit)
        ],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "test_rows_used_for_fitness": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
