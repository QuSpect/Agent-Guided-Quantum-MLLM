#!/usr/bin/env python3
"""Paired inference for the pre-registered QH031 512-block confirmation."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts/c4_qh031_confirmation"
LOCK = ROOT / "artifacts/qh031-confirmation-lock.json"
FILES = {
    "base": RUN_DIR / "confirmation-base-s20260847-e512.json",
    "sharedsvd1024": RUN_DIR / "confirmation-sharedsvd1024-s20260847-e512.json",
    "qh031": RUN_DIR / "confirmation-qh031-s20260847-e512.json",
    "cc031": RUN_DIR / "confirmation-cc031-s20260847-e512.json",
}
OUTPUT = RUN_DIR / "qh031-confirmation-paired-analysis.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(payload: dict, key: str = "evaluation_active") -> dict[int, float]:
    return {
        int(row["block_index"]): float(row["mean_token_nll"])
        for row in payload[key]["rows"]
    }


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    expected = [int(value) for value in lock["eval_indices"]]
    payloads, values = {}, {}
    for name, path in FILES.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["dataset_lock_sha256"] != sha256(LOCK):
            raise AssertionError(f"{name} lock hash mismatch")
        if payload["training"] != {"steps": 0, "hyperparameter_changes": 0}:
            raise AssertionError(f"{name} was modified or trained during confirmation")
        if payload["dataset"]["test_rows_inspected_or_used_for_fitness"] != 0:
            raise AssertionError(f"{name} test leakage disclosure is nonzero")
        row_map = rows(payload)
        if list(row_map) != expected:
            raise AssertionError(f"{name} evaluation order differs from lock")
        payloads[name] = payload
        values[name] = np.asarray([row_map[index] for index in expected], dtype=np.float64)
    qh_no_ent_rows = rows(payloads["qh031"], "evaluation_no_entanglement")
    values["qh031_no_ent"] = np.asarray(
        [qh_no_ent_rows[index] for index in expected], dtype=np.float64
    )
    samples = int(lock["evaluation_protocol"]["paired_bootstrap_samples"])
    seed = int(lock["seed"])
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(expected), size=(samples, len(expected)))
    sign_rng = np.random.default_rng(seed + 1)
    signs = sign_rng.choice(np.asarray([-1.0, 1.0]), size=(samples, len(expected)))
    definitions = {
        "sharedsvd1024_minus_base": ("sharedsvd1024", "base"),
        "qh031_minus_base": ("qh031", "base"),
        "cc031_minus_base": ("cc031", "base"),
        "qh031_active_minus_sharedsvd1024_own_zero": ("qh031", "sharedsvd1024"),
        "qh031_minus_cc031": ("qh031", "cc031"),
        "qh031_active_minus_no_entanglement": ("qh031", "qh031_no_ent"),
    }
    comparisons = {}
    for label, (left, right) in definitions.items():
        difference = values[left] - values[right]
        bootstrap = difference[resamples].mean(axis=1)
        observed = float(difference.mean())
        permuted = (difference[None, :] * signs).mean(axis=1)
        comparisons[label] = {
            "definition": f"NLL({left}) - NLL({right}); lower is better",
            "mean_delta_nll": observed,
            "paired_percentile_bootstrap_95_ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "paired_sign_flip_two_sided_p": float(
                (np.count_nonzero(np.abs(permuted) >= abs(observed)) + 1) / (samples + 1)
            ),
            "block_count": len(expected),
        }

    def upper(label: str) -> float:
        return comparisons[label]["paired_percentile_bootstrap_95_ci"][1]

    margin = float(lock["evaluation_protocol"]["compression_noninferiority_margin_nll"])
    reduction = int(payloads["qh031"]["injection"]["net_model_parameter_reduction"])
    gates = {
        "compression_noninferior_to_base": upper("qh031_minus_base") <= margin and reduction > 0,
        "active_better_than_own_zero": upper("qh031_active_minus_sharedsvd1024_own_zero") < 0.0,
        "quantum_better_than_equal_parameter_classical": upper("qh031_minus_cc031") < 0.0,
        "entanglement_ablation_causally_worse": upper("qh031_active_minus_no_entanglement") < 0.0,
        "net_parameter_reduction_positive": reduction > 0,
    }
    gates["all_preregistered_quantum_specific_confirmation_gates"] = all(
        gates[key] for key in (
            "compression_noninferior_to_base",
            "active_better_than_own_zero",
            "quantum_better_than_equal_parameter_classical",
            "entanglement_ablation_causally_worse",
        )
    )
    if gates["all_preregistered_quantum_specific_confirmation_gates"]:
        verdict = "qh031_passes_disjoint_development_confirmation_requires_external_test"
    else:
        verdict = "retain_qh031_compression_reject_quantum_specific_confirmation"
    output = {
        "status": "ok",
        "analysis": "QH031 frozen-checkpoint disjoint 512-block C4 confirmation",
        "verdict": verdict,
        "metric": "mean token negative log likelihood; lower is better",
        "lock": str(LOCK),
        "lock_sha256": sha256(LOCK),
        "input_sha256": {name: sha256(path) for name, path in FILES.items()},
        "bootstrap_samples": samples,
        "seed": seed,
        "test_rows_inspected_or_used_for_fitness": 0,
        "point_metrics": {
            name: {
                "mean_token_nll": float(array.mean()),
                "token_perplexity": float(math.exp(array.mean())),
            }
            for name, array in values.items()
        },
        "comparisons": comparisons,
        "gates": gates,
        "parameter_audit": {
            "net_model_parameter_reduction": reduction,
            "checkpoint_trainable_parameters": 82,
            "confirmation_trainable_parameters": 0,
        },
        "claim_limits": [
            "The architecture and sample size were informed by an earlier 64-block development experiment.",
            "This disjoint confirmation is not a final external benchmark.",
            "Exact 10-qubit GPU simulation is classically reproducible and provides no computational quantum advantage.",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

