#!/usr/bin/env python3
"""Paired, lock-aware statistical analysis for the QH-024 replacement study."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
RUN_DIR = ARTIFACTS / "wikitext_qh024"
LOCK_PATH = ARTIFACTS / "wikitext-qh024-distillation-lock.json"
FILES = {
    "base": RUN_DIR / "wikitext-base-s20260838-n512-e32.json",
    "svd512": RUN_DIR / "wikitext-svd512-s20260838-n512-e32.json",
    "qh024": RUN_DIR / "wikitext-qh024-s20260838-n512-e32.json",
    "cc024": RUN_DIR / "wikitext-cc024-s20260838-n512-e32.json",
}
OUTPUT = RUN_DIR / "wikitext-qh024-paired-analysis.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path, key: str = "evaluation_active") -> tuple[dict, dict[int, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {
        int(row["block_index"]): float(row["mean_token_nll"])
        for row in payload[key]["rows"]
    }
    return payload, rows


def main() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    samples = int(lock["evaluation_protocol"]["paired_bootstrap_samples"])
    margin = float(lock["evaluation_protocol"]["compression_noninferiority_margin_nll"])
    seed = int(lock["seed"])
    payloads: dict[str, dict] = {}
    rows: dict[str, dict[int, float]] = {}
    for name, path in FILES.items():
        payloads[name], rows[name] = load_rows(path)

    expected = [int(value) for value in lock["eval_indices"]]
    for name, values in rows.items():
        if list(values) != expected:
            raise AssertionError(f"{name} evaluation order does not match the lock")
        if payloads[name]["dataset_lock_sha256"] != sha256(LOCK_PATH):
            raise AssertionError(f"{name} lock hash mismatch")
        if payloads[name]["dataset"]["test_rows_inspected_or_used_for_fitness"] != 0:
            raise AssertionError(f"{name} reports test leakage")

    qh_zero_payload, qh_zero = load_rows(FILES["qh024"], "evaluation_zero_residual")
    cc_zero_payload, cc_zero = load_rows(FILES["cc024"], "evaluation_zero_residual")
    del qh_zero_payload, cc_zero_payload
    if qh_zero != rows["svd512"] or cc_zero != rows["svd512"]:
        raise AssertionError("active-checkpoint zero branch is not identical to SVD-512")

    arrays = {
        name: np.asarray([values[index] for index in expected], dtype=np.float64)
        for name, values in rows.items()
    }
    arrays["qh024_zero"] = np.asarray([qh_zero[index] for index in expected], dtype=np.float64)

    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, len(expected), size=(samples, len(expected)))
    sign_rng = np.random.default_rng(seed + 1)
    signs = sign_rng.choice(np.asarray([-1.0, 1.0]), size=(samples, len(expected)))

    comparisons = {
        "svd512_minus_base": ("svd512", "base"),
        "qh024_minus_base": ("qh024", "base"),
        "cc024_minus_base": ("cc024", "base"),
        "qh024_minus_svd512": ("qh024", "svd512"),
        "cc024_minus_svd512": ("cc024", "svd512"),
        "qh024_minus_cc024": ("qh024", "cc024"),
        "qh024_active_minus_own_zero": ("qh024", "qh024_zero"),
    }
    results: dict[str, dict] = {}
    for label, (left, right) in comparisons.items():
        differences = arrays[left] - arrays[right]
        boot = differences[bootstrap_indices].mean(axis=1)
        observed = float(differences.mean())
        permutation = (differences[None, :] * signs).mean(axis=1)
        results[label] = {
            "definition": f"mean token NLL({left}) - mean token NLL({right}); lower is better",
            "mean_delta_nll": observed,
            "paired_percentile_bootstrap_95_ci": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
            "paired_sign_flip_two_sided_p": float(
                (np.count_nonzero(np.abs(permutation) >= abs(observed)) + 1) / (samples + 1)
            ),
            "block_count": len(expected),
        }

    qh_base_upper = results["qh024_minus_base"]["paired_percentile_bootstrap_95_ci"][1]
    svd_base_upper = results["svd512_minus_base"]["paired_percentile_bootstrap_95_ci"][1]
    causal_upper = results["qh024_active_minus_own_zero"]["paired_percentile_bootstrap_95_ci"][1]
    quantum_upper = results["qh024_minus_cc024"]["paired_percentile_bootstrap_95_ci"][1]
    reduction = int(payloads["qh024"]["injection"]["net_model_parameter_reduction"])
    gates = {
        "svd_compression_noninferior_to_base": bool(svd_base_upper <= margin),
        "qh024_compression_noninferior_to_base": bool(qh_base_upper <= margin and reduction > 0),
        "qh024_causal_active_better_than_own_zero": bool(causal_upper < 0.0),
        "qh024_quantum_better_than_equal_parameter_classical": bool(quantum_upper < 0.0),
        "net_model_parameter_reduction_positive": bool(reduction > 0),
    }
    gates["quantum_specific_promotion"] = bool(
        gates["qh024_compression_noninferior_to_base"]
        and gates["qh024_causal_active_better_than_own_zero"]
        and gates["qh024_quantum_better_than_equal_parameter_classical"]
    )
    if gates["quantum_specific_promotion"]:
        verdict = "promote_quantum_specific_replacement"
    elif gates["svd_compression_noninferior_to_base"]:
        verdict = "retain_svd_compression_route_reject_qh024_quantum_residual"
    else:
        verdict = "reject_qh024_and_svd_compression_route"

    output = {
        "status": "ok",
        "analysis": "QH-024 paired locked evaluation",
        "verdict": verdict,
        "metric": "mean token negative log likelihood; lower is better",
        "lock": str(LOCK_PATH),
        "lock_sha256": sha256(LOCK_PATH),
        "input_sha256": {name: sha256(path) for name, path in FILES.items()},
        "bootstrap_samples": samples,
        "sign_flip_samples": samples,
        "seed": seed,
        "compression_noninferiority_margin_nll": margin,
        "test_rows_inspected_or_used_for_fitness": 0,
        "point_metrics": {
            name: {
                "mean_token_nll": float(values.mean()),
                "token_perplexity": float(math.exp(values.mean())),
            }
            for name, values in arrays.items()
        },
        "comparisons": results,
        "gates": gates,
        "parameter_audit": {
            "base_model_parameters": payloads["qh024"]["injection"]["base_model_parameters"],
            "deployed_model_parameters": payloads["qh024"]["injection"]["deployed_model_parameters"],
            "net_model_parameter_reduction": reduction,
            "trainable_parameter_count_qh024": payloads["qh024"]["injection"]["trainable_parameter_count"],
            "trainable_parameter_count_cc024": payloads["cc024"]["injection"]["trainable_parameter_count"],
        },
        "interpretation_limits": [
            "The exact 10-qubit circuit is classically dequantizable at this scale.",
            "A compression noninferiority pass is not evidence that the quantum residual caused the result.",
            "The quantum-specific claim requires both active-vs-zero and QH-vs-equal-parameter-classical gates.",
            "This formal evaluation uses 32 locked WikiText validation blocks and no test rows.",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
