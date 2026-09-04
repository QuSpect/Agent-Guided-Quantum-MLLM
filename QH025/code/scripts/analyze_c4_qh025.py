#!/usr/bin/env python3
"""Locked paired statistics for the C4 QH-025 true-layer replacement study."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts/c4_qh025"
LOCK_PATH = ROOT / "artifacts/wikitext-qh025-end-to-end-lock.json"
FILES = {
    "base": RUN_DIR / "c4-base-s20260839-n1024-e64.json",
    "svd512": RUN_DIR / "c4-svd512-s20260839-n1024-e64.json",
    "qh025": RUN_DIR / "c4-qh025-s20260839-n1024-e64.json",
    "cc025": RUN_DIR / "c4-cc025-s20260839-n1024-e64.json",
}
OUTPUT = RUN_DIR / "c4-qh025-paired-analysis.json"


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
    seed = int(lock["seed"])
    samples = int(lock["evaluation_protocol"]["paired_bootstrap_samples"])
    margin = float(lock["evaluation_protocol"]["compression_noninferiority_margin_nll"])
    expected = [int(value) for value in lock["eval_indices"]]
    payloads, rows = {}, {}
    for name, path in FILES.items():
        payloads[name], rows[name] = load_rows(path)
        if list(rows[name]) != expected:
            raise AssertionError(f"{name} evaluation order differs from the lock")
        if payloads[name]["dataset_lock_sha256"] != sha256(LOCK_PATH):
            raise AssertionError(f"{name} lock hash mismatch")
        if payloads[name]["dataset"]["test_rows_inspected_or_used_for_fitness"] != 0:
            raise AssertionError(f"{name} test leakage disclosure is nonzero")

    _, qh_zero = load_rows(FILES["qh025"], "evaluation_zero_residual")
    _, cc_zero = load_rows(FILES["cc025"], "evaluation_zero_residual")
    if qh_zero != rows["svd512"] or cc_zero != rows["svd512"]:
        raise AssertionError("own zero branches do not exactly match SVD-512")
    arrays = {
        name: np.asarray([values[index] for index in expected], dtype=np.float64)
        for name, values in rows.items()
    }
    arrays["qh025_zero"] = np.asarray([qh_zero[index] for index in expected], dtype=np.float64)

    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(expected), size=(samples, len(expected)))
    sign_rng = np.random.default_rng(seed + 1)
    signs = sign_rng.choice(np.asarray([-1.0, 1.0]), size=(samples, len(expected)))
    comparisons = {
        "svd512_minus_base": ("svd512", "base"),
        "qh025_minus_base": ("qh025", "base"),
        "cc025_minus_base": ("cc025", "base"),
        "qh025_minus_svd512": ("qh025", "svd512"),
        "cc025_minus_svd512": ("cc025", "svd512"),
        "qh025_minus_cc025": ("qh025", "cc025"),
        "qh025_active_minus_own_zero": ("qh025", "qh025_zero"),
    }
    results = {}
    for label, (left, right) in comparisons.items():
        differences = arrays[left] - arrays[right]
        boot = differences[resamples].mean(axis=1)
        observed = float(differences.mean())
        permuted = (differences[None, :] * signs).mean(axis=1)
        results[label] = {
            "definition": f"mean token NLL({left}) - mean token NLL({right}); lower is better",
            "mean_delta_nll": observed,
            "paired_percentile_bootstrap_95_ci": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
            "paired_sign_flip_two_sided_p": float(
                (np.count_nonzero(np.abs(permuted) >= abs(observed)) + 1) / (samples + 1)
            ),
            "block_count": len(expected),
        }

    svd_base_upper = results["svd512_minus_base"]["paired_percentile_bootstrap_95_ci"][1]
    qh_base_upper = results["qh025_minus_base"]["paired_percentile_bootstrap_95_ci"][1]
    causal_upper = results["qh025_active_minus_own_zero"]["paired_percentile_bootstrap_95_ci"][1]
    quantum_upper = results["qh025_minus_cc025"]["paired_percentile_bootstrap_95_ci"][1]
    reduction = int(payloads["qh025"]["injection"]["net_model_parameter_reduction"])
    gates = {
        "svd_compression_noninferior_to_base": bool(svd_base_upper <= margin),
        "qh025_compression_noninferior_to_base": bool(qh_base_upper <= margin and reduction > 0),
        "qh025_strictly_better_than_base": bool(qh_base_upper < 0.0),
        "qh025_causal_active_better_than_own_zero": bool(causal_upper < 0.0),
        "qh025_quantum_better_than_equal_parameter_classical": bool(quantum_upper < 0.0),
        "net_model_parameter_reduction_positive": bool(reduction > 0),
    }
    gates["quantum_specific_promotion"] = bool(
        gates["qh025_compression_noninferior_to_base"]
        and gates["qh025_causal_active_better_than_own_zero"]
        and gates["qh025_quantum_better_than_equal_parameter_classical"]
    )
    if gates["quantum_specific_promotion"]:
        verdict = "promote_qh025_quantum_specific_replacement"
    elif gates["qh025_compression_noninferior_to_base"]:
        verdict = "retain_compressed_replacement_reject_quantum_specific_claim"
    else:
        verdict = "reject_qh025_compression_route_on_c4"

    output = {
        "status": "ok",
        "analysis": "QH-025 locked paired C4 evaluation",
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
        "training_diagnostics": {
            name: {
                key: value
                for key, value in payloads[name]["training"].items()
                if key != "losses"
            }
            for name in ("qh025", "cc025")
        },
        "comparisons": results,
        "gates": gates,
        "parameter_audit": {
            "base_model_parameters": payloads["qh025"]["injection"]["base_model_parameters"],
            "deployed_model_parameters": payloads["qh025"]["injection"]["deployed_model_parameters"],
            "net_model_parameter_reduction": reduction,
            "trainable_parameters_qh025": payloads["qh025"]["injection"]["trainable_parameter_count"],
            "trainable_parameters_cc025": payloads["cc025"]["injection"]["trainable_parameter_count"],
        },
        "claim_limits": [
            "A small exact statevector circuit is classically dequantizable.",
            "A point estimate is not sufficient; quantum attribution requires both locked CI gates.",
            "This is a 64-block C4 development evaluation, not a final test benchmark.",
            "No quantum hardware or simulator speed advantage is claimed.",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
