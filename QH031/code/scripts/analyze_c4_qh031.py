#!/usr/bin/env python3
"""Locked paired statistics for QH031 shared-basis replacement."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts/c4_qh031"
LOCK_PATH = ROOT / "artifacts/qh031-shared-basis-lock.json"
FILES = {
    "base": RUN_DIR / "c4-base-s20260843-n1024-e64.json",
    "sharedsvd1024": RUN_DIR / "c4-sharedsvd1024-s20260843-n1024-e64.json",
    "qh031": RUN_DIR / "c4-qh031-s20260843-n1024-e64.json",
    "cc031": RUN_DIR / "c4-cc031-s20260843-n1024-e64.json",
}
OUTPUT = RUN_DIR / "c4-qh031-paired-analysis.json"


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
    expected = [int(value) for value in lock["eval_indices"]]
    samples = int(lock["evaluation_protocol"]["paired_bootstrap_samples"])
    margin = float(lock["evaluation_protocol"]["compression_noninferiority_margin_nll"])
    seed = int(lock["seed"])
    payloads, row_maps = {}, {}
    for name, path in FILES.items():
        payloads[name], row_maps[name] = load_rows(path)
        if list(row_maps[name]) != expected:
            raise AssertionError(f"{name} evaluation order differs from lock")
        if payloads[name]["dataset_lock_sha256"] != sha256(LOCK_PATH):
            raise AssertionError(f"{name} lock hash mismatch")
        if payloads[name]["dataset"]["test_rows_inspected_or_used_for_fitness"] != 0:
            raise AssertionError(f"{name} test leakage disclosure is nonzero")

    _, qh_zero = load_rows(FILES["qh031"], "evaluation_zero_shared_svd")
    _, cc_zero = load_rows(FILES["cc031"], "evaluation_zero_shared_svd")
    _, qh_no_ent = load_rows(FILES["qh031"], "evaluation_no_entanglement")
    _, qh_dq = load_rows(FILES["qh031"], "evaluation_dequantized_audit")
    arrays = {
        name: np.asarray([rows[index] for index in expected], dtype=np.float64)
        for name, rows in row_maps.items()
    }
    arrays.update({
        "qh031_zero": np.asarray([qh_zero[index] for index in expected]),
        "cc031_zero": np.asarray([cc_zero[index] for index in expected]),
        "qh031_no_ent": np.asarray([qh_no_ent[index] for index in expected]),
        "qh031_dq": np.asarray([qh_dq[index] for index in expected]),
    })
    zero_max_abs = {
        "qh_vs_shared": float(np.max(np.abs(arrays["qh031_zero"] - arrays["sharedsvd1024"]))),
        "cc_vs_shared": float(np.max(np.abs(arrays["cc031_zero"] - arrays["sharedsvd1024"]))),
    }
    if max(zero_max_abs.values()) > 1.0e-8:
        raise AssertionError(f"own-zero branch differs from shared SVD: {zero_max_abs}")

    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(expected), size=(samples, len(expected)))
    sign_rng = np.random.default_rng(seed + 1)
    signs = sign_rng.choice(np.asarray([-1.0, 1.0]), size=(samples, len(expected)))
    comparisons = {
        "sharedsvd1024_minus_base": ("sharedsvd1024", "base"),
        "qh031_minus_base": ("qh031", "base"),
        "cc031_minus_base": ("cc031", "base"),
        "qh031_minus_sharedsvd1024": ("qh031", "sharedsvd1024"),
        "cc031_minus_sharedsvd1024": ("cc031", "sharedsvd1024"),
        "qh031_minus_cc031": ("qh031", "cc031"),
        "qh031_active_minus_own_zero": ("qh031", "qh031_zero"),
        "qh031_active_minus_no_entanglement": ("qh031", "qh031_no_ent"),
        "qh031_quantum_minus_dequantized_audit": ("qh031", "qh031_dq"),
    }
    results = {}
    for label, (left, right) in comparisons.items():
        differences = arrays[left] - arrays[right]
        bootstrap = differences[resamples].mean(axis=1)
        observed = float(differences.mean())
        permuted = (differences[None, :] * signs).mean(axis=1)
        results[label] = {
            "definition": f"mean token NLL({left}) - mean token NLL({right}); lower is better",
            "mean_delta_nll": observed,
            "paired_percentile_bootstrap_95_ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "paired_sign_flip_two_sided_p": float(
                (np.count_nonzero(np.abs(permuted) >= abs(observed)) + 1)
                / (samples + 1)
            ),
            "block_count": len(expected),
        }

    def upper(label: str) -> float:
        return results[label]["paired_percentile_bootstrap_95_ci"][1]

    reduction = int(payloads["qh031"]["injection"]["net_model_parameter_reduction"])
    gates = {
        "shared_compression_noninferior_to_base": upper("sharedsvd1024_minus_base") <= margin,
        "qh031_compression_noninferior_to_base": upper("qh031_minus_base") <= margin and reduction > 0,
        "qh031_strictly_better_than_base": upper("qh031_minus_base") < 0.0,
        "qh031_causal_active_better_than_own_zero": upper("qh031_active_minus_own_zero") < 0.0,
        "qh031_quantum_better_than_equal_parameter_classical": upper("qh031_minus_cc031") < 0.0,
        "qh031_entanglement_ablation_causally_worse": upper("qh031_active_minus_no_entanglement") < 0.0,
        "net_model_parameter_reduction_positive": reduction > 0,
    }
    gates["quantum_specific_promotion"] = bool(
        gates["qh031_compression_noninferior_to_base"]
        and gates["qh031_causal_active_better_than_own_zero"]
        and gates["qh031_quantum_better_than_equal_parameter_classical"]
        and gates["qh031_entanglement_ablation_causally_worse"]
    )
    if gates["quantum_specific_promotion"]:
        verdict = "promote_qh031_quantum_specific_shared_basis_replacement"
    elif gates["qh031_compression_noninferior_to_base"]:
        verdict = "retain_qh031_compressed_replacement_reject_quantum_specific_claim"
    else:
        verdict = "reject_qh031_shared_basis_route_on_c4"

    output = {
        "status": "ok",
        "analysis": "QH-031 locked paired C4 evaluation",
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
        "own_zero_max_abs_check": zero_max_abs,
        "training_diagnostics": {
            name: {
                key: value
                for key, value in payloads[name]["training"].items()
                if key != "losses"
            }
            for name in ("qh031", "cc031")
        },
        "parameter_audit": {
            "base_model_parameters": payloads["qh031"]["injection"]["base_model_parameters"],
            "deployed_model_parameters": payloads["qh031"]["injection"]["deployed_model_parameters"],
            "net_model_parameter_reduction": reduction,
            "trainable_parameters_qh031": payloads["qh031"]["injection"]["trainable_parameter_count"],
            "trainable_parameters_cc031": payloads["cc031"]["injection"]["trainable_parameter_count"],
        },
        "claim_limits": [
            "The exact 10-qubit circuit is classically dequantizable; no simulator speed advantage is claimed.",
            "Quantum attribution is rejected unless all three locked causal CI gates pass.",
            "This is a 64-block C4 development evaluation, not a final test benchmark.",
            "A small quantum-vs-dequantized NLL difference can arise from numerical ordering amplified by later BF16 layers.",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

