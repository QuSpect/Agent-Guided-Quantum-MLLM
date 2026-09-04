#!/usr/bin/env python3
"""Analyze the locked train-only QH032 screen without promotion claims."""

from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/qh032_train_screen"
LOCK = ROOT / "artifacts/qh032-train-screen-lock.json"
FILES = {name: RUN / f"screen-{name}-s20260851-n256-e64.json" for name in ("base", "sharedsvd1024", "qh032", "cc032")}
OUTPUT = RUN / "qh032-train-screen-analysis.json"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""): digest.update(chunk)
    return digest.hexdigest()

def main():
    lock = json.loads(LOCK.read_text())
    expected = lock["eval_indices"]
    payloads, arrays = {}, {}
    for name, path in FILES.items():
        payload = json.loads(path.read_text())
        if payload["dataset_lock_sha256"] != sha256(LOCK): raise AssertionError("lock mismatch")
        rows = payload["evaluation"]["rows"]
        if [row["block_index"] for row in rows] != expected: raise AssertionError("eval order mismatch")
        payloads[name] = payload
        arrays[name] = np.asarray([row["mean_token_nll"] for row in rows])
    comparisons = {}
    rng = np.random.default_rng(lock["seed"])
    resamples = rng.integers(0, len(expected), size=(20000, len(expected)))
    for label, left, right in (
        ("shared_minus_base", "sharedsvd1024", "base"),
        ("qh_minus_shared_own_zero", "qh032", "sharedsvd1024"),
        ("qh_minus_cc", "qh032", "cc032"),
    ):
        diff = arrays[left] - arrays[right]
        boot = diff[resamples].mean(1)
        comparisons[label] = {"mean_delta_nll": float(diff.mean()),
            "descriptive_paired_bootstrap_95_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}
    gates = {
        "qh_point_better_than_own_zero": comparisons["qh_minus_shared_own_zero"]["mean_delta_nll"] < 0,
        "qh_point_better_than_equal_parameter_cc": comparisons["qh_minus_cc"]["mean_delta_nll"] < 0,
    }
    gates["continue_to_new_c4_lock"] = all(gates.values())
    output = {
        "status": "ok", "analysis": "QH032 train-split pre-holdout screen",
        "verdict": "continue_to_new_c4_lock" if gates["continue_to_new_c4_lock"] else "reject_or_mutate_before_c4",
        "point_nll": {name: float(array.mean()) for name, array in arrays.items()},
        "comparisons": comparisons, "screen_gates": gates,
        "claim_limit": "Train-only selection screen; intervals are descriptive and cannot support a quality or quantum claim.",
        "validation_rows_inspected": 0, "test_rows_inspected": 0,
        "training_diagnostics": {name: {k: v for k, v in payloads[name]["training"].items() if k != "losses"} for name in ("qh032", "cc032")},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))

if __name__ == "__main__": main()

