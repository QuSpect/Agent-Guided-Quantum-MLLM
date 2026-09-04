#!/usr/bin/env python3
"""Lock disjoint WikiText-train blocks for QH032 pre-holdout screening."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT / "artifacts"
TOKENS = PROJECT / "datasets/processed/wikitext2_qwen38/train-tokens.pt"
STATIC = ARTIFACTS / "qh032-static-validation.json"
SMOKE = ARTIFACTS / "qh032-full-model-smoke.json"
CONFIRMATION = ARTIFACTS / "c4_qh031_confirmation/qh031-confirmation-paired-analysis.json"
OUTPUT = ARTIFACTS / "qh032-train-screen-lock.json"
SEED = 20260851
LENGTH = 256
TRAIN_COUNT = 256
EVAL_COUNT = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prior_train_indices() -> tuple[set[int], int]:
    used: set[int] = set()
    parsed = 0
    for path in ARTIFACTS.rglob("*.json"):
        if path == OUTPUT:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        parsed += 1
        def walk(value: object) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in {"train_indices", "train_block_indices"} and isinstance(child, list):
                        used.update(int(item) for item in child)
                    elif key == "train_block_index" and isinstance(child, int):
                        used.add(int(child))
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(payload)
    return used, parsed


def main() -> None:
    for path in (TOKENS, STATIC, SMOKE, CONFIRMATION):
        if not path.exists():
            raise FileNotFoundError(path)
    if json.loads(STATIC.read_text())["status"] != "pass" or json.loads(SMOKE.read_text())["status"] != "pass":
        raise RuntimeError("QH032 engineering gates did not pass")
    confirmation = json.loads(CONFIRMATION.read_text())
    if confirmation["verdict"] != "retain_qh031_compression_reject_quantum_specific_confirmation":
        raise RuntimeError("QH032 requires the recorded QH031 core rejection")
    tokens = torch.load(TOKENS, map_location="cpu", weights_only=True)
    complete = len(tokens) // LENGTH
    used, parsed = prior_train_indices()
    available = np.asarray(sorted(set(range(complete)) - used), dtype=np.int64)
    selected = np.random.default_rng(SEED).permutation(available)[: TRAIN_COUNT + EVAL_COUNT].tolist()
    if len(selected) != TRAIN_COUNT + EVAL_COUNT:
        raise RuntimeError("insufficient unused WikiText train blocks")
    train_indices, eval_indices = selected[:TRAIN_COUNT], selected[TRAIN_COUNT:]
    if set(train_indices) & set(eval_indices) or set(selected) & used:
        raise AssertionError("QH032 train screen overlap")
    payload = {
        "status": "locked_before_qh032_train_only_screen_results",
        "candidate": "QH-032 Pauli-observable nonlinear replacement",
        "purpose": "Pre-holdout train-split screen; cannot support a quality or quantum-advantage claim.",
        "parent_confirmation": {"path": str(CONFIRMATION), "sha256": sha256(CONFIRMATION), "verdict": confirmation["verdict"]},
        "engineering_gates": {
            "static": {"path": str(STATIC), "sha256": sha256(STATIC)},
            "full_model_smoke": {"path": str(SMOKE), "sha256": sha256(SMOKE)},
        },
        "architecture": {
            "layers": [55, 59], "rank": 1024, "qubits": 10,
            "circuit_parameters": 20, "pauli_observables": 60,
            "layer_gains": 2, "total_trainable": 82,
            "net_model_parameter_reduction": 3145646,
        },
        "dataset": "Salesforce/wikitext wikitext-2-raw-v1 train only",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "tokens_sha256": sha256(TOKENS),
        "sequence_length": LENGTH,
        "seed": SEED,
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "train_count": TRAIN_COUNT,
        "eval_count": EVAL_COUNT,
        "excluded_prior_train_indices_count": len(used),
        "available_before_selection": len(available),
        "prior_artifact_json_files_parsed": parsed,
        "overlap": 0,
        "training_protocol": {
            "objective": "relative MSE of final normalized hidden states",
            "optimizer": "AdamW", "learning_rate": 0.003, "weight_decay": 0.0,
            "gradient_clip_norm": 1.0, "steps": TRAIN_COUNT,
            "all_original_and_shared_basis_parameters_frozen": True,
        },
        "screen_rule": {
            "continue_to_new_C4_lock_only_if": [
                "QH032 point NLL is lower than shared-SVD own-zero",
                "QH032 point NLL is lower than equal-parameter CC032",
            ],
            "no_CI_or_quantum_claim_from_train_screen": True,
        },
        "validation_rows_inspected": 0,
        "test_rows_inspected": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "train_count": TRAIN_COUNT, "eval_count": EVAL_COUNT,
        "excluded_prior_train": len(used), "available_before_selection": len(available),
        "record": str(OUTPUT), "sha256": sha256(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()

