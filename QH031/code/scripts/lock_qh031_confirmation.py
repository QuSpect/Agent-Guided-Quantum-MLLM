#!/usr/bin/env python3
"""Pre-register a larger, disjoint C4 confirmation for frozen QH031 checkpoints."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT / "artifacts"
TOKENS = PROJECT / "datasets/processed/c4_validation_qwen38/validation-tokens.pt"
PARENT_LOCK = ARTIFACTS / "qh031-shared-basis-lock.json"
PARENT_ANALYSIS = ARTIFACTS / "c4_qh031/c4-qh031-paired-analysis.json"
QH_CHECKPOINT = ARTIFACTS / "c4_qh031/c4-qh031-s20260843-replacement.pt"
CC_CHECKPOINT = ARTIFACTS / "c4_qh031/c4-cc031-s20260843-replacement.pt"
OUTPUT = ARTIFACTS / "qh031-confirmation-lock.json"
SEED = 20260847
SEQUENCE_LENGTH = 256
EVAL_COUNT = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def used_c4_indices() -> tuple[set[int], int]:
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

        def walk(value: object, c4_context: bool = False) -> None:
            if isinstance(value, dict):
                text = json.dumps(value, ensure_ascii=False)[:4000].lower()
                local_c4 = c4_context or "allenai/c4" in text or "c4_validation" in text
                for key, child in value.items():
                    if local_c4 and key == "eval_indices" and isinstance(child, list):
                        used.update(int(item) for item in child)
                    elif local_c4 and key == "block_index" and isinstance(child, int):
                        used.add(int(child))
                    walk(child, local_c4)
            elif isinstance(value, list):
                for child in value:
                    walk(child, c4_context)

        walk(payload)
    return used, parsed


def main() -> None:
    required = (TOKENS, PARENT_LOCK, PARENT_ANALYSIS, QH_CHECKPOINT, CC_CHECKPOINT)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    parent_lock = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_ANALYSIS.read_text(encoding="utf-8"))
    expected_verdict = "retain_qh031_compressed_replacement_reject_quantum_specific_claim"
    if parent["verdict"] != expected_verdict:
        raise RuntimeError("confirmation requires the recorded QH031 development verdict")
    tokens = torch.load(TOKENS, map_location="cpu", weights_only=True)
    complete = len(tokens) // SEQUENCE_LENGTH
    used, parsed = used_c4_indices()
    available = np.asarray(sorted(set(range(complete)) - used), dtype=np.int64)
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(available)[:EVAL_COUNT].tolist()
    if len(indices) != EVAL_COUNT:
        raise RuntimeError(f"insufficient unused C4 blocks: {len(available)} available")
    if set(indices) & used:
        raise AssertionError("confirmation lock overlaps a prior C4 block")
    payload = {
        "status": "locked_before_qh031_confirmation_results",
        "candidate": "QH-031 frozen-checkpoint independent C4 confirmation",
        "purpose": (
            "Estimate the three previously inconclusive QH031 causal contrasts with "
            "a fixed larger sample; no retraining or hyperparameter selection is allowed."
        ),
        "parent": {
            "development_analysis": str(PARENT_ANALYSIS),
            "development_analysis_sha256": sha256(PARENT_ANALYSIS),
            "development_verdict": parent["verdict"],
            "development_eval_count": int(parent["comparisons"]["qh031_minus_cc031"]["block_count"]),
            "parent_lock": str(PARENT_LOCK),
            "parent_lock_sha256": sha256(PARENT_LOCK),
        },
        "frozen_inputs": {
            "qh031_checkpoint": str(QH_CHECKPOINT),
            "qh031_checkpoint_sha256": sha256(QH_CHECKPOINT),
            "cc031_checkpoint": str(CC_CHECKPOINT),
            "cc031_checkpoint_sha256": sha256(CC_CHECKPOINT),
            "architecture": parent_lock["architecture"],
            "training_steps_in_confirmation": 0,
            "hyperparameter_changes": 0,
        },
        "evaluation_dataset": {
            "name": "allenai/c4 en validation",
            "source_revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
            "selection": "unused token blocks from pinned first-2048-row validation shard subset",
            "tokens_sha256": sha256(TOKENS),
        },
        "sequence_length": SEQUENCE_LENGTH,
        "seed": SEED,
        "eval_indices": indices,
        "eval_count": EVAL_COUNT,
        "excluded_prior_c4_eval_indices_count": len(used),
        "available_unused_c4_blocks_before_selection": len(available),
        "prior_artifact_json_files_parsed": parsed,
        "c4_eval_overlap": 0,
        "evaluation_protocol": {
            "primary_metric": "mean token negative log likelihood",
            "paired_bootstrap_samples": 20000,
            "compression_noninferiority_margin_nll": 0.005,
            "fixed_arms": [
                "frozen_dense_base",
                "shared_svd1024_equals_qh031_own_zero",
                "frozen_checkpoint_qh031_active",
                "frozen_checkpoint_cc031_active",
                "same_qh031_checkpoint_no_entanglement",
            ],
            "quantum_specific_confirmation_requires": [
                "QH031 active 95% CI strictly better than shared-SVD own zero",
                "QH031 active 95% CI strictly better than equal-parameter CC031",
                "QH031 active 95% CI strictly better than its no-entanglement intervention",
            ],
            "multiple_testing_note": (
                "All three directional gates must pass, so no single favorable contrast is sufficient."
            ),
        },
        "claim_limit": (
            "This is a larger disjoint development confirmation, not a final external test set "
            "and not evidence of computational quantum advantage."
        ),
        "test_rows_inspected_or_used_for_fitness": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "eval_count": EVAL_COUNT,
        "excluded_prior_c4": len(used),
        "available_before_selection": len(available),
        "record": str(OUTPUT),
        "sha256": sha256(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()

