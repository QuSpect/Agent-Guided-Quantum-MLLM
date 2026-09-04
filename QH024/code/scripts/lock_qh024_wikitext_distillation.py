#!/usr/bin/env python3
"""Lock QH-024 train/eval blocks and hyperparameters before formal runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT / "artifacts"
STAGE_A = ARTIFACTS / "wikitext_stage_a"
TRAIN = PROJECT / "datasets/processed/wikitext2_qwen38/train-tokens.pt"
VALIDATION = PROJECT / "datasets/processed/wikitext2_qwen38/validation-tokens.pt"
PARENT_LOCK = ARTIFACTS / "wikitext-replacement-screen-lock.json"
SCREEN = ARTIFACTS / "wikitext-svd-value-replacement-screen.json"
OUTPUT = ARTIFACTS / "wikitext-qh024-distillation-lock.json"
SEED = 20260838
SEQUENCE_LENGTH = 256
TRAIN_COUNT = 512
EVAL_COUNT = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prior_train_indices() -> set[int]:
    paths = [
        STAGE_A / "wikitext-qh010-s20260829-n256-l256.json",
        STAGE_A / "wikitext-cc010-s20260829-n256-l256.json",
        STAGE_A / "wikitext-qh022-unused-complement-s20260829-n256-l256.json",
        STAGE_A / "wikitext-cc022-unused-complement-s20260829-n256-l256.json",
    ]
    values: set[int] = set()
    for path in paths:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            values.update(int(index) for index in payload["dataset"]["train_indices"])
    return values


def main() -> None:
    parent = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    if parent["status"] != "locked_before_any_replacement_performance_result":
        raise RuntimeError("invalid parent replacement lock")
    if not SCREEN.exists():
        raise RuntimeError("SVD screen record is required before architecture locking")
    eval_indices = [int(value) for value in parent["indices"][32:64]]
    if len(eval_indices) != EVAL_COUNT:
        raise RuntimeError("parent lock does not contain the reserved holdout")
    screen_payload = json.loads(SCREEN.read_text(encoding="utf-8"))
    screened_indices = {
        int(row["block_index"]) for row in screen_payload["base"]["rows"]
    }
    if screened_indices & set(eval_indices):
        raise RuntimeError("QH-024 holdout overlaps the SVD selection blocks")

    train_tokens = torch.load(TRAIN, map_location="cpu", weights_only=True)
    complete_train = len(train_tokens) // SEQUENCE_LENGTH
    excluded_train = prior_train_indices()
    available = np.array(sorted(set(range(complete_train)) - excluded_train), dtype=np.int64)
    train_indices = np.random.default_rng(SEED).permutation(available)[:TRAIN_COUNT].tolist()
    if len(train_indices) != TRAIN_COUNT or set(train_indices) & excluded_train:
        raise RuntimeError("failed to create disjoint QH-024 training blocks")

    payload = {
        "status": "locked_before_qh024_qh_cc_base_performance_results",
        "candidate": "QH-024 true v_proj replacement",
        "architecture": {
            "layer_index": 7,
            "target": "v_proj",
            "svd_rank": 512,
            "qubits": 10,
            "depth": 2,
            "quantum_trainable_parameters": 112701,
            "original_projection_parameters": 5242880,
            "deployed_replacement_parameters": 3258429,
            "net_model_parameter_reduction": 1984451,
        },
        "selection_basis": {
            "screen_record": str(SCREEN),
            "screen_sha256": sha256(SCREEN),
            "selected_candidate": "svd-v-l7-r512",
            "selection_rule": (
                "largest tested parameter reduction among candidates with development-screen "
                "mean NLL increase <= 0.005"
            ),
        },
        "dataset": "Salesforce/wikitext wikitext-2-raw-v1",
        "source_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "license": "CC BY-SA 4.0",
        "sequence_length": SEQUENCE_LENGTH,
        "seed": SEED,
        "train_indices": train_indices,
        "train_count": TRAIN_COUNT,
        "excluded_prior_train_indices_count": len(excluded_train),
        "train_overlap_with_excluded": len(set(train_indices) & excluded_train),
        "eval_indices": eval_indices,
        "eval_count": EVAL_COUNT,
        "eval_overlap_with_svd_selection_blocks": len(screened_indices & set(eval_indices)),
        "parent_lock": str(PARENT_LOCK),
        "parent_lock_sha256": sha256(PARENT_LOCK),
        "train_tokens_sha256": sha256(TRAIN),
        "validation_tokens_sha256": sha256(VALIDATION),
        "training_protocol": {
            "objective": "relative MSE distillation of the deleted dense v_proj output",
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "epochs": 1,
            "all_original_model_parameters_frozen": True,
            "teacher_dense_weight_is_training_only_nonpersistent_buffer": True,
        },
        "evaluation_protocol": {
            "primary_metric": "mean token NLL",
            "compression_noninferiority_margin_nll": 0.005,
            "paired_bootstrap_samples": 20000,
            "required_controls": ["frozen_dense_base", "svd512_zero_branch", "CC-024"],
        },
        "test_rows_inspected_or_used_for_fitness": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "train_count": TRAIN_COUNT,
        "eval_count": EVAL_COUNT,
        "train_overlap": payload["train_overlap_with_excluded"],
        "eval_overlap": payload["eval_overlap_with_svd_selection_blocks"],
        "record": str(OUTPUT),
        "sha256": sha256(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
