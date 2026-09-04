#!/usr/bin/env python3
"""Lock QH-025 end-to-end distillation before touching its holdout scores."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT / "artifacts"
TRAIN = PROJECT / "datasets/processed/wikitext2_qwen38/train-tokens.pt"
VALIDATION = PROJECT / "datasets/processed/c4_validation_qwen38/validation-tokens.pt"
C4_PREPARATION = ARTIFACTS / "c4-validation-preparation.json"
QH024_LOCK = ARTIFACTS / "wikitext-qh024-distillation-lock.json"
QH024_ANALYSIS = ARTIFACTS / "wikitext_qh024/wikitext-qh024-paired-analysis.json"
QH024_CHECKPOINT = ARTIFACTS / "wikitext_qh024/wikitext-qh024-s20260838-replacement.pt"
CC024_CHECKPOINT = ARTIFACTS / "wikitext_qh024/wikitext-cc024-s20260838-replacement.pt"
OUTPUT = ARTIFACTS / "wikitext-qh025-end-to-end-lock.json"
SEED = 20260839
SEQUENCE_LENGTH = 256
TRAIN_COUNT = 1024
EVAL_COUNT = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_prior_indices() -> tuple[set[int], set[int], int]:
    """Conservatively collect all prior train and evaluation block indices."""
    prior_train: set[int] = set()
    prior_eval: set[int] = set()
    parsed = 0
    for path in ARTIFACTS.rglob("*.json"):
        if path == OUTPUT:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        parsed += 1

        def walk(value: object, parent_key: str = "") -> None:
            if isinstance(value, dict):
                if "block_index" in value and isinstance(value["block_index"], int):
                    prior_eval.add(int(value["block_index"]))
                for key, child in value.items():
                    if key == "train_indices" and isinstance(child, list):
                        prior_train.update(int(item) for item in child)
                    elif key == "eval_indices" and isinstance(child, list):
                        prior_eval.update(int(item) for item in child)
                    elif key == "indices" and isinstance(child, list) and "lock" in path.name:
                        prior_eval.update(int(item) for item in child)
                    walk(child, key)
            elif isinstance(value, list):
                for child in value:
                    walk(child, parent_key)

        walk(payload)
    return prior_train, prior_eval, parsed


def main() -> None:
    for required in [
        QH024_LOCK, QH024_ANALYSIS, QH024_CHECKPOINT, CC024_CHECKPOINT,
        C4_PREPARATION, TRAIN, VALIDATION,
    ]:
        if not required.exists():
            raise FileNotFoundError(required)
    analysis = json.loads(QH024_ANALYSIS.read_text(encoding="utf-8"))
    if analysis["verdict"] != "retain_svd_compression_route_reject_qh024_quantum_residual":
        raise RuntimeError("QH-025 mutation is not justified by the locked QH-024 verdict")

    train_tokens = torch.load(TRAIN, map_location="cpu", weights_only=True)
    validation_tokens = torch.load(VALIDATION, map_location="cpu", weights_only=True)
    complete_train = len(train_tokens) // SEQUENCE_LENGTH
    complete_validation = len(validation_tokens) // SEQUENCE_LENGTH
    prior_train, prior_eval, parsed = collect_prior_indices()
    train_available = np.asarray(sorted(set(range(complete_train)) - prior_train), dtype=np.int64)
    # C4 validation has never been used by this project. Historical integer block
    # indices belong to other datasets and must not be conflated with C4 indices.
    eval_available = np.asarray(range(complete_validation), dtype=np.int64)
    rng = np.random.default_rng(SEED)
    train_indices = rng.permutation(train_available)[:TRAIN_COUNT].tolist()
    eval_indices = rng.permutation(eval_available)[:EVAL_COUNT].tolist()
    if len(train_indices) != TRAIN_COUNT or len(eval_indices) != EVAL_COUNT:
        raise RuntimeError(
            "insufficient untouched blocks for QH-025: "
            f"train_available={len(train_available)}, eval_available={len(eval_available)}, "
            f"complete_train={complete_train}, complete_validation={complete_validation}, "
            f"prior_train={len(prior_train)}, prior_eval={len(prior_eval)}"
        )
    if set(train_indices) & prior_train:
        raise RuntimeError("QH-025 lock overlaps prior blocks")

    payload = {
        "status": "locked_before_qh025_qh_cc_base_performance_results",
        "candidate": "QH-025 end-to-end hidden-state-distilled true v_proj replacement",
        "mutation_from_qh024": {
            "changed": "move supervision from local deleted-projection MSE to final normalized hidden-state MSE",
            "unchanged": [
                "layer 7 v_proj replacement",
                "rank-512 frozen SVD backbone",
                "10 qubits, depth 2, exact CUDA statevector",
                "112701 trainable parameters and equal-parameter classical control",
            ],
            "reason": (
                "QH-024 reduced local projection loss but worsened language NLL; hybrid-model priming "
                "literature recommends end-to-end pre-LM hidden-state distillation."
            ),
            "qh024_analysis": str(QH024_ANALYSIS),
            "qh024_analysis_sha256": sha256(QH024_ANALYSIS),
        },
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
        "initialization": {
            "qh025_from": str(QH024_CHECKPOINT),
            "qh025_from_sha256": sha256(QH024_CHECKPOINT),
            "cc025_from": str(CC024_CHECKPOINT),
            "cc025_from_sha256": sha256(CC024_CHECKPOINT),
        },
        "training_dataset": {
            "name": "Salesforce/wikitext wikitext-2-raw-v1 train",
            "source_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "license": "CC BY-SA 4.0",
        },
        "evaluation_dataset": {
            "name": "allenai/c4 en validation",
            "source_revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
            "license": "ODC-BY 1.0; underlying Common Crawl terms also apply",
            "selection": "first 2048 rows of pinned shard 0, then seeded token-block sample",
            "preparation_record": str(C4_PREPARATION),
            "preparation_sha256": sha256(C4_PREPARATION),
        },
        "sequence_length": SEQUENCE_LENGTH,
        "seed": SEED,
        "train_indices": train_indices,
        "train_count": TRAIN_COUNT,
        "eval_indices": eval_indices,
        "eval_count": EVAL_COUNT,
        "prior_artifact_json_files_parsed": parsed,
        "excluded_prior_train_indices_count": len(prior_train),
        "historical_non_c4_eval_indices_seen_but_not_applied_to_c4": len(prior_eval),
        "train_overlap_with_prior": len(set(train_indices) & prior_train),
        "eval_overlap_with_prior_c4_fitness": 0,
        "train_tokens_sha256": sha256(TRAIN),
        "validation_tokens_sha256": sha256(VALIDATION),
        "training_protocol": {
            "objective": "relative MSE of final normalized hidden states after all decoder layers",
            "teacher_path": "original dense v_proj through the complete frozen model",
            "student_path": "QH-025 or equal-parameter CC-025 through the complete frozen model",
            "optimizer": "AdamW",
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "steps": TRAIN_COUNT,
            "all_original_model_parameters_frozen": True,
            "initialization_uses_qh024_local_distillation_checkpoint": True,
        },
        "evaluation_protocol": {
            "primary_metric": "mean token NLL",
            "compression_noninferiority_margin_nll": 0.005,
            "paired_bootstrap_samples": 20000,
            "required_controls": [
                "frozen_dense_base",
                "svd512_zero_branch",
                "QH-025 own zero branch",
                "equal-parameter CC-025",
            ],
        },
        "literature_basis": [
            "https://github.com/awslabs/hybrid-model-factory/blob/main/docs/Priming.md",
            "https://arxiv.org/abs/2601.11667",
        ],
        "test_rows_inspected_or_used_for_fitness": 0,
        "c4_train_rows_inspected_or_used": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "train_count": TRAIN_COUNT,
        "eval_count": EVAL_COUNT,
        "train_overlap": payload["train_overlap_with_prior"],
        "eval_overlap": payload["eval_overlap_with_prior_c4_fitness"],
        "historical_non_c4_eval_indices": len(prior_eval),
        "record": str(OUTPUT),
        "sha256": sha256(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
