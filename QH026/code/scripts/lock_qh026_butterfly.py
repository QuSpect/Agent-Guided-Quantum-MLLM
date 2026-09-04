#!/usr/bin/env python3
"""Lock QH-026 train/eval blocks before any QH/CC performance result."""

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
QH025_LOCK = ARTIFACTS / "wikitext-qh025-end-to-end-lock.json"
QH025_ANALYSIS = ARTIFACTS / "c4_qh025/c4-qh025-paired-analysis.json"
QH025_CHECKPOINT = ARTIFACTS / "c4_qh025/c4-qh025-s20260839-replacement.pt"
CC025_CHECKPOINT = ARTIFACTS / "c4_qh025/c4-cc025-s20260839-replacement.pt"
STATIC = ARTIFACTS / "qh026-butterfly-gpu-validation.json"
OUTPUT = ARTIFACTS / "qh026-butterfly-lock.json"
SEED = 20260841
SEQUENCE_LENGTH = 256
TRAIN_COUNT = 1024
EVAL_COUNT = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prior_indices() -> tuple[set[int], set[int], int]:
    train_indices: set[int] = set()
    c4_eval_indices: set[int] = set()
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
                    if key == "train_indices" and isinstance(child, list):
                        train_indices.update(int(item) for item in child)
                    elif key == "eval_indices" and isinstance(child, list) and local_c4:
                        c4_eval_indices.update(int(item) for item in child)
                    elif key == "block_index" and isinstance(child, int) and local_c4:
                        c4_eval_indices.add(int(child))
                    walk(child, local_c4)
            elif isinstance(value, list):
                for child in value:
                    walk(child, c4_context)

        walk(payload)
    return train_indices, c4_eval_indices, parsed


def main() -> None:
    for required in (TRAIN, VALIDATION, QH025_LOCK, QH025_ANALYSIS, QH025_CHECKPOINT, CC025_CHECKPOINT, STATIC):
        if not required.exists():
            raise FileNotFoundError(required)
    analysis = json.loads(QH025_ANALYSIS.read_text(encoding="utf-8"))
    if analysis["verdict"] != "retain_compressed_replacement_reject_quantum_specific_claim":
        raise RuntimeError("QH026 is not justified by the QH025 verdict")
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    if static["status"] != "ok" or not static["cpu_quantum_path_rejected"]:
        raise RuntimeError("QH026 static gate is incomplete")

    train_tokens = torch.load(TRAIN, map_location="cpu", weights_only=True)
    eval_tokens = torch.load(VALIDATION, map_location="cpu", weights_only=True)
    complete_train = len(train_tokens) // SEQUENCE_LENGTH
    complete_eval = len(eval_tokens) // SEQUENCE_LENGTH
    used_train, used_c4_eval, parsed = prior_indices()
    available_train = np.asarray(sorted(set(range(complete_train)) - used_train), dtype=np.int64)
    available_eval = np.asarray(sorted(set(range(complete_eval)) - used_c4_eval), dtype=np.int64)
    rng = np.random.default_rng(SEED)
    train_indices = rng.permutation(available_train)[:TRAIN_COUNT].tolist()
    eval_indices = rng.permutation(available_eval)[:EVAL_COUNT].tolist()
    if len(train_indices) != TRAIN_COUNT or len(eval_indices) != EVAL_COUNT:
        raise RuntimeError("insufficient unused train or C4 blocks")
    if set(train_indices) & used_train or set(eval_indices) & used_c4_eval:
        raise AssertionError("QH026 data overlap")

    payload = {
        "status": "locked_before_qh026_qh_cc_base_performance_results",
        "candidate": "QH-026 depth3 normalized data-reupload butterfly true v_proj replacement",
        "parent": {
            "candidate": "QH-025",
            "analysis": str(QH025_ANALYSIS),
            "analysis_sha256": sha256(QH025_ANALYSIS),
            "failure": "point improvements did not pass active-zero or equal-parameter-classical CI gates",
        },
        "mutation": {
            "changed": [
                "circuit depth 2 to 3",
                "data encoding once to re-upload before every variational block",
                "ring entanglement to log-distance offsets 1,2,4 butterfly mixing",
                "parameter-free RMS normalization on encoded features and readout",
            ],
            "unchanged": [
                "layer 7 complete v_proj deletion",
                "rank-512 frozen SVD backbone",
                "10-qubit exact CUDA statevector inference",
                "final normalized hidden-state relative-MSE priming",
                "equal-total-parameter classical control",
            ],
            "literature_basis": [
                "https://arxiv.org/abs/2604.23931",
                "https://arxiv.org/abs/2606.11673",
            ],
        },
        "architecture": {
            "layer_index": 7,
            "target": "v_proj",
            "svd_rank": 512,
            "qubits": 10,
            "depth": 3,
            "quantum_trainable_parameters": 112731,
            "classical_trainable_parameters": 112731,
            "original_projection_parameters": 5242880,
            "deployed_replacement_parameters": 3258459,
            "net_model_parameter_reduction": 1984421,
        },
        "initialization": {
            "shared_qh025_parameters_from": str(QH025_CHECKPOINT),
            "qh025_sha256": sha256(QH025_CHECKPOINT),
            "shared_cc025_parameters_from": str(CC025_CHECKPOINT),
            "cc025_sha256": sha256(CC025_CHECKPOINT),
            "rule": "copy gamma/down/up and first two core layers; initialize only the new third core layer from seed",
        },
        "static_gate": {"path": str(STATIC), "sha256": sha256(STATIC)},
        "training_dataset": {
            "name": "Salesforce/wikitext wikitext-2-raw-v1 train",
            "source_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        },
        "evaluation_dataset": {
            "name": "allenai/c4 en validation",
            "source_revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
            "selection": "unused token blocks from pinned first-2048-row validation shard subset",
        },
        "sequence_length": SEQUENCE_LENGTH,
        "seed": SEED,
        "train_indices": train_indices,
        "train_count": TRAIN_COUNT,
        "eval_indices": eval_indices,
        "eval_count": EVAL_COUNT,
        "excluded_prior_train_indices_count": len(used_train),
        "excluded_prior_c4_eval_indices_count": len(used_c4_eval),
        "prior_artifact_json_files_parsed": parsed,
        "train_overlap": 0,
        "c4_eval_overlap": 0,
        "train_tokens_sha256": sha256(TRAIN),
        "validation_tokens_sha256": sha256(VALIDATION),
        "training_protocol": {
            "objective": "relative MSE of final normalized hidden states after all decoder layers",
            "optimizer": "AdamW",
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
            "steps": TRAIN_COUNT,
            "gradient_clip_norm": 1.0,
            "all_original_model_parameters_frozen": True,
        },
        "evaluation_protocol": {
            "primary_metric": "mean token NLL",
            "compression_noninferiority_margin_nll": 0.005,
            "paired_bootstrap_samples": 20000,
            "required_controls": ["frozen_dense_base", "svd512", "QH026-own-zero", "CC026"],
        },
        "test_rows_inspected_or_used_for_fitness": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "train_count": TRAIN_COUNT,
        "eval_count": EVAL_COUNT,
        "excluded_train": len(used_train),
        "excluded_c4_eval": len(used_c4_eval),
        "record": str(OUTPUT),
        "sha256": sha256(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
