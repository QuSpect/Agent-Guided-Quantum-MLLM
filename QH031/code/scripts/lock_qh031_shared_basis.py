#!/usr/bin/env python3
"""Lock unused train/C4 blocks before QH031 QH/CC/base performance runs."""

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
PARENT_ANALYSIS = ARTIFACTS / "c4_qh026/c4-qh026-paired-analysis.json"
FEASIBILITY = ARTIFACTS / "qh031-shared-v-basis-feasibility.json"
TRAIN_SCREEN = ARTIFACTS / "qh031-shared-v-l55-l59-train-only-screen.json"
STATIC = ARTIFACTS / "qh031-static-validation.json"
SMOKE = ARTIFACTS / "qh031-full-model-smoke.json"
OUTPUT = ARTIFACTS / "qh031-shared-basis-lock.json"
SEED = 20260843
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
                    if key in {"train_indices"} and isinstance(child, list):
                        train_indices.update(int(item) for item in child)
                    elif key == "train_block_index" and isinstance(child, int):
                        train_indices.add(int(child))
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
    required = (
        TRAIN,
        VALIDATION,
        PARENT_ANALYSIS,
        FEASIBILITY,
        TRAIN_SCREEN,
        STATIC,
        SMOKE,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    parent = json.loads(PARENT_ANALYSIS.read_text(encoding="utf-8"))
    if parent["verdict"] != "retain_compressed_replacement_reject_quantum_specific_claim":
        raise RuntimeError("QH031 requires the recorded QH026 compression-only verdict")
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    if static["status"] != "pass" or smoke["status"] != "pass":
        raise RuntimeError("QH031 static/full-model gates are incomplete")
    train_tokens = torch.load(TRAIN, map_location="cpu", weights_only=True)
    eval_tokens = torch.load(VALIDATION, map_location="cpu", weights_only=True)
    complete_train = len(train_tokens) // SEQUENCE_LENGTH
    complete_eval = len(eval_tokens) // SEQUENCE_LENGTH
    used_train, used_c4_eval, parsed = prior_indices()
    available_train = np.asarray(
        sorted(set(range(complete_train)) - used_train), dtype=np.int64
    )
    available_eval = np.asarray(
        sorted(set(range(complete_eval)) - used_c4_eval), dtype=np.int64
    )
    rng = np.random.default_rng(SEED)
    train_indices = rng.permutation(available_train)[:TRAIN_COUNT].tolist()
    eval_indices = rng.permutation(available_eval)[:EVAL_COUNT].tolist()
    if len(train_indices) != TRAIN_COUNT or len(eval_indices) != EVAL_COUNT:
        raise RuntimeError("insufficient unused WikiText train or C4 evaluation blocks")
    if set(train_indices) & used_train or set(eval_indices) & used_c4_eval:
        raise AssertionError("QH031 lock overlaps prior data")

    payload = {
        "status": "locked_before_qh031_qh_cc_base_performance_results",
        "candidate": "QH-031 two-layer shared-basis exact quantum replacement",
        "parent": {
            "candidate": "QH-026",
            "analysis": str(PARENT_ANALYSIS),
            "analysis_sha256": sha256(PARENT_ANALYSIS),
            "verdict": parent["verdict"],
            "lesson": (
                "single-layer quantum residual preserved quality but could not pass "
                "quantum-attribution confidence gates"
            ),
        },
        "mutation": {
            "changed": [
                "replace two full-attention value projections at layers 55 and 59",
                "joint stacked-SVD rank-1024 input basis shared across both layers",
                "one shared 10-qubit core reused at both layers",
                "identity-initialized RY plus controlled-RY butterfly circuit",
                "only 82 trainable parameters: 80 circuit angles and two layer gains",
            ],
            "unchanged": [
                "all original surviving model parameters frozen",
                "exact differentiable CUDA statevector simulation for train and inference",
                "final normalized hidden-state relative-MSE priming",
                "equal-total-parameter structured classical control",
                "own-zero shared-SVD causal control",
            ],
            "literature_basis": [
                "https://arxiv.org/abs/2410.03765",
                "https://arxiv.org/abs/2504.05336",
                "https://arxiv.org/abs/2603.26494",
                "https://arxiv.org/abs/2604.23931",
            ],
        },
        "architecture": {
            "layer_indices": [55, 59],
            "target": "self_attn.v_proj",
            "shared_svd_rank": 1024,
            "qubits": 10,
            "depth": 2,
            "entangling_offsets": [1, 2, 4],
            "original_parameters_removed": 10485760,
            "frozen_replacement_parameters": 7340032,
            "quantum_trainable_parameters": 82,
            "classical_trainable_parameters": 82,
            "deployed_replacement_parameters": 7340114,
            "net_model_parameter_reduction": 3145646,
            "initialization": "joint SVD; zero circuit angles; gains equal one",
        },
        "prelock_evidence": {
            "weight_feasibility": {"path": str(FEASIBILITY), "sha256": sha256(FEASIBILITY)},
            "consumed_train_only_screen": {"path": str(TRAIN_SCREEN), "sha256": sha256(TRAIN_SCREEN)},
            "static_gate": {"path": str(STATIC), "sha256": sha256(STATIC)},
            "full_model_smoke": {"path": str(SMOKE), "sha256": sha256(SMOKE)},
            "performance_holdout_was_not_read_by_these_gates": True,
        },
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
            "teacher": "same frozen model with the two original dense v_proj modules restored",
            "optimizer": "AdamW",
            "learning_rate": 0.003,
            "weight_decay": 0.0,
            "steps": TRAIN_COUNT,
            "gradient_clip_norm": 1.0,
            "all_original_and_shared_svd_parameters_frozen": True,
        },
        "evaluation_protocol": {
            "primary_metric": "mean token NLL",
            "compression_noninferiority_margin_nll": 0.005,
            "paired_bootstrap_samples": 20000,
            "required_controls": [
                "frozen_dense_base",
                "shared_svd1024",
                "QH031-own-zero",
                "CC031-equal-parameter",
                "QH031-no-entanglement-ablation",
            ],
            "quantum_attribution_requires": [
                "QH031 active 95% CI strictly better than own zero",
                "QH031 active 95% CI strictly better than equal-parameter CC031",
                "entangling-gate ablation causally worsens the trained QH031",
            ],
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

