#!/usr/bin/env python3
"""Weight-only cross-layer shared-basis audit for Qwen3.8 full-attention V projections.

No language data are loaded.  For each consecutive pair of full-attention
layers, the script compares separate truncated SVDs with a common right basis.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM


PROJECT = Path(__file__).resolve().parents[2]
RECORDS = PROJECT / "records"
ARTIFACTS = PROJECT / "artifacts"
FULL_ATTENTION_LAYERS = list(range(3, 64, 4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", nargs="+", type=int, default=[256, 384, 512, 640, 768, 896, 1024])
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "qh031-shared-v-basis-feasibility.json")
    return parser.parse_args()


@torch.inference_mode()
def singular_energy(weight: torch.Tensor) -> torch.Tensor:
    # Non-zero singular values of W (1024 x 5120) are eigenvalues of W W^T.
    gram = weight @ weight.T
    values = torch.linalg.eigvalsh(gram).clamp_min_(0).flip(0)
    return values.double()


@torch.inference_mode()
def shared_singular_energy(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    stacked = torch.cat([first, second], dim=0)
    gram = stacked @ stacked.T
    values = torch.linalg.eigvalsh(gram).clamp_min_(0).flip(0)
    return values.double()


def cumulative_fraction(values: torch.Tensor, rank: int) -> float:
    return float(values[:rank].sum() / values.sum())


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("QH031 feasibility audit is GPU-only")
    model_dir = Path((RECORDS / "active_model_path.txt").read_text(encoding="utf-8").strip())
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    device = torch.device("cuda:0")
    weights = {}
    energies = {}
    for layer_index in FULL_ATTENTION_LAYERS:
        projection = model.model.language_model.layers[layer_index].self_attn.v_proj
        weight = projection.weight.detach().to(device=device, dtype=torch.float32)
        if tuple(weight.shape) != (1024, 5120):
            raise AssertionError((layer_index, tuple(weight.shape)))
        weights[layer_index] = weight
        energies[layer_index] = singular_energy(weight)

    output_size, input_size = 1024, 5120
    dense_pair_parameters = 2 * output_size * input_size
    rows = []
    for first_layer, second_layer in zip(FULL_ATTENTION_LAYERS[:-1], FULL_ATTENTION_LAYERS[1:]):
        shared_values = shared_singular_energy(weights[first_layer], weights[second_layer])
        pair = {
            "layers": [first_layer, second_layer],
            "fourier_distance_in_transformer_blocks": second_layer - first_layer,
            "ranks": [],
        }
        for rank in args.ranks:
            separate_parameters = 2 * rank * (input_size + output_size)
            shared_parameters = rank * (input_size + 2 * output_size)
            equal_budget_shared_rank = min(
                2 * output_size,
                math.floor(separate_parameters / (input_size + 2 * output_size)),
            )
            separate_retained = (
                energies[first_layer][:rank].sum() + energies[second_layer][:rank].sum()
            ) / (energies[first_layer].sum() + energies[second_layer].sum())
            pair["ranks"].append({
                "rank": rank,
                "separate_svd_parameter_count": separate_parameters,
                "shared_basis_parameter_count": shared_parameters,
                "dense_pair_parameter_count": dense_pair_parameters,
                "shared_basis_net_reduction": dense_pair_parameters - shared_parameters,
                "separate_svd_retained_energy_fraction": float(separate_retained),
                "shared_basis_retained_energy_fraction": cumulative_fraction(shared_values, rank),
                "equal_separate_parameter_budget_shared_rank": equal_budget_shared_rank,
                "equal_budget_shared_retained_energy_fraction": cumulative_fraction(
                    shared_values, equal_budget_shared_rank
                ),
            })
        rows.append(pair)

    for first_layer, second_layer in zip(FULL_ATTENTION_LAYERS[:-1], FULL_ATTENTION_LAYERS[1:]):
        del weights[first_layer]
    del model
    payload = {
        "status": "ok",
        "candidate_id": "QH-031-shared-v-basis-feasibility",
        "scope": "weight-only; no train, validation, or test data loaded",
        "source_motivation": "Basis Sharing, arXiv:2410.03765",
        "model_dir": str(model_dir),
        "target": "self_attn.v_proj",
        "full_attention_layers": FULL_ATTENTION_LAYERS,
        "weight_shape": [output_size, input_size],
        "pair_rows": rows,
        "decision_rule": (
            "Weight energy only selects candidates. Promotion requires activation-aware action fidelity, positive net "
            "parameter reduction including the persistent quantum circuit, exact CUDA simulator execution, and locked language controls."
        ),
        "proposed_quantum_role": (
            "Run an identity-initialized 10-qubit unitary in the shared rank space, preserving the activation norm; compare "
            "against an equal-parameter classical orthogonal butterfly and branch-off control."
        ),
        "test_rows_inspected_or_used_for_fitness": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
