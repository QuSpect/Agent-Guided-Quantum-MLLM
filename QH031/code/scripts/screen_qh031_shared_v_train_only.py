#!/usr/bin/env python3
"""Train-only language/action screen for a two-layer shared V-projection basis.

This screen deliberately reuses already-consumed QH026 training blocks.  It
does not read C4 validation or any test data and cannot establish generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForMultimodalLM


PROJECT = Path(__file__).resolve().parents[2]
RECORDS = PROJECT / "records"
TOKENS = PROJECT / "datasets/processed/wikitext2_qwen38/train-tokens.pt"
SOURCE_LOCK = PROJECT / "artifacts/qh026-butterfly-lock.json"
OUTPUT = PROJECT / "artifacts/qh031-shared-v-train-only-screen.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs=2, type=int, default=[55, 59])
    parser.add_argument("--ranks", nargs="+", type=int, default=[768, 896, 1024])
    parser.add_argument("--sequences", type=int, default=16)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SharedDown(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(hidden_states, self.weight)


class SharedBasisLinear(nn.Module):
    def __init__(self, shared_down: SharedDown, up_weight: torch.Tensor) -> None:
        super().__init__()
        self.shared_down = shared_down
        self.up = nn.Linear(
            up_weight.shape[1], up_weight.shape[0], bias=False,
            device=up_weight.device, dtype=up_weight.dtype,
        )
        with torch.no_grad():
            self.up.weight.copy_(up_weight)
        self.up.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(self.shared_down(hidden_states))


def token_block(token_ids: torch.Tensor, index: int, length: int, device: torch.device) -> torch.Tensor:
    start = index * length
    return token_ids[start : start + length].to(device=device, dtype=torch.long).unsqueeze(0)


@torch.inference_mode()
def evaluate(model: nn.Module, token_ids: torch.Tensor, indices: list[int], length: int) -> dict:
    rows = []
    model.eval()
    for index in indices:
        input_ids = token_block(token_ids, index, length, model.device)
        loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        rows.append({"block_index": index, "mean_token_nll": float(loss)})
    values = np.asarray([row["mean_token_nll"] for row in rows], dtype=np.float64)
    return {
        "mean_token_nll": float(values.mean()),
        "token_perplexity": float(math.exp(values.mean())),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("QH031 train-only screen is GPU-only")
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    indices = [int(value) for value in lock["train_indices"][: args.sequences]]
    if len(indices) != args.sequences:
        raise RuntimeError("not enough consumed train indices")
    length = int(lock["sequence_length"])
    token_ids = torch.load(TOKENS, map_location="cpu", weights_only=True)
    model_dir = Path((RECORDS / "active_model_path.txt").read_text(encoding="utf-8").strip())
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    base = evaluate(model, token_ids, indices, length)
    base_rows = {row["block_index"]: row["mean_token_nll"] for row in base["rows"]}

    first_index, second_index = args.layers
    first_layer = model.model.language_model.layers[first_index]
    second_layer = model.model.language_model.layers[second_index]
    if first_layer.block_type != "full_attention" or second_layer.block_type != "full_attention":
        raise ValueError("both layers must be full_attention")
    first_original = first_layer.self_attn.v_proj
    second_original = second_layer.self_attn.v_proj
    first_weight = first_original.weight.detach().float().to("cuda:0")
    second_weight = second_original.weight.detach().float().to("cuda:0")
    stacked = torch.cat([first_weight, second_weight], dim=0)
    u, s, vh = torch.linalg.svd(stacked, full_matrices=False)
    total_energy = s.square().sum()
    dense_pair_parameters = first_weight.numel() + second_weight.numel()
    candidates = []

    for rank in args.ranks:
        down_weight = vh[:rank]
        combined_up = u[:, :rank] * s[:rank]
        first_up = combined_up[: first_weight.shape[0]]
        second_up = combined_up[first_weight.shape[0] :]
        first_device = first_original.weight.device
        second_device = second_original.weight.device
        if first_device != second_device:
            raise RuntimeError(f"shared basis currently requires a common device, got {first_device}/{second_device}")
        shared_down = SharedDown(down_weight.to(device=first_device, dtype=first_original.weight.dtype))
        first_replacement = SharedBasisLinear(
            shared_down, first_up.to(device=first_device, dtype=first_original.weight.dtype)
        )
        second_replacement = SharedBasisLinear(
            shared_down, second_up.to(device=second_device, dtype=second_original.weight.dtype)
        )
        first_layer.self_attn.v_proj = first_replacement
        second_layer.self_attn.v_proj = second_replacement
        result = evaluate(model, token_ids, indices, length)
        result_rows = {row["block_index"]: row["mean_token_nll"] for row in result["rows"]}
        paired = np.asarray([result_rows[index] - base_rows[index] for index in indices])
        shared_parameters = rank * (first_weight.shape[1] + 2 * first_weight.shape[0])
        candidates.append({
            "candidate": f"shared-v-l{first_index}-l{second_index}-r{rank}",
            "rank": rank,
            "shared_basis_parameters": shared_parameters,
            "dense_pair_parameters": dense_pair_parameters,
            "net_model_parameter_reduction": dense_pair_parameters - shared_parameters,
            "stacked_weight_energy_fraction": float(s[:rank].square().sum() / total_energy),
            "train_only_evaluation": result,
            "paired_train_nll_delta_vs_base": {
                "estimate": float(paired.mean()),
                "min": float(paired.min()),
                "max": float(paired.max()),
            },
        })
        first_layer.self_attn.v_proj = first_original
        second_layer.self_attn.v_proj = second_original
        del shared_down, first_replacement, second_replacement
        torch.cuda.empty_cache()

    payload = {
        "status": "ok",
        "candidate_id": "QH-031-shared-v-train-only-screen",
        "scope": "reuses already-consumed QH026 WikiText training blocks; no validation or test data read",
        "model_dir": str(model_dir),
        "layers": args.layers,
        "sequence_length": length,
        "train_indices": indices,
        "source_lock": str(SOURCE_LOCK),
        "source_lock_sha256": sha256(SOURCE_LOCK),
        "train_tokens_sha256": sha256(TOKENS),
        "base_model_parameters": base_parameters,
        "base_train_only_evaluation": base,
        "candidates": candidates,
        "validation_rows_inspected_or_used_for_fitness": 0,
        "test_rows_inspected_or_used_for_fitness": 0,
        "decision_rule": (
            "This train-only result may reject a destructive factorization but cannot promote language quality. "
            "Any promoted architecture needs a fresh locked C4 partition and quantum/classical causal controls."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = args.output or OUTPUT.with_name(
        f"qh031-shared-v-l{first_index}-l{second_index}-train-only-screen.json"
    )
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "base_nll": base["mean_token_nll"],
        "candidates": [
            {
                "candidate": row["candidate"],
                "energy": row["stacked_weight_energy_fraction"],
                "net_reduction": row["net_model_parameter_reduction"],
                "train_nll": row["train_only_evaluation"]["mean_token_nll"],
                "delta": row["paired_train_nll_delta_vs_base"]["estimate"],
            }
            for row in candidates
        ],
        "record": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
