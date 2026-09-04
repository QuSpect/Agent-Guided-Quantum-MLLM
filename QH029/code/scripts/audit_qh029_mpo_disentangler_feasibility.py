#!/usr/bin/env python3
"""Weight-only feasibility audit for a VQC-MPO-VQC Qwen projection replacement.

No language data are loaded.  The script TT-SVD factorizes the selected dense
projection as an MPO, reports truncation error and parameter count, and keeps
the quantum circuit storage ledger separate from the residual MPO ledger.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--target", default="v_proj")
    parser.add_argument(
        "--factor-plan",
        choices=["balanced6", "head_aligned8", "qubit11"],
        default="balanced6",
    )
    parser.add_argument("--bond-dims", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "qh029-mpo-feasibility.json")
    return parser.parse_args()


def factor_plan(output_size: int, input_size: int, plan: str) -> tuple[list[int], list[int]]:
    if (output_size, input_size) != (1024, 5120):
        raise ValueError(f"the audited Qwen v_proj shape must be (1024, 5120), got {(output_size, input_size)}")
    if plan == "balanced6":
        return [1, 4, 4, 4, 4, 4], [5, 4, 4, 4, 4, 4]
    if plan == "head_aligned8":
        # Qwen has eight KV heads and forty input-sized head groups.
        return [8, 2, 2, 2, 2, 2, 2, 2], [40, 2, 2, 2, 2, 2, 2, 2]
    if plan == "qubit11":
        return [1] + [2] * 10, [5] + [2] * 10
    raise AssertionError(plan)


@torch.inference_mode()
def tt_svd_audit(weight: torch.Tensor, output_dims: list[int], input_dims: list[int], cap: int) -> dict:
    site_count = len(output_dims)
    tensor = weight.reshape(*output_dims, *input_dims)
    permutation = []
    for site in range(site_count):
        permutation.extend([site, site_count + site])
    physical_dims = [a * b for a, b in zip(output_dims, input_dims)]
    remainder = tensor.permute(*permutation).contiguous().reshape(*physical_dims)
    original_norm_sq = weight.float().square().sum().double()
    discarded_norm_sq = torch.zeros((), dtype=torch.float64, device=weight.device)
    ranks = [1]
    core_parameter_count = 0
    previous_rank = 1
    singular_spectra = []

    for site, physical_dim in enumerate(physical_dims[:-1]):
        matrix = remainder.reshape(previous_rank * physical_dim, -1).float()
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        kept_rank = min(cap, s.numel())
        if kept_rank < s.numel():
            discarded_norm_sq += s[kept_rank:].double().square().sum()
        core_parameter_count += previous_rank * physical_dim * kept_rank
        ranks.append(kept_rank)
        singular_spectra.append({
            "site": site,
            "unfolding_shape": list(matrix.shape),
            "available_rank": int(s.numel()),
            "kept_rank": int(kept_rank),
            "leading_singular_value": float(s[0]),
            "discarded_squared_norm": float(s[kept_rank:].double().square().sum()) if kept_rank < s.numel() else 0.0,
        })
        remainder = s[:kept_rank, None] * vh[:kept_rank]
        previous_rank = kept_rank
        del matrix, u, s, vh

    core_parameter_count += previous_rank * physical_dims[-1]
    ranks.append(1)
    relative_frobenius_error = float(torch.sqrt(discarded_norm_sq / original_norm_sq))
    return {
        "bond_cap": cap,
        "realized_tt_ranks": ranks,
        "mpo_parameter_count": int(core_parameter_count),
        "compression_ratio_dense_over_mpo": weight.numel() / core_parameter_count,
        "relative_frobenius_error_upper_exact_ttsvd": relative_frobenius_error,
        "retained_frobenius_energy_fraction": 1.0 - relative_frobenius_error**2,
        "site_svd_diagnostics": singular_spectra,
    }


def circuit_storage_ledger(input_size: int, output_size: int) -> dict:
    input_qubits = math.ceil(math.log2(input_size))
    output_qubits = math.ceil(math.log2(output_size))
    input_amplitudes = 1 << input_qubits
    output_amplitudes = 1 << output_qubits

    def unitary_dof(qubits: int) -> int:
        dimension = 1 << qubits
        return dimension * dimension - 1

    def orthogonal_dof(qubits: int) -> int:
        dimension = 1 << qubits
        return dimension * (dimension - 1) // 2

    generic_unitary = unitary_dof(input_qubits) + unitary_dof(output_qubits)
    generic_real_orthogonal = orthogonal_dof(input_qubits) + orthogonal_dof(output_qubits)
    return {
        "input_qubits_after_padding": input_qubits,
        "input_padded_amplitudes": input_amplitudes,
        "input_padding_fraction": (input_amplitudes - input_size) / input_amplitudes,
        "output_qubits": output_qubits,
        "output_padded_amplitudes": output_amplitudes,
        "two_generic_complex_unitaries_real_dof": generic_unitary,
        "two_generic_real_orthogonals_dof": generic_real_orthogonal,
        "warning": (
            "These are storage lower-ledgers for generic full gates, not a claim that a shallow VQC needs this many angles. "
            "A shallow local-gate circuit uses fewer angles but no longer exactly encodes an arbitrary pretrained matrix."
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("QH029 feasibility audit is GPU-only")
    model_dir = Path((RECORDS / "active_model_path.txt").read_text(encoding="utf-8").strip())
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    layer = model.model.language_model.layers[args.layer]
    projection = getattr(layer.self_attn, args.target)
    weight = projection.weight.detach().float()
    output_size, input_size = weight.shape
    output_dims, input_dims = factor_plan(output_size, input_size, args.factor_plan)
    mpo_rows = [tt_svd_audit(weight, output_dims, input_dims, cap) for cap in args.bond_dims]
    ledger = circuit_storage_ledger(input_size, output_size)
    dense_parameters = weight.numel()
    for row in mpo_rows:
        row["deployment_count_with_generic_unitaries"] = row["mpo_parameter_count"] + ledger["two_generic_complex_unitaries_real_dof"]
        row["deployment_count_with_generic_real_orthogonals"] = row["mpo_parameter_count"] + ledger["two_generic_real_orthogonals_dof"]
        row["net_reduction_with_generic_unitaries"] = dense_parameters - row["deployment_count_with_generic_unitaries"]
        row["net_reduction_with_generic_real_orthogonals"] = dense_parameters - row["deployment_count_with_generic_real_orthogonals"]

    payload = {
        "status": "ok",
        "candidate_id": "QH-029-feasibility",
        "scope": "weight-only; no train, validation, or test data loaded",
        "model_dir": str(model_dir),
        "layer": args.layer,
        "target": args.target,
        "factor_plan": args.factor_plan,
        "weight_shape": list(weight.shape),
        "dense_parameter_count": dense_parameters,
        "mpo_output_dims": output_dims,
        "mpo_input_dims": input_dims,
        "mpo_physical_dims": [a * b for a, b in zip(output_dims, input_dims)],
        "mpo_rows": mpo_rows,
        "quantum_circuit_storage_ledger": ledger,
        "decision_rule": (
            "MPO fidelity alone cannot promote QH029. A later local-gate disentangler must jointly beat the original dense "
            "parameter count, satisfy an explicit matrix/action fidelity threshold, and pass GPU simulator latency before language fitness."
        ),
        "claim_limit": (
            "TT-SVD is classical. Generic-unitary degrees of freedom are a conservative accounting bound for exact arbitrary "
            "re-expression, not a measured shallow-circuit requirement or a quantum advantage claim."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
