#!/usr/bin/env python3
"""Post-training structure diagnostics for QH-010 two-qubit Cayley blocks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.cayley_two_qubit import QH010Config, TwoQubitBlockTransform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--product-state-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def summary(values: torch.Tensor) -> dict:
    values = values.detach().float().cpu().numpy()
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def random_qubit(count: int, device: torch.device) -> torch.Tensor:
    state = torch.randn(count, 2, device=device) + 1j * torch.randn(count, 2, device=device)
    return state / torch.linalg.vector_norm(state, dim=-1, keepdim=True)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("QH-010 diagnostics are GPU-only")
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    theta_keys = [key for key in checkpoint if key.endswith("transform.theta")]
    if len(theta_keys) != 1:
        raise ValueError(f"expected one QH-010 theta tensor, got {theta_keys}")
    theta = checkpoint[theta_keys[0]].float()
    config = QH010Config(hidden_size=theta.shape[0] * 4)
    module = TwoQubitBlockTransform(config, core_kind="quantum").cuda()
    with torch.no_grad():
        module.theta.copy_(theta.cuda())
        unitary = module._quantum_unitary().to(torch.complex64)
        identity = torch.eye(4, device="cuda", dtype=torch.complex64).expand_as(unitary)
        unitarity = unitary.mH @ unitary
        unitarity_error = torch.linalg.matrix_norm(unitarity - identity) / 2.0
        identity_distance = torch.linalg.matrix_norm(unitary - identity) / 2.0

        # Operator Schmidt decomposition across the natural 1-qubit | 1-qubit cut.
        reshaped = unitary.reshape(-1, 2, 2, 2, 2).permute(0, 1, 3, 2, 4).reshape(-1, 4, 4)
        singular = torch.linalg.svdvals(reshaped)
        probabilities = singular.square() / singular.square().sum(dim=-1, keepdim=True)
        operator_entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log2()).sum(dim=-1)
        operator_rank = (singular > singular[:, :1] * 1.0e-6).sum(dim=-1)

        entanglement = torch.zeros(unitary.shape[0], device="cuda")
        for _ in range(args.product_state_samples):
            first = random_qubit(unitary.shape[0], unitary.device)
            second = random_qubit(unitary.shape[0], unitary.device)
            product = torch.einsum("bi,bj->bij", first, second).reshape(-1, 4)
            output = torch.einsum("bij,bj->bi", unitary, product).reshape(-1, 2, 2)
            rho_first = output @ output.mH
            purity = torch.einsum("bij,bji->b", rho_first, rho_first).real
            entanglement += 1.0 - purity
        entangling_power = entanglement / args.product_state_samples

        random_real = torch.randn(256, unitary.shape[0], 4, device="cuda")
        transformed = torch.einsum("bij,nbj->nbi", unitary.real, random_real)
        sign_flip_rate = (torch.sign(transformed) != torch.sign(random_real)).float().mean(dim=(0, 2))
        reconstructed = transformed.abs() * torch.sign(random_real)
        reconstruction_difference = torch.linalg.vector_norm(
            reconstructed - transformed, dim=-1
        ) / torch.linalg.vector_norm(transformed, dim=-1).clamp_min(1.0e-12)

    payload = {
        "status": "ok",
        "candidate": "QH-010",
        "checkpoint": str(args.checkpoint),
        "theta_key": theta_keys[0],
        "blocks": int(theta.shape[0]),
        "parameters": int(theta.numel()),
        "theta_l2": float(theta.norm()),
        "theta_abs_max": float(theta.abs().max()),
        "unitarity_error": summary(unitarity_error),
        "identity_distance": summary(identity_distance),
        "near_identity_fraction_distance_lt_1e_3": float((identity_distance < 1.0e-3).float().mean()),
        "operator_schmidt_rank_counts": {
            str(rank): int((operator_rank == rank).sum()) for rank in torch.unique(operator_rank).tolist()
        },
        "operator_entanglement_entropy_bits": summary(operator_entropy),
        "entangling_power_linear_entropy": summary(entangling_power),
        "random_gaussian_sign_flip_rate": summary(sign_flip_rate),
        "random_gaussian_sign_reconstruction_relative_difference": summary(
            reconstruction_difference.flatten()
        ),
        "interpretation_limits": [
            "random-Gaussian sign diagnostics are not a substitute for Qwen activation capture",
            "two-qubit blocks are classically exactly simulable",
            "entanglement is diagnostic and not evidence of task or computational advantage",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
