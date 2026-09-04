#!/usr/bin/env python3
"""Task-independent local geometry diagnostic for QH-009 and its controls.

This is not a quantum-advantage metric.  It measures the empirical Jacobian
Gram matrix of the 16 gate observables with respect to the 32 core parameters
on the same fixed angle inputs.  Effective rank and conditioning indicate
whether a core exposes trainable local directions; downstream quality and
causal controls remain the promotion criteria.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.correlation_gated_low_rank import (
    CorrelatorClassicalGateCore,
    CorrelatorNoEntanglementVQC,
    CorrelatorStatevectorVQC,
)


def core_parameter(module: torch.nn.Module) -> torch.nn.Parameter:
    for name in ("theta", "affine"):
        value = getattr(module, name, None)
        if value is not None:
            return value
    raise ValueError("core has no registered comparison parameter")


def empirical_geometry(module: torch.nn.Module, angles: torch.Tensor) -> dict:
    parameter = core_parameter(module)
    gram = torch.zeros(parameter.numel(), parameter.numel(), device=angles.device)
    row_norms = []
    outputs = []
    for sample in angles:
        output = module(sample.unsqueeze(0)).squeeze(0)
        outputs.append(output.detach())
        for coordinate in range(output.numel()):
            gradient = torch.autograd.grad(
                output[coordinate], parameter, retain_graph=True, create_graph=False
            )[0].reshape(-1)
            gram.add_(torch.outer(gradient, gradient))
            row_norms.append(float(gradient.norm().detach()))
    gram.div_(angles.shape[0] * outputs[0].numel())
    eigenvalues = torch.linalg.eigvalsh(gram.double()).clamp_min(0.0)
    total = eigenvalues.sum()
    squared_total = eigenvalues.square().sum()
    maximum = eigenvalues.max()
    positive = eigenvalues[eigenvalues > maximum * 1.0e-6]
    effective_rank = float((total.square() / squared_total).cpu()) if squared_total > 0 else 0.0
    condition = (
        float((positive.max() / positive.min()).cpu()) if positive.numel() > 1 else None
    )
    output_matrix = torch.stack(outputs).float()
    centered = output_matrix - output_matrix.mean(dim=0, keepdim=True)
    output_singular = torch.linalg.svdvals(centered.double())
    output_energy = output_singular.square()
    output_effective_rank = float(
        (output_energy.sum().square() / output_energy.square().sum()).cpu()
    )
    return {
        "parameter_count": parameter.numel(),
        "observable_count": outputs[0].numel(),
        "sample_count": angles.shape[0],
        "jacobian_gram_trace": float(total.cpu()),
        "jacobian_effective_rank_participation_ratio": effective_rank,
        "jacobian_numerical_rank_relative_1e-6": int(positive.numel()),
        "jacobian_positive_condition_number": condition,
        "gradient_row_norm_mean": float(torch.tensor(row_norms).mean()),
        "gradient_row_norm_std": float(torch.tensor(row_norms).std()),
        "observable_mean_abs": float(output_matrix.abs().mean()),
        "observable_saturation_rate_abs_098": float((output_matrix.abs() >= 0.98).float().mean()),
        "observable_effective_rank": output_effective_rank,
        "eigenvalues_descending": [
            float(value) for value in eigenvalues.flip(0).detach().cpu()
        ],
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH-009 local geometry diagnostic is GPU-only")
    torch.manual_seed(20260829)
    torch.cuda.manual_seed_all(20260829)
    device = torch.device("cuda:0")
    angles = (2.0 * torch.rand(32, 8, device=device) - 1.0) * math.pi
    factories = {
        "quantum": CorrelatorStatevectorVQC,
        "classical": CorrelatorClassicalGateCore,
        "no_entanglement": CorrelatorNoEntanglementVQC,
    }
    results = {}
    for name, factory in factories.items():
        torch.manual_seed(20260909)
        module = factory(8, 2).to(device).float()
        results[name] = empirical_geometry(module, angles)
    result = {
        "status": "ok",
        "candidate": "QH-009",
        "diagnostic": "empirical observable-Jacobian Gram geometry",
        "claim_boundary": (
            "diagnostic only; higher effective rank is neither task improvement nor quantum advantage"
        ),
        "fixed_angle_seed": 20260829,
        "core_initialization_seed": 20260909,
        "results": results,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    path = PROJECT_DIR / "artifacts" / "qh009-local-geometry.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
