#!/usr/bin/env python3
"""CUDA-only static gate for QH-022 and its matched controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code/src"))

from quantum_qwen38.prefill_parameter_generator import (
    QH022Config,
    SimulatorParameterGeneratedAdapter,
)


def probe(kind: str, config: QH022Config) -> dict:
    adapter = SimulatorParameterGeneratedAdapter(config, core_kind=kind).cuda()
    x = torch.randn(2, 7, config.hidden_size, device="cuda", requires_grad=True)
    before = adapter.core.circuit_calls
    residual = adapter.residual(x).float()
    calls = adapter.core.circuit_calls - before
    weight = torch.linspace(-0.2, 0.3, residual.numel(), device="cuda").reshape_as(residual)
    objective = (residual * weight).sum()
    objective.backward()
    gradients = {
        name: float(parameter.grad.detach().norm())
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    theta = adapter.core.theta
    flat_index = int(theta.grad.detach().abs().argmax())
    analytic = float(theta.grad.detach().flatten()[flat_index])
    epsilon = 1.0e-3
    with torch.no_grad():
        original = float(theta.flatten()[flat_index])
        theta.flatten()[flat_index] = original + epsilon
        plus = float((adapter.residual(x).float() * weight).sum())
        theta.flatten()[flat_index] = original - epsilon
        minus = float((adapter.residual(x).float() * weight).sum())
        theta.flatten()[flat_index] = original
        gamma = float(adapter.gamma)
        adapter.gamma.zero_()
        branch_off_difference = float((adapter(x) - x).abs().max())
        adapter.gamma.fill_(gamma)
    return {
        "kind": kind,
        "parameters": sum(p.numel() for p in adapter.parameters()),
        "one_generator_call_per_forward": calls == 1,
        "finite": bool(torch.isfinite(residual).all()),
        "gradient_norms": gradients,
        "all_gradient_groups_nonzero": len(gradients) == 2 and all(v > 0 for v in gradients.values()),
        "finite_difference_absolute_error": abs(analytic - (plus - minus) / (2 * epsilon)),
        "coefficient_min": float(adapter.last_coefficients.min()),
        "coefficient_max": float(adapter.last_coefficients.max()),
        "branch_off_max_difference": branch_off_difference,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH-022 static validation is CUDA-only")
    torch.manual_seed(20260829)
    config = QH022Config(hidden_size=64, rank=12, n_qubits=6, depth=2)
    records = {kind: probe(kind, config) for kind in ("quantum", "classical", "no_entanglement")}
    counts = {record["parameters"] for record in records.values()}
    cpu_rejected = False
    try:
        SimulatorParameterGeneratedAdapter(config, core_kind="quantum")(
            torch.randn(2, 7, config.hidden_size)
        )
    except RuntimeError:
        cpu_rejected = True
    passed = (
        len(counts) == 1
        and counts == {25}
        and cpu_rejected
        and all(record["finite"] for record in records.values())
        and all(record["one_generator_call_per_forward"] for record in records.values())
        and all(record["all_gradient_groups_nonzero"] for record in records.values())
        and all(record["finite_difference_absolute_error"] < 1.0e-3 for record in records.values())
        and all(record["branch_off_max_difference"] == 0.0 for record in records.values())
    )
    result = {
        "status": "pass" if passed else "fail",
        "candidate": "QH-022",
        "device": torch.cuda.get_device_name(),
        "equal_parameter_count": len(counts) == 1,
        "cpu_path_rejected": cpu_rejected,
        "records": records,
    }
    output = PROJECT / "artifacts" / "qh022-gpu-validation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
