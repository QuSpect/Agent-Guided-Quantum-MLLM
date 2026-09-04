#!/usr/bin/env python3
"""CUDA static gate for the QH-023 Pauli/Stiefel prototype."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code/src"))

from quantum_qwen38.pauli_stiefel_adapter import (
    ClassicalStiefelAdapter,
    DequantizedPauliStiefelAdapter,
    PauliStiefelAdapter,
    QH023Config,
)


def probe(kind: str) -> dict:
    config = QH023Config()
    adapter = (
        ClassicalStiefelAdapter(config)
        if kind == "classical"
        else DequantizedPauliStiefelAdapter(config)
        if kind == "dequantized"
        else PauliStiefelAdapter(config, entangle=kind == "quantum")
    ).cuda()
    x = torch.randn(2, 3, config.hidden_size, device="cuda", requires_grad=True)
    before = adapter.circuit_calls
    residual = adapter.residual(x).float()
    calls = adapter.circuit_calls - before
    weight = torch.randn_like(residual)
    loss = (residual * weight).sum()
    loss.backward()
    gradients = {
        name: float(parameter.grad.detach().norm())
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    theta = adapter.circuits["u_high"].theta
    index = int(theta.grad.detach().abs().argmax())
    analytic = float(theta.grad.detach()[index])
    epsilon = 1.0e-3
    with torch.no_grad():
        original = float(theta[index])
        theta[index] = original + epsilon
        plus = float((adapter.residual(x).float() * weight).sum())
        theta[index] = original - epsilon
        minus = float((adapter.residual(x).float() * weight).sum())
        theta[index] = original
        gamma = float(adapter.gamma)
        adapter.gamma.zero_()
        branch_off = float((adapter(x) - x).abs().max())
        adapter.gamma.fill_(gamma)
    return {
        "kind": kind,
        "audit": adapter.audit(),
        "finite": bool(torch.isfinite(residual).all()),
        "circuit_calls_per_forward": calls,
        "orthogonality_max_error": max(adapter.last_orthogonality_error.values()),
        "gradient_norms": gradients,
        "all_six_gradient_groups_nonzero": len(gradients) == 6 and all(v > 0 for v in gradients.values()),
        "finite_difference_absolute_error": abs(analytic - (plus - minus) / (2 * epsilon)),
        "gamma_zero_max_difference": branch_off,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH-023 static validation is CUDA-only")
    torch.manual_seed(20260835)
    records = [
        probe(kind)
        for kind in ("quantum", "dequantized", "classical", "no_entanglement")
    ]
    equivalence_config = QH023Config()
    quantum = PauliStiefelAdapter(equivalence_config).cuda()
    dequantized = DequantizedPauliStiefelAdapter(equivalence_config).cuda()
    dequantized.load_state_dict(quantum.state_dict(), strict=True)
    x_q = torch.randn(1, 3, equivalence_config.hidden_size, device="cuda", requires_grad=True)
    x_dq = x_q.detach().clone().requires_grad_(True)
    q_output = quantum.residual(x_q).float()
    dq_output = dequantized.residual(x_dq).float()
    objective_weight = torch.randn_like(q_output)
    (q_output * objective_weight).sum().backward()
    (dq_output * objective_weight).sum().backward()
    q_gradients = dict(quantum.named_parameters())
    dq_gradients = dict(dequantized.named_parameters())
    gradient_errors = {
        name: float((parameter.grad - dq_gradients[name].grad).abs().max())
        for name, parameter in q_gradients.items()
    }
    equivalence = {
        "forward_max_absolute_error": float((q_output - dq_output).detach().abs().max()),
        "input_gradient_max_absolute_error": float((x_q.grad - x_dq.grad).detach().abs().max()),
        "parameter_gradient_max_absolute_error": max(gradient_errors.values()),
        "parameter_gradient_errors": gradient_errors,
    }
    cpu_rejected = False
    try:
        PauliStiefelAdapter(QH023Config())(torch.randn(1, 1, 5120))
    except RuntimeError:
        cpu_rejected = True
    passed = (
        cpu_rejected
        and all(record["audit"]["trainable_parameters"] == 209 for record in records)
        and all(record["audit"]["angle_parameters"] == 204 for record in records)
        and all(record["finite"] for record in records)
        and all(record["circuit_calls_per_forward"] == 4 for record in records)
        and all(record["orthogonality_max_error"] < 1.0e-5 for record in records)
        and all(record["all_six_gradient_groups_nonzero"] for record in records)
        and all(record["finite_difference_absolute_error"] < 1.0e-3 for record in records)
        and all(record["gamma_zero_max_difference"] == 0.0 for record in records)
        and equivalence["forward_max_absolute_error"] < 2.0e-5
        and equivalence["input_gradient_max_absolute_error"] < 1.0e-4
        and equivalence["parameter_gradient_max_absolute_error"] < 1.0e-4
    )
    result = {
        "status": "pass" if passed else "fail",
        "candidate": "QH-023-static",
        "device": torch.cuda.get_device_name(),
        "cpu_path_rejected": cpu_rejected,
        "records": records,
        "qh_dq_equivalence": equivalence,
        "performance_data_read": False,
        "claim": "static feasibility only",
    }
    output = PROJECT / "artifacts" / "qh023-gpu-validation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
