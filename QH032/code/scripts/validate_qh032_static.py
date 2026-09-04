#!/usr/bin/env python3
"""CUDA static, gradient, and dequantization checks for QH032."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code/src"))
from quantum_qwen38.qh032_pauli_observable_replacement import (  # noqa: E402
    DequantizedPauliObservableCore,
    EqualParameterClassicalObservableCore,
    ExactPauliObservableCore,
    QH032Config,
    qh032_parameter_audit,
)


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).float().norm() / b.float().norm().clamp_min(1.0e-12))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH032 static validation requires CUDA")
    torch.manual_seed(20260849)
    device = torch.device("cuda:0")
    config = QH032Config(hidden_size=32, output_size=16, rank=16, layer_indices=(0, 1), num_qubits=4, observable_offsets=(1, 2))
    qh = ExactPauliObservableCore(config).to(device)
    dq = DequantizedPauliObservableCore(config).to(device)
    cc = EqualParameterClassicalObservableCore(config).to(device)
    no_ent = ExactPauliObservableCore(config, use_entanglers=False).to(device)
    if {module.parameter_count for module in (qh, dq, cc, no_ent)} != {32}:
        raise AssertionError("small-core parameter counts are not equal")

    identity_input = torch.randn(3, 5, config.rank, device=device, requires_grad=True)
    identity_error = float((qh(identity_input) - identity_input).float().abs().max())
    if identity_error > 1.0e-7:
        raise AssertionError(f"zero-alpha QH032 is not identity: {identity_error}")

    with torch.no_grad():
        qh.theta.uniform_(-0.25, 0.25)
        qh.alpha.uniform_(-0.08, 0.08)
        dq.theta.copy_(qh.theta)
        dq.alpha.copy_(qh.alpha)
    xq = torch.randn(2, 3, config.rank, device=device, requires_grad=True)
    xd = xq.detach().clone().requires_grad_(True)
    yq, yd = qh(xq), dq(xd)
    probe = torch.randn_like(yq)
    (yq * probe).sum().backward()
    (yd * probe).sum().backward()
    checks = {
        "forward_max_abs": float((yq - yd).float().abs().max()),
        "forward_relative_l2": relative(yq, yd),
        "input_gradient_max_abs": float((xq.grad - xd.grad).float().abs().max()),
        "input_gradient_relative_l2": relative(xq.grad, xd.grad),
        "theta_gradient_max_abs": float((qh.theta.grad - dq.theta.grad).abs().max()),
        "theta_gradient_relative_l2": relative(qh.theta.grad, dq.theta.grad),
        "alpha_gradient_max_abs": float((qh.alpha.grad - dq.alpha.grad).abs().max()),
        "alpha_gradient_relative_l2": relative(qh.alpha.grad, dq.alpha.grad),
    }
    if checks["forward_max_abs"] > 2.0e-5 or checks["input_gradient_max_abs"] > 2.0e-5:
        raise AssertionError(f"QH/DQ mismatch: {checks}")
    if checks["theta_gradient_max_abs"] > 3.0e-4 or checks["alpha_gradient_max_abs"] > 3.0e-4:
        raise AssertionError(f"QH/DQ gradient mismatch: {checks}")

    gradient_audit = {}
    for name, module in (("qh", qh), ("dq", dq), ("cc", cc), ("no_ent", no_ent)):
        module.zero_grad(set_to_none=True)
        with torch.no_grad():
            module.theta.uniform_(-0.2, 0.2)
            module.alpha.uniform_(-0.1, 0.1)
        value = torch.randn(4, config.rank, device=device)
        module(value).float().square().mean().backward()
        gradient_audit[name] = {
            "theta_norm": float(module.theta.grad.norm()),
            "alpha_norm": float(module.alpha.grad.norm()),
            "all_finite": bool(torch.isfinite(module.theta.grad).all() and torch.isfinite(module.alpha.grad).all()),
        }
        if gradient_audit[name]["theta_norm"] == 0 or gradient_audit[name]["alpha_norm"] == 0 or not gradient_audit[name]["all_finite"]:
            raise AssertionError(f"invalid {name} gradients")

    cpu_rejected = False
    try:
        ExactPauliObservableCore(config)(torch.randn(1, config.rank))
    except RuntimeError:
        cpu_rejected = True
    if not cpu_rejected:
        raise AssertionError("CPU exact-quantum path was not rejected")
    full_audit = qh032_parameter_audit(QH032Config())
    if full_audit["trainable_parameters"] != 82 or full_audit["net_parameter_reduction"] != 3145646:
        raise AssertionError(f"full parameter audit changed: {full_audit}")
    payload = {
        "status": "pass",
        "candidate": "QH-032 Pauli-observable nonlinear shared-basis replacement",
        "small_config": config.__dict__,
        "small_parameter_count_per_core": qh.parameter_count,
        "zero_alpha_identity_max_abs": identity_error,
        "dequantization_checks": checks,
        "gradient_audit": gradient_audit,
        "cpu_quantum_path_rejected": cpu_rejected,
        "full_parameter_audit": full_audit,
        "simulator_calls": int(qh.circuit_calls.item()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = PROJECT / "artifacts/qh032-static-validation.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

