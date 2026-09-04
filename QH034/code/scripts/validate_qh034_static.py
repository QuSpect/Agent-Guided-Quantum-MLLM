#!/usr/bin/env python3
"""CUDA identity, gradient, DQ, and control checks for QH034."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code/src"))
from quantum_qwen38.qh034_spectral_modulation_replacement import (  # noqa: E402
    DequantizedSpectralCore,
    EqualParameterClassicalSpectralCore,
    ExactQuantumSpectralCore,
    QH034Config,
    qh034_parameter_audit,
)


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).float().norm() / b.float().norm().clamp_min(1.0e-12))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH034 static validation requires CUDA")
    torch.manual_seed(20260856)
    device = torch.device("cuda:0")
    config = QH034Config(
        hidden_size=128, output_size=64, rank=64, layer_indices=(0, 1),
        num_qubits=6, depth=2, entangling_offsets=(1, 2, 3),
    )
    qh = ExactQuantumSpectralCore(config).to(device)
    dq = DequantizedSpectralCore(config).to(device)
    cc = EqualParameterClassicalSpectralCore(config).to(device)
    no_ent = ExactQuantumSpectralCore(config, use_entanglers=False).to(device)
    if {module.parameter_count for module in (qh, dq, cc, no_ent)} != {48}:
        raise AssertionError("small QH034 parameter counts differ")

    identity_input = torch.randn(3, 5, config.rank, device=device)
    identity_errors = {
        name: float((module(identity_input) - identity_input).float().abs().max())
        for name, module in (("qh", qh), ("dq", dq), ("cc", cc), ("no_ent", no_ent))
    }
    if max(identity_errors.values()) > 2.0e-7:
        raise AssertionError(f"zero-angle identity failed: {identity_errors}")

    with torch.no_grad():
        qh.theta.uniform_(-0.25, 0.25)
        dq.theta.copy_(qh.theta)
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
    }
    if checks["forward_max_abs"] > 2.0e-5:
        raise AssertionError(f"QH/DQ forward mismatch: {checks}")
    if checks["input_gradient_max_abs"] > 2.0e-5 or checks["theta_gradient_max_abs"] > 3.0e-4:
        raise AssertionError(f"QH/DQ gradient mismatch: {checks}")

    gradient_audit = {}
    for name, module in (("qh", qh), ("dq", dq), ("cc", cc), ("no_ent", no_ent)):
        module.zero_grad(set_to_none=True)
        with torch.no_grad():
            module.theta.uniform_(-0.2, 0.2)
        value = torch.randn(4, config.rank, device=device)
        module(value).float().square().mean().backward()
        gradient = module.theta.grad
        gradient_audit[name] = {
            "norm": float(gradient.norm()),
            "nonzero_elements": int((gradient != 0).sum()),
            "all_finite": bool(torch.isfinite(gradient).all()),
        }
        if gradient_audit[name]["norm"] == 0 or not gradient_audit[name]["all_finite"]:
            raise AssertionError(f"invalid {name} gradients")

    scale_audit = {}
    with torch.no_grad():
        for name, module in (("qh", qh), ("dq", dq), ("cc", cc), ("no_ent", no_ent)):
            output = module(torch.ones(1, config.rank, device=device)).float().squeeze(0)
            scale_audit[name] = {
                "minimum": float(output.min()),
                "mean": float(output.mean()),
                "maximum": float(output.max()),
            }
            if output.min() < 0 or abs(float(output.mean()) - 1.0) > 3.0e-6:
                raise AssertionError(f"invalid positive mean-one scale: {name}")

    cpu_rejected = False
    try:
        ExactQuantumSpectralCore(config)(torch.randn(1, config.rank))
    except RuntimeError:
        cpu_rejected = True
    if not cpu_rejected:
        raise AssertionError("CPU exact-quantum path was not rejected")
    full_audit = qh034_parameter_audit(QH034Config())
    if full_audit["trainable_parameters"] != 82 or full_audit["net_parameter_reduction"] != 3145646:
        raise AssertionError(f"full parameter audit changed: {full_audit}")
    payload = {
        "status": "pass",
        "candidate": "QH-034 quantum shared-SVD spectral modulation",
        "small_config": config.__dict__,
        "small_parameter_count_per_core": qh.parameter_count,
        "zero_angle_identity_max_abs": identity_errors,
        "dequantization_checks": checks,
        "gradient_audit": gradient_audit,
        "positive_mean_one_scale_audit": scale_audit,
        "cpu_quantum_path_rejected": cpu_rejected,
        "full_parameter_audit": full_audit,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = PROJECT / "artifacts/qh034-static-validation.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
