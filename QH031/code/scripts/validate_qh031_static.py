#!/usr/bin/env python
"""Static and gradient contract checks for QH031 on CUDA."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_qwen38.qh031_shared_basis_replacement import (  # noqa: E402
    DequantizedAuditCore,
    EqualParameterClassicalCore,
    ExactQuantumCore,
    QH031Config,
    build_shared_svd_replacements,
    qh031_parameter_audit,
    unique_trainable_parameters,
)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH031 static validation requires CUDA")
    torch.manual_seed(31031)
    torch.cuda.manual_seed_all(31031)
    device = torch.device("cuda")
    config = QH031Config(
        hidden_size=32,
        output_size=8,
        rank=16,
        layer_indices=(1, 3),
        num_qubits=4,
        depth=2,
        entangling_offsets=(1, 2),
    )
    quantum = ExactQuantumCore(config).to(device)
    dq = DequantizedAuditCore(config).to(device)
    classical = EqualParameterClassicalCore(config).to(device)
    x = torch.randn(3, 5, config.rank, device=device, requires_grad=True)

    q_identity = quantum(x)
    dq_identity = dq(x)
    cc_identity = classical(x)
    identity_errors = {
        "quantum": _max_abs(q_identity, x),
        "dq": _max_abs(dq_identity, x),
        "classical": _max_abs(cc_identity, x),
    }
    if max(identity_errors.values()) > 2.0e-6:
        raise AssertionError(f"identity initialization failed: {identity_errors}")

    theta = 0.15 * torch.randn_like(quantum.theta)
    quantum.theta.data.copy_(theta)
    dq.theta.data.copy_(theta)
    q_input = x.detach().clone().requires_grad_(True)
    d_input = x.detach().clone().requires_grad_(True)
    q_out = quantum(q_input)
    d_out = dq(d_input)
    weights = torch.randn_like(q_out)
    (q_out * weights).sum().backward()
    (d_out * weights).sum().backward()
    equivalence = {
        "forward_max_abs": _max_abs(q_out, d_out),
        "input_grad_max_abs": _max_abs(q_input.grad, d_input.grad),
        "theta_grad_max_abs": _max_abs(quantum.theta.grad, dq.theta.grad),
    }
    if equivalence["forward_max_abs"] > 1.0e-5:
        raise AssertionError(f"Q/DQ forward mismatch: {equivalence}")
    if equivalence["input_grad_max_abs"] > 2.0e-5:
        raise AssertionError(f"Q/DQ input-gradient mismatch: {equivalence}")
    if equivalence["theta_grad_max_abs"] > 2.0e-4:
        raise AssertionError(f"Q/DQ parameter-gradient mismatch: {equivalence}")

    dense = {
        1: nn.Linear(config.hidden_size, config.output_size, bias=False).to(
            device=device, dtype=torch.float32
        ),
        3: nn.Linear(config.hidden_size, config.output_size, bias=False).to(
            device=device, dtype=torch.float32
        ),
    }
    replacements = build_shared_svd_replacements(dense, config, branch="quantum")
    if replacements[1].shared_basis is not replacements[3].shared_basis:
        raise AssertionError("input basis is not shared")
    if replacements[1].core is not replacements[3].core:
        raise AssertionError("quantum core is not shared")
    trainable = list(unique_trainable_parameters(replacements))
    observed_trainable = sum(parameter.numel() for parameter in trainable)
    expected = qh031_parameter_audit(config)["trainable_parameters"]
    if observed_trainable != expected:
        raise AssertionError(
            f"unique trainable mismatch: observed={observed_trainable}, expected={expected}"
        )
    loss = sum(module(torch.randn(2, 4, config.hidden_size, device=device)).square().mean()
               for module in replacements.values())
    loss.backward()
    frozen_gradient_count = sum(
        int(parameter.grad is not None)
        for module in replacements.values()
        for name, parameter in module.named_parameters()
        if not parameter.requires_grad
    )
    trainable_gradient_count = sum(int(parameter.grad is not None) for parameter in trainable)
    if frozen_gradient_count != 0:
        raise AssertionError("a frozen shared-SVD parameter received gradients")
    if trainable_gradient_count != len(trainable):
        raise AssertionError("not all unique QH031 parameters received gradients")

    cpu_rejected = False
    try:
        ExactQuantumCore(config)(torch.randn(1, config.rank))
    except RuntimeError as error:
        cpu_rejected = "GPU-only" in str(error)
    if not cpu_rejected:
        raise AssertionError("CPU execution was not rejected")

    full_audit = qh031_parameter_audit(QH031Config())
    report = {
        "status": "pass",
        "cuda_device": torch.cuda.get_device_name(0),
        "identity_errors": identity_errors,
        "quantum_dequantized_equivalence": equivalence,
        "small_config_trainable_parameters": observed_trainable,
        "trainable_gradient_tensors": trainable_gradient_count,
        "frozen_gradient_tensors": frozen_gradient_count,
        "quantum_circuit_calls": int(quantum.circuit_calls.item()),
        "dequantized_circuit_calls": int(dq.circuit_calls.item()),
        "cpu_rejected": cpu_rejected,
        "full_qh031_parameter_audit": full_audit,
    }
    output = ROOT / "artifacts" / "qh031-static-validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

