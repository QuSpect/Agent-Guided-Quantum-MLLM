#!/usr/bin/env python3
"""Static GPU tests for the QH-024 true replacement module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code/src"))

from quantum_qwen38.hyqut_replacement import HybridValueReplacement, QH024Config


def make(kind: str, config: QH024Config, dense: nn.Linear) -> HybridValueReplacement:
    return HybridValueReplacement.from_dense(dense, config, core_kind=kind)


def grad_vector(module: nn.Module) -> torch.Tensor:
    return torch.cat([
        p.grad.detach().flatten().float()
        for p in module.parameters()
        if p.requires_grad and p.grad is not None
    ])


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH-024 static tests are GPU-only")
    device = torch.device("cuda:0")
    torch.manual_seed(20260838)
    config = QH024Config(
        hidden_size=64, output_size=16, svd_rank=8,
        n_qubits=4, depth=2, gamma_init=0.1,
    )
    dense = nn.Linear(64, 16, bias=False, device=device, dtype=torch.float32)
    qh, dq = make("quantum", config, dense), make("dequantized", config, dense)
    cc, noent = make("classical", config, dense), make("no_entanglement", config, dense)
    dq.load_state_dict(qh.state_dict(), strict=True)
    expected = 64 * 8 + 16 * 4 + 3 * 4 * 2 + 1
    counts = {
        "qh024": qh.trainable_parameter_count(), "dq024": dq.trainable_parameter_count(),
        "cc024": cc.trainable_parameter_count(), "qh024_noent": noent.trainable_parameter_count(),
    }
    if set(counts.values()) != {expected}:
        raise AssertionError(f"matched parameter count failed: {counts}")

    x_qh = torch.randn(2, 3, 64, device=device, requires_grad=True)
    x_dq = x_qh.detach().clone().requires_grad_(True)
    y_qh, y_dq = qh.student_forward(x_qh), dq.student_forward(x_dq)
    forward_error = float((y_qh - y_dq).abs().max())
    y_qh.square().mean().backward()
    y_dq.square().mean().backward()
    input_gradient_error = float((x_qh.grad - x_dq.grad).abs().max())
    parameter_gradient_error = float((grad_vector(qh) - grad_vector(dq)).abs().max())
    if forward_error > 2e-6 or input_gradient_error > 2e-6 or parameter_gradient_error > 2e-5:
        raise AssertionError("QH-024/DQ-024 equivalence failed")
    gradient_groups = {
        "down": float(qh.down.weight.grad.abs().max()),
        "up": float(qh.up.weight.grad.abs().max()),
        "theta": float(qh.core.theta.grad.abs().max()),
        "gamma": float(qh.gamma.grad.abs()),
    }
    if any(value <= 0 or not torch.isfinite(torch.tensor(value)) for value in gradient_groups.values()):
        raise AssertionError(f"missing/nonfinite gradient: {gradient_groups}")

    qh.set_capture_teacher_only(True)
    teacher = qh(torch.randn(1, 2, 64, device=device))
    captured_x, captured_target = qh.take_capture()
    if float((teacher - captured_target).abs().max()) != 0.0 or captured_x.shape != (1, 2, 64):
        raise AssertionError("teacher capture failed")
    qh.set_capture_teacher_only(False)
    qh.strip_teacher()
    if qh.teacher_weight is not None:
        raise AssertionError("teacher buffer was not stripped")

    cpu_module = make("quantum", config, nn.Linear(64, 16, bias=False))
    try:
        cpu_module.student_forward(torch.randn(1, 1, 64))
        cpu_rejected = False
    except RuntimeError as error:
        cpu_rejected = "CUDA-only" in str(error)
    if not cpu_rejected:
        raise AssertionError("QH-024 CPU execution was not rejected")

    payload = {
        "status": "ok",
        "expected_trainable_parameters": expected,
        "matched_trainable_parameter_counts": counts,
        "audit": qh.audit(),
        "qh_dq_equivalence": {
            "forward_max_abs_error": forward_error,
            "input_gradient_max_abs_error": input_gradient_error,
            "parameter_gradient_max_abs_error": parameter_gradient_error,
            "exactly_dequantizable_within_tolerance": True,
        },
        "gradient_groups": gradient_groups,
        "teacher_buffer_stripped": True,
        "cpu_execution_rejected": cpu_rejected,
        "claim_ceiling": "quantum-executable layer replacement; exactly dequantized on the tested map",
    }
    output = PROJECT / "artifacts/qh024-gpu-validation.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
