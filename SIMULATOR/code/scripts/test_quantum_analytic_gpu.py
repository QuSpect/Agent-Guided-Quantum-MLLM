#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.quantum_residual import NativeStatevectorVQC


def main() -> None:
    torch.manual_seed(20260828)
    device = torch.device("cuda:0")

    angles = (torch.rand(32, 8, device=device) * 2.0 - 1.0).requires_grad_(True)
    analytic_circuit = NativeStatevectorVQC(n_qubits=8, depth=0).to(device)
    measured = analytic_circuit(angles)
    expected = torch.cos(angles)
    forward_max_abs_error = float((measured - expected).abs().max())
    measured.sum().backward()
    gradient_max_abs_error = float((angles.grad - (-torch.sin(angles.detach()))).abs().max())

    trainable_circuit = NativeStatevectorVQC(n_qubits=8, depth=1).to(device)
    fixed_angles = torch.randn(16, 8, device=device)
    loss = trainable_circuit(fixed_angles).square().mean()
    loss.backward()
    autograd_value = float(trainable_circuit.theta.grad[0, 0, 0])

    epsilon = 1.0e-3
    with torch.no_grad():
        original = trainable_circuit.theta[0, 0, 0].clone()
        trainable_circuit.theta[0, 0, 0] = original + epsilon
        plus = float(trainable_circuit(fixed_angles).square().mean())
        trainable_circuit.theta[0, 0, 0] = original - epsilon
        minus = float(trainable_circuit(fixed_angles).square().mean())
        trainable_circuit.theta[0, 0, 0] = original
    finite_difference_value = (plus - minus) / (2.0 * epsilon)
    finite_difference_abs_error = abs(autograd_value - finite_difference_value)
    finite_difference_relative_error = finite_difference_abs_error / max(
        abs(autograd_value), abs(finite_difference_value), 1.0e-12
    )

    result = {
        "status": "ok",
        "device": torch.cuda.get_device_name(0),
        "analytic_forward_max_abs_error": forward_max_abs_error,
        "analytic_input_gradient_max_abs_error": gradient_max_abs_error,
        "trainable_gate_autograd": autograd_value,
        "trainable_gate_finite_difference": finite_difference_value,
        "trainable_gate_abs_error": finite_difference_abs_error,
        "trainable_gate_relative_error": finite_difference_relative_error,
        "passes": {
            "forward": forward_max_abs_error < 1.0e-5,
            "input_gradient": gradient_max_abs_error < 1.0e-5,
            "trainable_gate_gradient": finite_difference_relative_error < 2.0e-2,
        },
    }
    if not all(result["passes"].values()):
        result["status"] = "failed"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

