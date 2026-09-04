"""GPU-only validation for the evolved QH-006e real-amplitude unitary."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from quantum_qwen38.amplitude_unitary import (
    AmplitudeUnitaryConfig,
    BlockAmplitudeUnitary,
)


def make_model(device: torch.device) -> BlockAmplitudeUnitary:
    return BlockAmplitudeUnitary(
        AmplitudeUnitaryConfig(
            width=5120,
            register_widths=(4096, 1024),
            depth=2,
            permutation_seed=20260828,
        )
    ).to(device)


def finite_difference_check(device: torch.device) -> dict:
    torch.manual_seed(7)
    model = BlockAmplitudeUnitary(
        AmplitudeUnitaryConfig(
            width=1024,
            register_widths=(1024,),
            depth=2,
            permutation_seed=7,
        )
    ).to(device)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.02, 0.02)
    x = torch.randn(2, 1024, device=device)
    probe = torch.randn_like(x)
    loss = (model(x) * probe).mean()
    loss.backward()
    register = model.registers[0]
    checks = {}
    epsilon = 1.0e-3
    for label, parameter in (
        ("local", register.local_angles),
        ("entangling", register.entangling_angles),
    ):
        automatic = float(parameter.grad[0, 0])
        with torch.no_grad():
            old = float(parameter[0, 0])
            parameter[0, 0] = old + epsilon
            plus = float((model(x) * probe).mean())
            parameter[0, 0] = old - epsilon
            minus = float((model(x) * probe).mean())
            parameter[0, 0] = old
        central = (plus - minus) / (2.0 * epsilon)
        checks[label] = {
            "autograd": automatic,
            "central_difference": central,
            "relative_error": abs(automatic - central)
            / max(abs(automatic), abs(central), 1.0e-12),
        }
    return {
        "epsilon": epsilon,
        "checks": checks,
        "pass": all(value["relative_error"] < 0.02 for value in checks.values()),
    }


def teacher_recovery(device: torch.device) -> dict:
    torch.manual_seed(37)
    student = make_model(device)
    teacher = make_model(device)
    with torch.no_grad():
        for parameter in teacher.parameters():
            parameter.uniform_(-0.12, 0.12)
    x = torch.randn(16, 5120, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        target = teacher(x)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.01, weight_decay=0.0)
    trace = []
    for step in range(101):
        optimizer.zero_grad(set_to_none=True)
        output = student(x)
        loss = torch.nn.functional.mse_loss(output.float(), target.float())
        if step in {0, 1, 2, 5, 10, 20, 50, 100}:
            trace.append(
                {
                    "step": step,
                    "mse": float(loss.detach()),
                    "relative_output_error": float(
                        (output.float() - target.float()).norm()
                        / target.float().norm().clamp_min(1.0e-12)
                    ),
                }
            )
        if step < 100:
            loss.backward()
            optimizer.step()
    return {
        "steps": 100,
        "learning_rate": 0.01,
        "teacher_angle_range": [-0.12, 0.12],
        "trace": trace,
        "loss_reduction_factor": trace[0]["mse"] / trace[-1]["mse"],
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.manual_seed(20260828)
    model = make_model(device)
    x = torch.randn(256, 5120, device=device, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn_like(x)
    samples = []
    torch.cuda.reset_peak_memory_stats(device)
    for iteration in range(7):
        model.zero_grad(set_to_none=True)
        x.grad = None
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = model(x)
        torch.nn.functional.mse_loss(output.float(), target.float()).backward()
        torch.cuda.synchronize()
        if iteration >= 3:
            samples.append((time.perf_counter() - started) * 1000)
    gradients = {
        name: float(parameter.grad.norm())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    result = {
        "candidate": "QH-006e",
        "audit": model.audit(),
        "tokens": 256,
        "hidden_size": 5120,
        "median_forward_backward_ms": statistics.median(samples),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "identity_initialization_relative_error": float(
            (output.float() - x.float()).norm() / x.float().norm()
        ),
        "all_gradients_nonzero": all(value > 0 for value in gradients.values()),
        "gradient_norms": gradients,
        "finite_difference": finite_difference_check(device),
        "teacher_recovery": teacher_recovery(device),
        "device": torch.cuda.get_device_name(device),
    }
    output_path = Path("artifacts/qh006e-gpu-validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
