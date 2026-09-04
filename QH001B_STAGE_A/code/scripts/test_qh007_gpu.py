#!/usr/bin/env python3
import json
import statistics
import sys
import time
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.data_reupload_correlator import QH007ResidualAdapter


def grad_norm(parameter: torch.Tensor) -> float:
    return float(parameter.grad.detach().float().norm())


def objective(adapter, x, probe):
    return (adapter(x).float() * probe).mean()


def finite_difference_vqc(vqc, angles, probe, parameter, index, step=1.0e-3):
    vqc.zero_grad(set_to_none=True)
    loss = (vqc(angles) * probe).mean()
    loss.backward()
    analytic = float(parameter.grad[index])
    original = float(parameter.data[index])
    with torch.no_grad():
        parameter.data[index] = original + step
        plus = float((vqc(angles) * probe).mean())
        parameter.data[index] = original - step
        minus = float((vqc(angles) * probe).mean())
        parameter.data[index] = original
    numerical = (plus - minus) / (2.0 * step)
    relative = abs(analytic - numerical) / max(abs(analytic), abs(numerical), 1.0e-8)
    return {"analytic": analytic, "numerical": numerical, "relative_error": relative}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("QH-007 validation is GPU-only")
    torch.manual_seed(20260828)
    device = torch.device("cuda:0")
    adapter = QH007ResidualAdapter().to(device)
    adapter.down.to(dtype=torch.bfloat16)
    adapter.up.to(dtype=torch.bfloat16)
    adapter.vqc.to(dtype=torch.float32)
    x = torch.randn(256, 5120, device=device, dtype=torch.bfloat16, requires_grad=True)
    probe = torch.randn(256, 5120, device=device, dtype=torch.float32)

    times = []
    for _ in range(2):
        adapter.zero_grad(set_to_none=True)
        x.grad = None
        objective(adapter, x, probe).backward()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(5):
        adapter.zero_grad(set_to_none=True)
        x.grad = None
        started = time.perf_counter()
        objective(adapter, x, probe).backward()
        torch.cuda.synchronize()
        times.append(1000.0 * (time.perf_counter() - started))

    with torch.no_grad():
        output = adapter(x.detach())
    delta = output.float() - x.detach().float()
    gradients = {
        "down.weight": grad_norm(adapter.down.weight),
        "vqc.theta": grad_norm(adapter.vqc.theta),
        "vqc.reupload_scale": grad_norm(adapter.vqc.reupload_scale),
        "up.weight": grad_norm(adapter.up.weight),
        "raw_scale": grad_norm(adapter.raw_scale),
    }

    # Check the circuit kernel before the BF16 projection boundary.  At the
    # adapter output the relevant loss differences are smaller than one BF16
    # quantization step, which makes central differences meaningless even when
    # autograd is correct.
    circuit_angles = torch.randn(16, 8, device=device, dtype=torch.float32)
    circuit_probe = torch.randn(16, 16, device=device, dtype=torch.float32)
    theta_check = finite_difference_vqc(
        adapter.vqc,
        circuit_angles,
        circuit_probe,
        adapter.vqc.theta,
        (0, 0, 0),
    )
    reupload_check = finite_difference_vqc(
        adapter.vqc,
        circuit_angles,
        circuit_probe,
        adapter.vqc.reupload_scale,
        (1, 0),
    )
    result = {
        "status": "ok",
        "candidate": "QH-007",
        "audit": adapter.audit(),
        "trainable_parameters": sum(p.numel() for p in adapter.parameters()),
        "input_shape": list(x.shape),
        "median_forward_backward_ms": statistics.median(times),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "relative_delta_rms": float(
            delta.square().mean().sqrt() / x.detach().float().square().mean().sqrt()
        ),
        "bf16_changed_fraction": float((output != x.detach()).float().mean()),
        "gradient_norms": gradients,
        "all_parameter_groups_receive_gradient": all(value > 0.0 for value in gradients.values()),
        "finite_difference_scope": "pure VQC FP32/complex64 before BF16 projection",
        "finite_difference": {"theta": theta_check, "reupload_scale": reupload_check},
        "finite_difference_pass": max(
            theta_check["relative_error"], reupload_check["relative_error"]
        ) < 0.02,
        "device": torch.cuda.get_device_name(device),
    }
    path = PROJECT_DIR / "artifacts/qh007-gpu-validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
