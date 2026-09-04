#!/usr/bin/env python3
"""Static GPU validation for QH-012 and all attribution controls."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.brickwork_four_qubit import QH012Config, FourQubitBrickworkTransform


def unravel(flat_index: int, shape: torch.Size) -> tuple[int, ...]:
    indices = []
    for width in reversed(shape):
        indices.append(flat_index % width)
        flat_index //= width
    return tuple(reversed(indices))


def objective(output: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (weights * output.float().square()).mean() + 0.01 * output.float().mean()


def measure(kind: str, config: QH012Config, x: torch.Tensor) -> dict:
    torch.manual_seed(20260829)
    module = FourQubitBrickworkTransform(config, core_kind=kind).cuda()
    parameter_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    y = module(x)
    weights = torch.linspace(0.25, 1.75, y.numel(), device=y.device).reshape_as(y)
    loss = objective(y, weights)
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    gradient = module.theta.grad.detach()
    index = unravel(int(gradient.abs().argmax()), gradient.shape)
    analytic = float(gradient[index])
    epsilon = 1.0e-2
    with torch.no_grad():
        original = float(module.theta[index])
        module.theta[index] = original + epsilon
        plus = float(objective(module(x), weights))
        module.theta[index] = original - epsilon
        minus = float(objective(module(x), weights))
        module.theta[index] = original
    finite_difference = (plus - minus) / (2.0 * epsilon)
    return {
        "kind": kind,
        "parameter_count": parameter_count,
        "output_rms": float(y.float().square().mean().sqrt().detach()),
        "finite": bool(torch.isfinite(y).all() and torch.isfinite(gradient).all()),
        "nonzero_gradient_fraction": float((gradient.abs() > 0.0).float().mean()),
        "gradient_norm": float(gradient.norm()),
        "finite_difference": finite_difference,
        "analytic_gradient": analytic,
        "finite_difference_index": list(index),
        "finite_difference_absolute_error": abs(finite_difference - analytic),
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def identity_error(kind: str, config: QH012Config, x: torch.Tensor) -> float:
    module = FourQubitBrickworkTransform(config, core_kind=kind).cuda()
    with torch.no_grad():
        module.theta.zero_()
        return float((module(x).float() - x.float()).abs().max())


def exact_dequantization(config: QH012Config, x: torch.Tensor) -> dict:
    torch.manual_seed(109)
    quantum = FourQubitBrickworkTransform(config, core_kind="quantum").cuda()
    dequantized = FourQubitBrickworkTransform(config, core_kind="dequantized").cuda()
    with torch.no_grad():
        quantum.theta.normal_(mean=0.0, std=0.05)
        dequantized.theta.copy_(quantum.theta)
    q_output = quantum(x)
    d_output = dequantized(x)
    weights = torch.linspace(0.5, 1.5, q_output.numel(), device=x.device).reshape_as(q_output)
    objective(q_output, weights).backward()
    objective(d_output, weights).backward()
    return {
        "max_absolute_output_error": float((q_output - d_output).abs().max().detach()),
        "max_absolute_gradient_error": float(
            (quantum.theta.grad - dequantized.theta.grad).abs().max().detach()
        ),
    }


def product_state_entanglement(config: QH012Config) -> dict:
    """Entropy across the 01|23 cut after the full gate schedule."""
    values = {}
    for kind in ("quantum", "no_entanglement"):
        torch.manual_seed(2027)
        module = FourQubitBrickworkTransform(config, core_kind=kind).cuda()
        with torch.no_grad():
            module.theta.normal_(mean=0.0, std=0.2)
            state = torch.zeros(
                1, module.n_registers, 2, 2, 2, 2,
                dtype=torch.complex64, device="cuda"
            )
            state[:, :, 0, 0, 0, 0] = 1.0
            gates = (
                module._cayley_gates().to(torch.complex64)
                if kind == "quantum"
                else module._separable_gates()
            )
            output = module._run_schedule(state, gates)[0, 0].reshape(4, 4)
            probabilities = torch.linalg.svdvals(output).square()
            probabilities = probabilities[probabilities > 1.0e-12]
            probabilities = probabilities / probabilities.sum()
            entropy = -(probabilities * torch.log2(probabilities)).sum()
            values[kind] = float(entropy)
    return values


def cpu_rejection(config: QH012Config) -> bool:
    module = FourQubitBrickworkTransform(config, core_kind="quantum")
    try:
        module(torch.zeros(1, config.hidden_size))
    except RuntimeError as error:
        return "GPU-only" in str(error)
    return False


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH-012 validation is GPU-only")
    config = QH012Config(hidden_size=64, init_std=1.0e-3)
    torch.manual_seed(7)
    x = torch.randn(12, 5, config.hidden_size, device="cuda", dtype=torch.float32)
    kinds = ("quantum", "dequantized", "classical", "no_entanglement")
    results = [measure(kind, config, x) for kind in kinds]
    counts = {result["parameter_count"] for result in results}
    identity = {kind: identity_error(kind, config, x) for kind in kinds}
    dequantization = exact_dequantization(config, x)
    entanglement = product_state_entanglement(config)
    rejected_cpu = cpu_rejection(config)
    passed = (
        len(counts) == 1
        and all(result["finite"] for result in results)
        and max(identity.values()) < 1.0e-5
        and dequantization["max_absolute_output_error"] < 1.0e-5
        and dequantization["max_absolute_gradient_error"] < 1.0e-5
        and entanglement["no_entanglement"] < 1.0e-5
        and entanglement["quantum"] > entanglement["no_entanglement"] + 1.0e-3
        and rejected_cpu
    )
    payload = {
        "status": "pass" if passed else "fail",
        "device": torch.cuda.get_device_name(),
        "config": config.__dict__,
        "equal_parameter_count": len(counts) == 1,
        "identity_max_absolute_error": identity,
        "quantum_vs_exact_dequantized": dequantization,
        "product_state_entropy_bits_01_cut": entanglement,
        "cpu_path_rejected": rejected_cpu,
        "results": results,
    }
    output = PROJECT_DIR / "artifacts/qh012-gpu-validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
