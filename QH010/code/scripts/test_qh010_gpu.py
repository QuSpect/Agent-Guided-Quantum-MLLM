#!/usr/bin/env python3
"""Static GPU validation for QH-010 and its equal-parameter controls."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.cayley_two_qubit import QH010Config, TwoQubitBlockTransform


def measure(kind: str, config: QH010Config, x: torch.Tensor) -> dict:
    torch.manual_seed(20260829)
    module = TwoQubitBlockTransform(config, core_kind=kind).cuda()
    parameter_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    y = module(x)
    weights = torch.linspace(0.25, 1.75, y.numel(), device=y.device).reshape_as(y)
    loss = (weights * y.float().square()).mean() + 0.01 * y.float().mean()
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    gradient = module.theta.grad.detach()
    flat_index = int(gradient.abs().argmax())
    index = (flat_index // gradient.shape[1], flat_index % gradient.shape[1])
    analytic = float(gradient[index])
    epsilon = 1.0e-2
    with torch.no_grad():
        original = float(module.theta[index])
        module.theta[index] = original + epsilon
        plus_y = module(x).float()
        plus = float(((weights * plus_y.square()).mean() + 0.01 * plus_y.mean()).detach())
        module.theta[index] = original - epsilon
        minus_y = module(x).float()
        minus = float(((weights * minus_y.square()).mean() + 0.01 * minus_y.mean()).detach())
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


def identity_error(kind: str, config: QH010Config, x: torch.Tensor) -> float:
    module = TwoQubitBlockTransform(config, core_kind=kind).cuda()
    with torch.no_grad():
        module.theta.zero_()
        return float((module(x).float() - x.float()).abs().max())


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH-010 validation is GPU-only")
    config = QH010Config(hidden_size=32, init_std=1.0e-3)
    torch.manual_seed(7)
    # Float32 makes finite differences informative; real-model integration separately uses BF16.
    x = torch.randn(12, 5, config.hidden_size, device="cuda", dtype=torch.float32)
    kinds = ("quantum", "dequantized", "classical", "no_entanglement")
    results = [measure(kind, config, x) for kind in kinds]
    counts = {result["parameter_count"] for result in results}
    identity = {kind: identity_error(kind, config, x) for kind in kinds}
    quantum = TwoQubitBlockTransform(config, core_kind="quantum").cuda()
    dequantized = TwoQubitBlockTransform(config, core_kind="dequantized").cuda()
    with torch.no_grad():
        dequantized.theta.copy_(quantum.theta)
        exact_dequantization_error = float((quantum(x) - dequantized(x)).abs().max())
    payload = {
        "status": "pass" if len(counts) == 1 and all(result["finite"] for result in results) else "fail",
        "device": torch.cuda.get_device_name(),
        "config": config.__dict__,
        "equal_parameter_count": len(counts) == 1,
        "identity_max_absolute_error": identity,
        "quantum_vs_exact_dequantized_max_absolute_error": exact_dequantization_error,
        "results": results,
    }
    output = PROJECT_DIR / "artifacts/qh010-gpu-validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
