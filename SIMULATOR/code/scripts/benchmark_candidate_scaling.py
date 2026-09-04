#!/usr/bin/env python3
import json
import statistics
import sys
import time
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.classical_control import ClassicalControlConfig, MatchedClassicalResidualAdapter
from quantum_qwen38.quantum_residual_bf16 import BF16QuantumResidualAdapter
from quantum_qwen38.segmented_anchor import QH003AConfig, SegmentedQuantumAnchorAdapter


def prepare(module: torch.nn.Module) -> torch.nn.Module:
    module = module.to(device="cuda:0")
    module.down.to(dtype=torch.bfloat16)
    module.up.to(dtype=torch.bfloat16)
    if hasattr(module, "vqc"):
        module.vqc.to(dtype=torch.float32)
    return module


def time_forward(module, hidden, split_sizes, repeats=7):
    times = []
    for _ in range(2):
        with torch.no_grad():
            module(hidden, split_sizes) if split_sizes else module(hidden)
    torch.cuda.synchronize()
    for _ in range(repeats):
        started = time.perf_counter()
        with torch.no_grad():
            module(hidden, split_sizes) if split_sizes else module(hidden)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(times)


def time_forward_backward(module, hidden, split_sizes, repeats=5):
    times = []
    for _ in range(2):
        output = module(hidden, split_sizes) if split_sizes else module(hidden)
        output.float().square().mean().backward()
        module.zero_grad(set_to_none=True)
        hidden.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    for _ in range(repeats):
        started = time.perf_counter()
        output = module(hidden, split_sizes) if split_sizes else module(hidden)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1000.0)
        module.zero_grad(set_to_none=True)
        hidden.grad = None
    return statistics.median(times), torch.cuda.max_memory_allocated(0) / (1024**2)


def evaluate(name, module, token_count, split_sizes):
    torch.manual_seed(20260828 + token_count)
    hidden = torch.randn(
        token_count, 5120, device="cuda:0", dtype=torch.bfloat16, requires_grad=True
    )
    with torch.no_grad():
        output = module(hidden, split_sizes) if split_sizes else module(hidden)
        delta = output.float() - hidden.float()
        relative_delta = delta.square().mean().sqrt() / hidden.float().square().mean().sqrt()
        changed_fraction = (output != hidden).float().mean()
    forward_ms = time_forward(module, hidden, split_sizes)
    step_ms, peak_mb = time_forward_backward(module, hidden, split_sizes)
    return {
        "candidate": name,
        "visual_tokens": token_count,
        "images": len(split_sizes) if split_sizes else 4,
        "forward_median_ms": forward_ms,
        "forward_backward_median_ms": step_ms,
        "peak_allocated_mb": peak_mb,
        "relative_delta_rms": float(relative_delta),
        "changed_fraction": float(changed_fraction),
    }


def main() -> None:
    torch.manual_seed(20260828)
    qh001b = prepare(BF16QuantumResidualAdapter())
    torch.manual_seed(20260828)
    qh003a = prepare(SegmentedQuantumAnchorAdapter(QH003AConfig(scale_init=0.5)))
    torch.manual_seed(20260828)
    cc001b = prepare(
        MatchedClassicalResidualAdapter(ClassicalControlConfig(up_init_std=0.006))
    )

    results = []
    for token_count in (256, 1024, 4096):
        splits = [token_count // 4] * 4
        results.append(evaluate("QH-001b", qh001b, token_count, None))
        results.append(evaluate("QH-003a-calibrated", qh003a, token_count, splits))
        results.append(evaluate("CC-001b-calibrated", cc001b, token_count, None))
    print(json.dumps({"status": "ok", "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

