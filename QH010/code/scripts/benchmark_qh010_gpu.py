#!/usr/bin/env python3
"""Exclusive warmup/measurement benchmark for QH-010 and controls."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.cayley_two_qubit import QH010Config, TwoQubitBlockTransform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-elements", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=50)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "artifacts/qh010-exclusive-benchmark.json")
    return parser.parse_args()


def run(kind: str, config: QH010Config, x_template: torch.Tensor, weights: torch.Tensor, args) -> dict:
    torch.manual_seed(20260829)
    module = TwoQubitBlockTransform(config, core_kind=kind).cuda()
    with torch.no_grad():
        module.theta.normal_(0.0, 0.02)

    def step() -> None:
        module.zero_grad(set_to_none=True)
        x = x_template.detach().clone().requires_grad_(True)
        output = module(x)
        loss = (output.float().square() * weights).mean()
        loss.backward()
        torch.cuda.synchronize()

    for _ in range(args.warmup):
        step()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(args.measured):
        started = time.perf_counter()
        step()
        timings.append(time.perf_counter() - started)
    return {
        "kind": kind,
        "median_ms": float(np.median(timings) * 1000.0),
        "p95_ms": float(np.quantile(timings, 0.95) * 1000.0),
        "mean_ms": float(np.mean(timings) * 1000.0),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "measured_iterations": args.measured,
        "warmup_iterations": args.warmup,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("QH-010 benchmark is GPU-only")
    config = QH010Config()
    torch.manual_seed(17)
    x = torch.randn(args.batch_elements, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    weights = torch.linspace(0.25, 1.75, x.numel(), device="cuda").reshape_as(x)
    results = [run(kind, config, x, weights, args) for kind in (
        "quantum", "dequantized", "classical", "no_entanglement"
    )]
    by_kind = {row["kind"]: row for row in results}
    payload = {
        "status": "ok",
        "device": torch.cuda.get_device_name(),
        "timing_context": "exclusive",
        "batch_elements": args.batch_elements,
        "hidden_size": config.hidden_size,
        "results": results,
        "ratios": {
            "quantum_vs_dequantized_median": (
                by_kind["quantum"]["median_ms"] / by_kind["dequantized"]["median_ms"]
            ),
            "quantum_vs_classical_median": (
                by_kind["quantum"]["median_ms"] / by_kind["classical"]["median_ms"]
            ),
        },
        "scope": "adapter-only forward+backward; not end-to-end TTFT or decode throughput",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
