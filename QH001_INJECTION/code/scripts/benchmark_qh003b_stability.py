#!/usr/bin/env python3
import json
import statistics
import sys
import time
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.segmented_anchor_stable import StableSegmentedQuantumAnchorAdapter


def main() -> None:
    torch.manual_seed(20260828)
    module = StableSegmentedQuantumAnchorAdapter().to(device="cuda:0")
    module.down.to(dtype=torch.bfloat16)
    module.up.to(dtype=torch.bfloat16)
    module.vqc.to(dtype=torch.float32)
    results = []
    for token_count in (256, 1024, 4096):
        torch.manual_seed(20260828 + token_count)
        hidden = torch.randn(token_count, 5120, device="cuda:0", dtype=torch.bfloat16, requires_grad=True)
        split_sizes = [token_count // 4] * 4
        for _ in range(2):
            module(hidden, split_sizes).float().square().mean().backward()
            module.zero_grad(set_to_none=True)
            hidden.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)
        times = []
        output = None
        for _ in range(5):
            started = time.perf_counter()
            output = module(hidden, split_sizes)
            output.float().square().mean().backward()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - started) * 1000.0)
            module.zero_grad(set_to_none=True)
            hidden.grad = None
        delta = output.detach().float() - hidden.detach().float()
        relative_delta = delta.square().mean().sqrt() / hidden.detach().float().square().mean().sqrt()
        results.append(
            {
                "visual_tokens": token_count,
                "images": 4,
                "anchors": module.last_num_anchors,
                "forward_backward_median_ms": statistics.median(times),
                "peak_allocated_mb": torch.cuda.max_memory_allocated(0) / (1024**2),
                "relative_delta_rms": float(relative_delta),
                "changed_fraction": float((output.detach() != hidden.detach()).float().mean()),
                "vqc_grad_norm": float(module.vqc.theta.grad.float().norm()) if module.vqc.theta.grad is not None else None,
            }
        )
    print(json.dumps({"status": "ok", "candidate": "QH-003b", "results": results}, indent=2))


if __name__ == "__main__":
    main()

