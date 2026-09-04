#!/usr/bin/env python3
import json
import statistics
import sys
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
        split_sizes = [token_count // 4] * 4
        relative_deltas = []
        changed_fractions = []
        for seed_offset in range(20):
            torch.manual_seed(20260828 + seed_offset)
            hidden = torch.randn(token_count, 5120, device="cuda:0", dtype=torch.bfloat16)
            with torch.no_grad():
                output = module(hidden, split_sizes)
                delta = output.float() - hidden.float()
                relative_delta = delta.square().mean().sqrt() / hidden.float().square().mean().sqrt()
                relative_deltas.append(float(relative_delta))
                changed_fractions.append(float((output != hidden).float().mean()))
        results.append(
            {
                "visual_tokens": token_count,
                "images": 4,
                "anchors_per_run": module.last_num_anchors,
                "seeds": 20,
                "relative_delta_rms_mean": statistics.mean(relative_deltas),
                "relative_delta_rms_std": statistics.stdev(relative_deltas),
                "relative_delta_rms_min": min(relative_deltas),
                "relative_delta_rms_max": max(relative_deltas),
                "changed_fraction_mean": statistics.mean(changed_fractions),
                "changed_fraction_std": statistics.stdev(changed_fractions),
            }
        )

    means = [item["relative_delta_rms_mean"] for item in results]
    ratio = max(means) / min(means)

    torch.manual_seed(20260828)
    hidden = torch.randn(1024, 5120, device="cuda:0", dtype=torch.bfloat16, requires_grad=True)
    output = module(hidden, [256, 256, 256, 256])
    output.float().square().mean().backward()
    gradient_check = {
        "down_grad_norm": float(module.down.weight.grad.float().norm()),
        "vqc_grad_norm": float(module.vqc.theta.grad.float().norm()),
        "up_grad_norm": float(module.up.weight.grad.float().norm()),
        "raw_scale_grad": float(module.raw_scale.grad.float()),
    }
    print(
        json.dumps(
            {
                "status": "ok",
                "candidate": "QH-003b",
                "mean_max_min_ratio": ratio,
                "passes_ratio_1_25": ratio < 1.25,
                "results": results,
                "gradient_check": gradient_check,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
