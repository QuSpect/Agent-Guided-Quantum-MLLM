#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.segmented_anchor import SegmentedQuantumAnchorAdapter


def main() -> None:
    torch.manual_seed(20260828)
    device = torch.device("cuda:0")
    module = SegmentedQuantumAnchorAdapter().to(device=device)
    module.down.to(dtype=torch.bfloat16)
    module.up.to(dtype=torch.bfloat16)
    module.vqc.to(dtype=torch.float32)
    hidden = torch.randn(256, 5120, device=device, dtype=torch.bfloat16, requires_grad=True)
    split_sizes = [64, 128, 64]

    for _ in range(3):
        module(hidden, split_sizes).float().square().mean().backward()
        module.zero_grad(set_to_none=True)
        hidden.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    output = module(hidden, split_sizes)
    loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    delta = output.detach().float() - hidden.detach().float()
    relative_delta = delta.square().mean().sqrt() / hidden.detach().float().square().mean().sqrt()
    result = {
        "status": "ok",
        "candidate": "QH-003a",
        "device": torch.cuda.get_device_name(0),
        "split_sizes": split_sizes,
        "num_anchors": module.last_num_anchors,
        "elapsed_forward_backward_ms": elapsed_ms,
        "relative_delta_rms": float(relative_delta),
        "changed_fraction": float((output.detach() != hidden.detach()).float().mean()),
        "vqc_grad_norm": float(module.vqc.theta.grad.float().norm()),
        "down_grad_norm": float(module.down.weight.grad.float().norm()),
        "up_grad_norm": float(module.up.weight.grad.float().norm()),
        "raw_scale_grad": float(module.raw_scale.grad.float()),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(0) / (1024**2),
        "audit": module.audit(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

