#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.classical_control import MatchedClassicalResidualAdapter


def main() -> None:
    torch.manual_seed(20260828)
    device = torch.device("cuda:0")
    module = MatchedClassicalResidualAdapter().to(device=device)
    module.down.to(dtype=torch.bfloat16)
    module.up.to(dtype=torch.bfloat16)
    module.core_affine.data = module.core_affine.data.float()
    hidden = torch.randn(256, 5120, device=device, dtype=torch.bfloat16, requires_grad=True)

    for _ in range(3):
        module(hidden).float().square().mean().backward()
        module.zero_grad(set_to_none=True)
        hidden.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    output = module(hidden)
    loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    delta = output.detach().float() - hidden.detach().float()
    relative_delta = delta.square().mean().sqrt() / hidden.detach().float().square().mean().sqrt()
    result = {
        "status": "ok",
        "candidate": "CC-001",
        "device": torch.cuda.get_device_name(0),
        "elapsed_forward_backward_ms": elapsed_ms,
        "relative_delta_rms": float(relative_delta),
        "changed_fraction": float((output.detach() != hidden.detach()).float().mean()),
        "core_grad_norm": float(module.core_affine.grad.float().norm()),
        "down_grad_norm": float(module.down.weight.grad.float().norm()),
        "up_grad_norm": float(module.up.weight.grad.float().norm()),
        "raw_scale_grad": float(module.raw_scale.grad.float()),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(0) / (1024**2),
        "audit": module.audit(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

