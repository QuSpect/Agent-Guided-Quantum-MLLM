#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.quantum_residual import QH001Config, QuantumResidualAdapter


def main() -> None:
    torch.manual_seed(20260828)
    device = torch.device("cuda:0")
    config = QH001Config(hidden_size=5120, n_qubits=8, depth=2, gamma_init=1.0e-3)
    module = QuantumResidualAdapter(config).to(device=device)
    module.down.to(dtype=torch.bfloat16)
    module.up.to(dtype=torch.bfloat16)
    module.vqc.to(dtype=torch.float32)
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
    input_rms = hidden.detach().float().square().mean().sqrt()
    delta_rms = delta.square().mean().sqrt()
    result = {
        "status": "ok",
        "device": torch.cuda.get_device_name(0),
        "projection_dtype": str(module.down.weight.dtype),
        "quantum_parameter_dtype": str(module.vqc.theta.dtype),
        "statevector_dtype": "torch.complex64",
        "batch_visual_tokens": hidden.shape[0],
        "hidden_size": hidden.shape[1],
        "elapsed_forward_backward_ms": elapsed_ms,
        "loss": float(loss.detach()),
        "delta_rms": float(delta_rms),
        "relative_delta_rms": float(delta_rms / input_rms),
        "hidden_grad_norm": float(hidden.grad.float().norm()),
        "down_grad_norm": float(module.down.weight.grad.float().norm()),
        "vqc_grad_norm": float(module.vqc.theta.grad.float().norm()),
        "up_grad_norm": float(module.up.weight.grad.float().norm()),
        "gamma_grad": float(module.gamma.grad.float()),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(0) / (1024**2),
        "audit": module.audit(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
