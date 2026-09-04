#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.quantum_residual import QH001Config, QuantumResidualAdapter


def evaluate(hidden: torch.Tensor, gamma: float, up_std: float) -> dict:
    torch.manual_seed(20260828)
    config = QH001Config(hidden_size=5120, n_qubits=8, depth=2, gamma_init=gamma)
    module = QuantumResidualAdapter(config).to(device=hidden.device)
    module.down.to(dtype=torch.bfloat16)
    module.up.to(dtype=torch.bfloat16)
    module.vqc.to(dtype=torch.float32)
    torch.nn.init.normal_(module.up.weight, mean=0.0, std=up_std)

    output = module(hidden)
    delta = output.detach().float() - hidden.detach().float()
    input_rms = hidden.detach().float().square().mean().sqrt()
    delta_rms = delta.square().mean().sqrt()
    changed_fraction = (output.detach() != hidden.detach()).float().mean()

    loss = output.float().square().mean()
    loss.backward()
    return {
        "gamma": gamma,
        "up_init_std": up_std,
        "relative_delta_rms": float(delta_rms / input_rms),
        "changed_fraction": float(changed_fraction),
        "down_grad_norm": float(module.down.weight.grad.float().norm()),
        "vqc_grad_norm": float(module.vqc.theta.grad.float().norm()),
        "up_grad_norm": float(module.up.weight.grad.float().norm()),
        "gamma_grad": float(module.gamma.grad.float()),
    }


def main() -> None:
    torch.manual_seed(20260828)
    device = torch.device("cuda:0")
    hidden = torch.randn(256, 5120, device=device, dtype=torch.bfloat16)
    results = []
    for gamma in (0.01, 0.03, 0.1, 0.3):
        for up_std in (0.001, 0.003, 0.01, 0.03):
            results.append(evaluate(hidden, gamma, up_std))
    print(json.dumps({"status": "ok", "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
