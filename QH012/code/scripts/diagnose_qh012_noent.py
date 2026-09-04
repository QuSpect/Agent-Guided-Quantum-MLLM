#!/usr/bin/env python3
"""Diagnose separability of the QH-012 no-entanglement control."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.brickwork_four_qubit import QH012Config, FourQubitBrickworkTransform


def entropy(state: torch.Tensor) -> float:
    matrix = state.reshape(4, 4)
    probabilities = torch.linalg.svdvals(matrix).square()
    probabilities = probabilities[probabilities > 1.0e-12]
    probabilities = probabilities / probabilities.sum()
    return float(-(probabilities * torch.log2(probabilities)).sum())


def main() -> None:
    config = QH012Config(hidden_size=16)
    torch.manual_seed(2027)
    module = FourQubitBrickworkTransform(config, core_kind="no_entanglement").cuda()
    with torch.no_grad():
        module.theta.normal_(mean=0.0, std=0.2)
        gates = module._separable_gates()
        identity = torch.eye(4, dtype=torch.complex64, device="cuda")
        unitary_errors = [
            float((gate.mH @ gate - identity).abs().max()) for gate in gates[0]
        ]
        schmidt_second = []
        for gate in gates[0]:
            reshaped = gate.reshape(2, 2, 2, 2).permute(0, 2, 1, 3).reshape(4, 4)
            schmidt_second.append(float(torch.linalg.svdvals(reshaped)[1]))
        state = torch.zeros(1, 1, 2, 2, 2, 2, dtype=torch.complex64, device="cuda")
        state[:, :, 0, 0, 0, 0] = 1.0
        entropies = [entropy(state[0, 0])]
        for gate_index, wires in enumerate(module.schedule):
            state = module._apply_two_qubit_gate(state, gates[:, gate_index], wires)
            entropies.append(entropy(state[0, 0]))
        print(json.dumps({
            "unitary_max_errors": unitary_errors,
            "operator_schmidt_second_singular_values": schmidt_second,
            "entropy_after_each_gate": entropies,
            "schedule": module.schedule,
        }, indent=2))


if __name__ == "__main__":
    main()
