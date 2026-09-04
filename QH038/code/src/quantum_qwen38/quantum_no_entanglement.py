"""Matched no-entanglement causal control for QH-001b."""

from __future__ import annotations

import torch
from torch import nn

from .quantum_residual import NativeStatevectorVQC, QuantumResidualMergerWrapper
from .quantum_residual_bf16 import BF16QuantumResidualAdapter, QH001BF16Config


class NoEntanglementVQC(NativeStatevectorVQC):
    """Native exact VQC with the ring-CNOT block removed and nothing else changed."""

    def forward(self, encoded_angles: torch.Tensor) -> torch.Tensor:
        if not encoded_angles.is_cuda:
            raise RuntimeError("QH-001b simulation is GPU-only")
        if encoded_angles.ndim != 2 or encoded_angles.shape[-1] != self.n_qubits:
            raise ValueError(
                f"expected [batch, {self.n_qubits}] angles, got {tuple(encoded_angles.shape)}"
            )
        angles = encoded_angles.float()
        state = torch.zeros(
            angles.shape[0],
            1 << self.n_qubits,
            dtype=torch.complex64,
            device=angles.device,
        )
        state[:, 0] = 1.0 + 0.0j
        for wire in range(self.n_qubits):
            state = self._apply_ry(state, angles[:, wire], wire)
        for layer in range(self.depth):
            for wire in range(self.n_qubits):
                state = self._apply_ry(state, self.theta[layer, wire, 0], wire)
                state = self._apply_rz(state, self.theta[layer, wire, 1], wire)
            # Causal intervention: the parent's ring-CNOT loop is deleted.
        return state.abs().square() @ self.z_signs


class NoEntanglementQuantumResidualAdapter(BF16QuantumResidualAdapter):
    """QH-001b with a parameter-matched, non-entangling circuit."""

    def __init__(self, config: QH001BF16Config | None = None) -> None:
        super().__init__(config)
        self.vqc = NoEntanglementVQC(self.config.n_qubits, self.config.depth)


def freeze_and_inject_no_entanglement(
    model: nn.Module,
    config: QH001BF16Config | None = None,
) -> dict:
    """Freeze Qwen3.8 and replace its visual merger with this causal control."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        visual = model.model.visual
        old_merger = visual.merger
    except AttributeError as error:
        raise AttributeError("expected model.model.visual.merger") from error
    reference = next(old_merger.parameters())
    adapter = NoEntanglementQuantumResidualAdapter(config).to(device=reference.device)
    adapter.down.to(dtype=reference.dtype)
    adapter.up.to(dtype=reference.dtype)
    adapter.vqc.to(dtype=torch.float32)
    visual.merger = QuantumResidualMergerWrapper(old_merger, adapter)
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    return {
        "candidate": "QH-001b-no-ent",
        "intervention": "delete ring-CNOT operations only",
        "injection_path": "model.model.visual.merger",
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
