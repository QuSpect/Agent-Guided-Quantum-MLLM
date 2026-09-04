"""QH045: learn which QH044 entanglers to reopen from an exact no-ent start.

Both arms load the same frozen QH044 replacement.  At gate_raw == 0 they are
exactly the QH044 no-ent branch.  The quantum arm scales the saved controlled-RY
angles; the matched control uses the same 64 gates and saved interaction angles
in a classical neighbour-product residual.  Only gate_raw is trainable.
"""
from __future__ import annotations

from typing import MutableMapping

import torch
from torch import Tensor, nn

from .qh037_grouped_qvaf_ffn_replacement import (
    DequantizedGroupedQVAFCore,
    ExactGroupedQVAFCore,
    GroupedQVAFReplacement,
    QH037Config,
    _require_cuda,
)


class SelectiveEntanglementExactCore(ExactGroupedQVAFCore):
    def __init__(self, config: QH037Config):
        super().__init__(config, use_entanglers=True)
        self.gate_raw = nn.Parameter(torch.zeros(config.groups, config.num_qubits))

    def forward(self, latent: Tensor):
        _require_cuda(latent)
        shape, v, encoded = self._inputs(latent)
        n = v.shape[0]
        state = torch.zeros(
            n, self.config.groups, self.state_dimension, device=latent.device, dtype=torch.float32
        )
        state[..., 0] = 1
        state = state.reshape(n * self.config.groups, self.state_dimension)
        gates = torch.tanh(self.gate_raw)
        for layer in range(self.config.depth):
            for q in range(self.config.num_qubits):
                state = self._rotate(
                    state, getattr(self, f"z_{q}"), getattr(self, f"o_{q}"), encoded[..., q].reshape(-1)
                )
                angle = self.theta[layer, :, q, 0].unsqueeze(0).expand(n, -1).reshape(-1)
                state = self._rotate(state, getattr(self, f"z_{q}"), getattr(self, f"o_{q}"), angle)
            for q in range(self.config.num_qubits):
                angle = (self.theta[layer, :, q, 1] * gates[:, q]).unsqueeze(0).expand(n, -1).reshape(-1)
                state = self._rotate(state, getattr(self, f"cz_{q}"), getattr(self, f"co_{q}"), angle)
        probs = state.square()
        out = torch.matmul(probs, self.z_signs.transpose(0, 1)).reshape(n, self.config.latent_size)
        self.circuit_calls.add_(1)
        return out.reshape(shape).to(latent.dtype)


class SelectiveEntanglementDQCore(DequantizedGroupedQVAFCore):
    def __init__(self, config: QH037Config):
        super().__init__(config, use_entanglers=True)
        self.gate_raw = nn.Parameter(torch.zeros(config.groups, config.num_qubits))

    def forward(self, latent: Tensor):
        _require_cuda(latent)
        shape, v, encoded = self._inputs(latent)
        n = v.shape[0]
        state = torch.zeros(
            n * self.config.groups,
            *([2] * self.config.num_qubits),
            device=latent.device,
            dtype=torch.float32,
        )
        state[(slice(None),) + (0,) * self.config.num_qubits] = 1
        gates = torch.tanh(self.gate_raw)
        for layer in range(self.config.depth):
            for q in range(self.config.num_qubits):
                state = self._single(state, q, encoded[..., q].reshape(-1))
                angle = self.theta[layer, :, q, 0].unsqueeze(0).expand(n, -1).reshape(-1)
                state = self._single(state, q, angle)
            for q in range(self.config.num_qubits):
                angle = (self.theta[layer, :, q, 1] * gates[:, q]).unsqueeze(0).expand(n, -1).reshape(-1)
                state = self._controlled(state, q, (q + 1) % self.config.num_qubits, angle)
        probs = state.reshape(n * self.config.groups, -1).square()
        basis = torch.arange(1 << self.config.num_qubits, device=latent.device)
        signs = torch.stack(
            [torch.where(basis.bitwise_and(1 << q) == 0, 1.0, -1.0) for q in range(self.config.num_qubits)]
        )
        out = torch.matmul(probs, signs.transpose(0, 1)).reshape(n, self.config.latent_size)
        self.circuit_calls.add_(1)
        return out.reshape(shape).to(latent.dtype)


class SelectiveEntanglementClassicalCore(ExactGroupedQVAFCore):
    """Same frozen separable simulator scaffold plus a 64-parameter classical interaction."""

    def __init__(self, config: QH037Config):
        super().__init__(config, use_entanglers=False)
        self.gate_raw = nn.Parameter(torch.zeros(config.groups, config.num_qubits))

    def forward(self, latent: Tensor):
        shape = latent.shape
        values = super().forward(latent).reshape(-1, self.config.groups, self.config.num_qubits).float()
        gates = torch.tanh(self.gate_raw).unsqueeze(0)
        for layer in range(self.config.depth):
            saved = torch.tanh(self.theta[layer, :, :, 1]).unsqueeze(0)
            neighbour = torch.roll(values, shifts=-1, dims=-1)
            values = values + gates * saved * torch.sin(values * neighbour)
        return values.reshape(shape).to(latent.dtype)


def install_qh045(model, start_state: dict[str, Tensor], branch: str):
    config = QH037Config()
    model.requires_grad_(False)
    layer = model.model.language_model.layers[config.target_layer]
    original = layer.mlp
    device = next(original.parameters()).device
    dtype = next(original.parameters()).dtype
    classes = {
        "quantum": SelectiveEntanglementExactCore,
        "dq": SelectiveEntanglementDQCore,
        "classical": SelectiveEntanglementClassicalCore,
    }
    replacement = GroupedQVAFReplacement(config, classes[branch](config), device, dtype)
    missing, unexpected = replacement.load_state_dict(start_state, strict=False)
    if missing not in ([], ["core.gate_raw"]) or unexpected:
        raise RuntimeError({"missing": missing, "unexpected": unexpected})
    replacement.requires_grad_(False)
    replacement.core.gate_raw.requires_grad_(True)
    layer.mlp = replacement
    removed = 3 * config.hidden_size * config.intermediate_size
    deployed = sum(parameter.numel() for parameter in replacement.parameters())
    trainable = sum(parameter.numel() for parameter in replacement.parameters() if parameter.requires_grad)
    audit: MutableMapping[str, int | str] = {
        "branch": branch,
        "target_layer": config.target_layer,
        "original_parameters_removed": removed,
        "replacement_parameters": deployed,
        "net_model_parameter_reduction": removed - deployed,
        "trainable_parameters": trainable,
        "shared_start": "QH044 checkpoint with entanglement exactly gated off",
    }
    if deployed != 655_680 or trainable != 64:
        raise AssertionError(audit)
    return replacement, original, audit
