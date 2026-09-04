"""QH038: QH037 full-FFN replacement with shifted-ring inter-block mixing.

The parameter count, 5120->64->5120 projections, target layer, and two local
four-qubit stages stay fixed.  Between stages, each lane is cyclically shifted
across the sixteen blocks before regrouping.  The second exact-statevector VQC
therefore receives features produced by four different first-stage blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping

import torch
from torch import Tensor

from .qh037_grouped_qvaf_ffn_replacement import (
    DequantizedGroupedQVAFCore,
    ExactGroupedQVAFCore,
    GroupedQVAFReplacement,
    QH037Config,
    _Core,
    _require_cuda,
    qh037_parameter_audit,
    unique_trainable_parameters,
)


@dataclass(frozen=True)
class QH038Config(QH037Config):
    init_seed: int = 20260866

    def validate(self):
        super().validate()
        if self.depth != 2:
            raise ValueError("QH038 is a two-stage, parameter-matched QH037 mutation")


def shifted_ring_reblock(v: Tensor) -> Tensor:
    """Lane q of new group g comes from old group (g+q) mod groups."""
    return torch.stack([torch.roll(v[..., q], shifts=-q, dims=-1) for q in range(v.shape[-1])], dim=-1)


class ExactShiftedRingGroupedQVAFCore(ExactGroupedQVAFCore):
    def _stage(self, v: Tensor, theta: Tensor) -> Tensor:
        n = v.shape[0]
        encoded = 2.0 * torch.atan(v)
        state = torch.zeros(n, self.config.groups, self.state_dimension, device=v.device, dtype=torch.float32)
        state[..., 0] = 1
        state = state.reshape(n * self.config.groups, self.state_dimension)
        for q in range(self.config.num_qubits):
            state = self._rotate(state, getattr(self, f"z_{q}"), getattr(self, f"o_{q}"), encoded[..., q].reshape(-1))
            angle = theta[:, q, 0].unsqueeze(0).expand(n, -1).reshape(-1)
            state = self._rotate(state, getattr(self, f"z_{q}"), getattr(self, f"o_{q}"), angle)
        if self.use_entanglers:
            for q in range(self.config.num_qubits):
                angle = theta[:, q, 1].unsqueeze(0).expand(n, -1).reshape(-1)
                state = self._rotate(state, getattr(self, f"cz_{q}"), getattr(self, f"co_{q}"), angle)
        probs = state.square()
        return torch.matmul(probs, self.z_signs.transpose(0, 1)).reshape(n, self.config.groups, self.config.num_qubits)

    def forward(self, latent: Tensor):
        _require_cuda(latent)
        shape, v, _ = self._inputs(latent)
        v = self._stage(v, self.theta[0])
        v = shifted_ring_reblock(v)
        v = self._stage(v, self.theta[1])
        self.circuit_calls.add_(2)
        return v.reshape(shape).to(latent.dtype)


class DequantizedShiftedRingGroupedQVAFCore(DequantizedGroupedQVAFCore):
    def _stage(self, v: Tensor, theta: Tensor) -> Tensor:
        n = v.shape[0]
        encoded = 2.0 * torch.atan(v)
        state = torch.zeros(n * self.config.groups, *([2] * self.config.num_qubits), device=v.device, dtype=torch.float32)
        state[(slice(None),) + (0,) * self.config.num_qubits] = 1
        for q in range(self.config.num_qubits):
            state = self._single(state, q, encoded[..., q].reshape(-1))
            angle = theta[:, q, 0].unsqueeze(0).expand(n, -1).reshape(-1)
            state = self._single(state, q, angle)
        if self.use_entanglers:
            for q in range(self.config.num_qubits):
                angle = theta[:, q, 1].unsqueeze(0).expand(n, -1).reshape(-1)
                state = self._controlled(state, q, (q + 1) % self.config.num_qubits, angle)
        probs = state.reshape(n * self.config.groups, -1).square()
        basis = torch.arange(1 << self.config.num_qubits, device=v.device)
        signs = torch.stack([torch.where(basis.bitwise_and(1 << q) == 0, 1.0, -1.0) for q in range(self.config.num_qubits)])
        return torch.matmul(probs, signs.transpose(0, 1)).reshape(n, self.config.groups, self.config.num_qubits)

    def forward(self, latent: Tensor):
        _require_cuda(latent)
        shape, v, _ = self._inputs(latent)
        v = self._stage(v, self.theta[0])
        v = shifted_ring_reblock(v)
        v = self._stage(v, self.theta[1])
        self.circuit_calls.add_(2)
        return v.reshape(shape).to(latent.dtype)


class EqualParameterShiftedRingClassicalCore(_Core):
    @staticmethod
    def _stage(v: Tensor, theta: Tensor) -> Tensor:
        a = torch.tanh(theta[..., 0]).unsqueeze(0)
        b = torch.tanh(theta[..., 1]).unsqueeze(0)
        return torch.tanh(v + a * torch.sin(v) + b * torch.sin(v * torch.roll(v, 1, dims=-1)))

    def forward(self, latent: Tensor):
        _require_cuda(latent)
        shape, v, _ = self._inputs(latent)
        v = self._stage(v, self.theta[0])
        v = shifted_ring_reblock(v)
        v = self._stage(v, self.theta[1])
        return v.reshape(shape).to(latent.dtype)


def qh038_parameter_audit(config: QH038Config) -> MutableMapping[str, int]:
    return qh037_parameter_audit(config)


def install_qh038(model, config: QH038Config = QH038Config(), branch: str = "quantum"):
    config.validate()
    model.requires_grad_(False)
    layer = model.model.language_model.layers[config.target_layer]
    original = layer.mlp
    device = next(original.parameters()).device
    dtype = next(original.parameters()).dtype
    classes = {
        "quantum": ExactShiftedRingGroupedQVAFCore,
        "dq": DequantizedShiftedRingGroupedQVAFCore,
        "classical": EqualParameterShiftedRingClassicalCore,
        "no_ent": lambda cfg: ExactShiftedRingGroupedQVAFCore(cfg, use_entanglers=False),
    }
    if branch not in classes:
        raise ValueError(f"unknown QH038 branch {branch}")
    base = sum(p.numel() for p in model.parameters())
    replacement = GroupedQVAFReplacement(config, classes[branch](config), device, dtype)
    layer.mlp = replacement
    deployed = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    audit = qh038_parameter_audit(config)
    if base - deployed != audit["net_parameter_reduction"] or trainable != audit["trainable_parameters"]:
        raise AssertionError("QH038 parameter audit mismatch")
    return replacement, original, {
        **audit,
        "base_model_parameters": base,
        "deployed_model_parameters": deployed,
        "net_model_parameter_reduction": audit["net_parameter_reduction"],
        "trainable_parameter_count": trainable,
        "branch": branch,
        "target_layer": config.target_layer,
        "latent_size": config.latent_size,
        "groups": config.groups,
        "group_size": config.group_size,
        "num_qubits_per_group": config.num_qubits,
        "stages": config.depth,
        "inter_stage_mixing": "lane-q cyclic shift by -q groups then regroup",
    }


__all__ = [
    "QH038Config",
    "ExactShiftedRingGroupedQVAFCore",
    "DequantizedShiftedRingGroupedQVAFCore",
    "EqualParameterShiftedRingClassicalCore",
    "GroupedQVAFReplacement",
    "install_qh038",
    "qh038_parameter_audit",
    "shifted_ring_reblock",
    "unique_trainable_parameters",
]
