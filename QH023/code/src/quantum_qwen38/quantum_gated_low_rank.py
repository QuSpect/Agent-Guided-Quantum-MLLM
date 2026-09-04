"""QH-008: a quantum-gated rank-64 residual with matched controls."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .quantum_no_entanglement import NoEntanglementVQC
from .quantum_residual import NativeStatevectorVQC, QuantumResidualMergerWrapper


@dataclass(frozen=True)
class QH008Config:
    hidden_size: int = 5120
    rank: int = 64
    n_qubits: int = 8
    depth: int = 2
    scale_init: float = 0.1
    scale_max: float = 1.0
    gate_strength: float = 0.5
    local_up_init_std: float = 0.003
    gate_up_init_std: float = 0.02


class GPUStatevectorVQC(NativeStatevectorVQC):
    def forward(self, encoded_angles: torch.Tensor) -> torch.Tensor:
        if not encoded_angles.is_cuda:
            raise RuntimeError("QH-008 quantum simulation is GPU-only")
        return super().forward(encoded_angles)


class MatchedClassicalGateCore(nn.Module):
    """Parameter-matched nonlinear core; contains no statevector operations."""

    def __init__(self, n_features: int, depth: int) -> None:
        super().__init__()
        self.affine = nn.Parameter(torch.empty(depth, n_features, 2, dtype=torch.float32))
        nn.init.uniform_(self.affine, -0.05, 0.05)

    def forward(self, encoded_angles: torch.Tensor) -> torch.Tensor:
        features = torch.cos(encoded_angles.float())
        for layer in range(self.affine.shape[0]):
            gain = 1.0 + self.affine[layer, :, 0]
            bias = self.affine[layer, :, 1]
            features = torch.tanh(features * gain + bias)
        return features


class QuantumGatedLowRankAdapter(nn.Module):
    """Tokenwise low-rank features multiplicatively gated by a small VQC."""

    def __init__(
        self,
        config: QH008Config | None = None,
        core_kind: str = "quantum",
    ) -> None:
        super().__init__()
        self.config = config or QH008Config()
        self.core_kind = core_kind
        if not 0.0 < self.config.scale_init < self.config.scale_max:
            raise ValueError("scale_init must lie strictly between zero and scale_max")
        self.local_down = nn.Linear(self.config.hidden_size, self.config.rank, bias=True)
        self.quantum_down = nn.Linear(self.config.hidden_size, self.config.n_qubits, bias=True)
        if core_kind == "quantum":
            self.core = GPUStatevectorVQC(self.config.n_qubits, self.config.depth)
        elif core_kind == "no_entanglement":
            self.core = NoEntanglementVQC(self.config.n_qubits, self.config.depth)
        elif core_kind == "classical":
            self.core = MatchedClassicalGateCore(self.config.n_qubits, self.config.depth)
        else:
            raise ValueError(f"unsupported core_kind: {core_kind}")
        self.gate_up = nn.Linear(self.config.n_qubits, self.config.rank, bias=False)
        self.local_up = nn.Linear(self.config.rank, self.config.hidden_size, bias=False)
        probability = self.config.scale_init / self.config.scale_max
        self.raw_scale = nn.Parameter(
            torch.tensor(math.log(probability / (1.0 - probability)), dtype=torch.float32)
        )
        nn.init.xavier_uniform_(self.local_down.weight)
        nn.init.zeros_(self.local_down.bias)
        nn.init.xavier_uniform_(self.quantum_down.weight)
        nn.init.zeros_(self.quantum_down.bias)
        nn.init.normal_(self.gate_up.weight, mean=0.0, std=self.config.gate_up_init_std)
        nn.init.normal_(self.local_up.weight, mean=0.0, std=self.config.local_up_init_std)

    @property
    def scale(self) -> torch.Tensor:
        return self.config.scale_max * torch.sigmoid(self.raw_scale)

    def residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        if original_shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected hidden size {self.config.hidden_size}, got {original_shape[-1]}"
            )
        flat = hidden_states.reshape(-1, original_shape[-1])
        normalized = F.layer_norm(flat.float(), (self.config.hidden_size,)).to(flat.dtype)
        local_features = F.gelu(self.local_down(normalized).float())
        angles = torch.pi * torch.tanh(self.quantum_down(normalized).float())
        core_features = self.core(angles)
        gate = torch.tanh(self.gate_up(core_features.to(self.gate_up.weight.dtype)).float())
        modulated = local_features * (1.0 + self.config.gate_strength * gate)
        residual = self.local_up(modulated.to(self.local_up.weight.dtype)).to(flat.dtype)
        return (residual * self.scale.to(residual.dtype)).reshape(original_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.residual(hidden_states)

    def audit(self) -> dict:
        return {
            "candidate": "QH-008" if self.core_kind == "quantum" else f"QH-008-{self.core_kind}",
            "config": asdict(self.config),
            "core_kind": self.core_kind,
            "current_scale": float(self.scale.detach().cpu()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }


def freeze_and_inject_qh008(
    model: nn.Module,
    core_kind: str = "quantum",
    config: QH008Config | None = None,
) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.model.visual
    old_merger = visual.merger
    reference = next(old_merger.parameters())
    adapter = QuantumGatedLowRankAdapter(config, core_kind=core_kind).to(device=reference.device)
    adapter.local_down.to(dtype=reference.dtype)
    adapter.quantum_down.to(dtype=reference.dtype)
    adapter.gate_up.to(dtype=reference.dtype)
    adapter.local_up.to(dtype=reference.dtype)
    adapter.core.to(dtype=torch.float32)
    visual.merger = QuantumResidualMergerWrapper(old_merger, adapter)
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    candidate = {
        "quantum": "QH-008",
        "classical": "CC-008",
        "no_entanglement": "QH-008-no-ent",
    }[core_kind]
    return {
        "candidate": candidate,
        "parent": "QH-001b",
        "injection_path": "model.model.visual.merger",
        "quantum_compute_dtype": (
            "torch.float32/torch.complex64" if core_kind != "classical" else None
        ),
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
