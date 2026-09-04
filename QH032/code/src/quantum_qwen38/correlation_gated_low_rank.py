"""QH-009: low-rank residual gated by local-Z and entanglement-sensitive ZZ readout."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .quantum_no_entanglement import NoEntanglementVQC
from .quantum_residual import NativeStatevectorVQC, QuantumResidualMergerWrapper


@dataclass(frozen=True)
class QH009Config:
    hidden_size: int = 5120
    rank: int = 64
    n_qubits: int = 8
    depth: int = 2
    scale_init: float = 0.1
    scale_max: float = 1.0
    gate_strength: float = 0.5
    local_up_init_std: float = 0.003
    gate_up_init_std: float = 0.02


def _with_ring_products(local_features: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [local_features, local_features * torch.roll(local_features, shifts=-1, dims=-1)],
        dim=-1,
    )


class CorrelatorStatevectorVQC(NativeStatevectorVQC):
    """Exact CUDA statevector circuit returning <Z_i> and <Z_i Z_(i+1)>."""

    def __init__(self, n_qubits: int = 8, depth: int = 2) -> None:
        super().__init__(n_qubits=n_qubits, depth=depth)
        zz_signs = [
            self.z_signs[:, wire] * self.z_signs[:, (wire + 1) % n_qubits]
            for wire in range(n_qubits)
        ]
        self.register_buffer("zz_ring_signs", torch.stack(zz_signs, dim=1), persistent=False)

    def forward(self, encoded_angles: torch.Tensor) -> torch.Tensor:
        if not encoded_angles.is_cuda:
            raise RuntimeError("QH-009 quantum simulation is GPU-only")
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
            for control in range(self.n_qubits):
                target = (control + 1) % self.n_qubits
                state = state.index_select(1, getattr(self, f"cnot_{control}_{target}"))
        probabilities = state.abs().square()
        return torch.cat(
            [probabilities @ self.z_signs, probabilities @ self.zz_ring_signs], dim=-1
        )


class CorrelatorNoEntanglementVQC(NoEntanglementVQC):
    """Product-state causal control with identical trainable tensors."""

    def forward(self, encoded_angles: torch.Tensor) -> torch.Tensor:
        if not encoded_angles.is_cuda:
            raise RuntimeError("QH-009 no-entanglement simulation is GPU-only")
        return _with_ring_products(super().forward(encoded_angles))


class CorrelatorClassicalGateCore(nn.Module):
    """Parameter-matched nonlinear control with explicit adjacent products."""

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
        return _with_ring_products(features)


class CorrelationGatedLowRankAdapter(nn.Module):
    """QH-008 mutation whose gate exposes pairwise quantum correlations."""

    def __init__(
        self,
        config: QH009Config | None = None,
        core_kind: str = "quantum",
    ) -> None:
        super().__init__()
        self.config = config or QH009Config()
        self.core_kind = core_kind
        if not 0.0 < self.config.scale_init < self.config.scale_max:
            raise ValueError("scale_init must lie strictly between zero and scale_max")
        self.local_down = nn.Linear(self.config.hidden_size, self.config.rank, bias=True)
        self.quantum_down = nn.Linear(self.config.hidden_size, self.config.n_qubits, bias=True)
        if core_kind == "quantum":
            self.core = CorrelatorStatevectorVQC(self.config.n_qubits, self.config.depth)
        elif core_kind == "no_entanglement":
            self.core = CorrelatorNoEntanglementVQC(
                self.config.n_qubits, self.config.depth
            )
        elif core_kind == "classical":
            self.core = CorrelatorClassicalGateCore(
                self.config.n_qubits, self.config.depth
            )
        else:
            raise ValueError(f"unsupported core_kind: {core_kind}")
        self.gate_up = nn.Linear(2 * self.config.n_qubits, self.config.rank, bias=False)
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
        observables = self.core(angles)
        gate = torch.tanh(self.gate_up(observables.to(self.gate_up.weight.dtype)).float())
        modulated = local_features * (1.0 + self.config.gate_strength * gate)
        residual = self.local_up(modulated.to(self.local_up.weight.dtype)).to(flat.dtype)
        return (residual * self.scale.to(residual.dtype)).reshape(original_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.residual(hidden_states)

    def audit(self) -> dict:
        candidate = {
            "quantum": "QH-009",
            "classical": "CC-009",
            "no_entanglement": "QH-009-no-ent",
        }[self.core_kind]
        return {
            "candidate": candidate,
            "config": asdict(self.config),
            "core_kind": self.core_kind,
            "readout": "8 local-Z + 8 adjacent-ZZ observables",
            "current_scale": float(self.scale.detach().cpu()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }


def freeze_and_inject_qh009(
    model: nn.Module,
    core_kind: str = "quantum",
    config: QH009Config | None = None,
) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.model.visual
    old_merger = visual.merger
    reference = next(old_merger.parameters())
    adapter = CorrelationGatedLowRankAdapter(config, core_kind=core_kind).to(
        device=reference.device
    )
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
    return {
        "candidate": {
            "quantum": "QH-009",
            "classical": "CC-009",
            "no_entanglement": "QH-009-no-ent",
        }[core_kind],
        "parent": "QH-008",
        "injection_path": "model.model.visual.merger",
        "quantum_compute_dtype": (
            "torch.float32/torch.complex64" if core_kind != "classical" else None
        ),
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
