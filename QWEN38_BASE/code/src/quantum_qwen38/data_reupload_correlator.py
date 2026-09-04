"""QH-007: data-reuploading VQC with local-Z and ring-ZZ readout."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .quantum_residual import NativeStatevectorVQC, QuantumResidualMergerWrapper


@dataclass(frozen=True)
class QH007Config:
    hidden_size: int = 5120
    n_qubits: int = 8
    depth: int = 2
    scale_init: float = 0.2
    scale_max: float = 1.0
    up_init_std: float = 0.007


class DataReuploadCorrelatorVQC(NativeStatevectorVQC):
    """Exact CUDA statevector VQC with re-uploading and correlation observables.

    Each layer re-encodes the token.  Even layers use a nearest-neighbour ring;
    odd layers use half-register long-range CNOTs.  Reading both Z_i and
    Z_i Z_{i+1} exposes correlations that a local readout can hide.
    """

    def __init__(self, n_qubits: int = 8, depth: int = 2) -> None:
        if n_qubits % 2:
            raise ValueError("QH-007 long-range topology requires an even qubit count")
        super().__init__(n_qubits=n_qubits, depth=depth)
        self.reupload_scale = nn.Parameter(
            torch.ones(depth, n_qubits, dtype=torch.float32)
        )
        basis = torch.arange(1 << n_qubits, dtype=torch.long)
        stride = n_qubits // 2
        for control in range(n_qubits):
            target = (control + stride) % n_qubits
            permutation = basis.clone()
            active = (basis & (1 << control)) != 0
            permutation[active] ^= 1 << target
            self.register_buffer(
                f"cnot_long_{control}_{target}", permutation, persistent=False
            )
        zz_signs = [
            self.z_signs[:, wire]
            * self.z_signs[:, (wire + 1) % n_qubits]
            for wire in range(n_qubits)
        ]
        self.register_buffer(
            "zz_ring_signs", torch.stack(zz_signs, dim=1), persistent=False
        )

    def forward(self, encoded_angles: torch.Tensor) -> torch.Tensor:
        if not encoded_angles.is_cuda:
            raise RuntimeError("QH-007 simulation is GPU-only")
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
        for layer in range(self.depth):
            for wire in range(self.n_qubits):
                data_angle = angles[:, wire] * self.reupload_scale[layer, wire]
                state = self._apply_ry(
                    state, data_angle + self.theta[layer, wire, 0], wire
                )
                state = self._apply_rz(state, self.theta[layer, wire, 1], wire)
            for control in range(self.n_qubits):
                if layer % 2 == 0:
                    target = (control + 1) % self.n_qubits
                    permutation = getattr(self, f"cnot_{control}_{target}")
                else:
                    target = (control + self.n_qubits // 2) % self.n_qubits
                    permutation = getattr(self, f"cnot_long_{control}_{target}")
                state = state.index_select(1, permutation)
        probabilities = state.abs().square()
        return torch.cat(
            [probabilities @ self.z_signs, probabilities @ self.zz_ring_signs], dim=-1
        )


class QH007ResidualAdapter(nn.Module):
    def __init__(self, config: QH007Config | None = None) -> None:
        super().__init__()
        self.config = config or QH007Config()
        if not 0.0 < self.config.scale_init < self.config.scale_max:
            raise ValueError("scale_init must lie strictly between zero and scale_max")
        self.down = nn.Linear(
            self.config.hidden_size, self.config.n_qubits, bias=True
        )
        self.vqc = DataReuploadCorrelatorVQC(
            self.config.n_qubits, self.config.depth
        )
        self.up = nn.Linear(
            2 * self.config.n_qubits, self.config.hidden_size, bias=False
        )
        probability = self.config.scale_init / self.config.scale_max
        self.raw_scale = nn.Parameter(
            torch.tensor(math.log(probability / (1.0 - probability)), dtype=torch.float32)
        )
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, mean=0.0, std=self.config.up_init_std)

    @property
    def scale(self) -> torch.Tensor:
        return self.config.scale_max * torch.sigmoid(self.raw_scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1])
        normalized = F.layer_norm(flat.float(), (self.config.hidden_size,)).to(
            flat.dtype
        )
        encoded = torch.pi * torch.tanh(self.down(normalized).float())
        observables = self.vqc(encoded)
        residual = self.up(observables.to(self.up.weight.dtype)).to(flat.dtype)
        return (flat + residual * self.scale.to(residual.dtype)).reshape(original_shape)

    def audit(self) -> dict:
        return {
            "candidate": "QH-007",
            "config": asdict(self.config),
            "topology": "nearest ring then half-register long-range CNOT",
            "readout": "8 local-Z + 8 nearest-neighbour ZZ",
            "current_scale": float(self.scale.detach().cpu()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }


def freeze_and_inject_qh007(model: nn.Module, config: QH007Config | None = None) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.model.visual
    old_merger = visual.merger
    reference = next(old_merger.parameters())
    adapter = QH007ResidualAdapter(config).to(device=reference.device)
    adapter.down.to(dtype=reference.dtype)
    adapter.up.to(dtype=reference.dtype)
    adapter.vqc.to(dtype=torch.float32)
    visual.merger = QuantumResidualMergerWrapper(old_merger, adapter)
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "candidate": "QH-007",
        "parent": "QH-001b",
        "injection_path": "model.model.visual.merger",
        "quantum_compute_dtype": "torch.float32/torch.complex64",
        "projection_dtype": str(reference.dtype),
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
