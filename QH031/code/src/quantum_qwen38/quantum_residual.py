from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class QH001Config:
    hidden_size: int = 5120
    n_qubits: int = 8
    depth: int = 2
    gamma_init: float = 1.0e-3


class NativeStatevectorVQC(nn.Module):
    """Exact, differentiable batched statevector simulator implemented in PyTorch CUDA ops."""

    def __init__(self, n_qubits: int = 8, depth: int = 2) -> None:
        super().__init__()
        if n_qubits < 2:
            raise ValueError("n_qubits must be at least 2 for ring entanglement")
        self.n_qubits = n_qubits
        self.depth = depth
        self.theta = nn.Parameter(torch.empty(depth, n_qubits, 2, dtype=torch.float32))
        nn.init.uniform_(self.theta, -0.05, 0.05)

        basis = torch.arange(1 << n_qubits, dtype=torch.long)
        z_signs = []
        for wire in range(n_qubits):
            zero = basis[(basis & (1 << wire)) == 0]
            one = basis[(basis & (1 << wire)) != 0]
            self.register_buffer(f"wire_{wire}_zero", zero, persistent=False)
            self.register_buffer(f"wire_{wire}_one", one, persistent=False)
            z_signs.append(1.0 - 2.0 * ((basis >> wire) & 1).float())

        for control in range(n_qubits):
            target = (control + 1) % n_qubits
            permutation = basis.clone()
            active = (basis & (1 << control)) != 0
            permutation[active] ^= 1 << target
            self.register_buffer(f"cnot_{control}_{target}", permutation, persistent=False)
        self.register_buffer("z_signs", torch.stack(z_signs, dim=1), persistent=False)

    @staticmethod
    def _expand_angle(angle: torch.Tensor, target_ndim: int) -> torch.Tensor:
        while angle.ndim < target_ndim:
            angle = angle.unsqueeze(-1)
        return angle

    def _apply_ry(self, state: torch.Tensor, angle: torch.Tensor, wire: int) -> torch.Tensor:
        zero = getattr(self, f"wire_{wire}_zero")
        one = getattr(self, f"wire_{wire}_one")
        amp_zero = state.index_select(1, zero)
        amp_one = state.index_select(1, one)
        cosine = self._expand_angle(torch.cos(angle * 0.5), amp_zero.ndim)
        sine = self._expand_angle(torch.sin(angle * 0.5), amp_zero.ndim)
        output = state.clone()
        output[:, zero] = cosine * amp_zero - sine * amp_one
        output[:, one] = sine * amp_zero + cosine * amp_one
        return output

    def _apply_rz(self, state: torch.Tensor, angle: torch.Tensor, wire: int) -> torch.Tensor:
        zero = getattr(self, f"wire_{wire}_zero")
        one = getattr(self, f"wire_{wire}_one")
        amp_zero = state.index_select(1, zero)
        amp_one = state.index_select(1, one)
        angle = self._expand_angle(angle, amp_zero.ndim)
        phase_zero = torch.polar(torch.ones_like(angle), -0.5 * angle)
        phase_one = torch.polar(torch.ones_like(angle), 0.5 * angle)
        output = state.clone()
        output[:, zero] = phase_zero * amp_zero
        output[:, one] = phase_one * amp_one
        return output

    def forward(self, encoded_angles: torch.Tensor) -> torch.Tensor:
        if encoded_angles.ndim != 2 or encoded_angles.shape[-1] != self.n_qubits:
            raise ValueError(f"expected [batch, {self.n_qubits}] angles, got {tuple(encoded_angles.shape)}")
        angles = encoded_angles.float()
        batch = angles.shape[0]
        state = torch.zeros(
            batch,
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
                permutation = getattr(self, f"cnot_{control}_{target}")
                state = state.index_select(1, permutation)

        probabilities = state.abs().square()
        return probabilities @ self.z_signs


class QuantumResidualAdapter(nn.Module):
    def __init__(self, config: QH001Config | None = None) -> None:
        super().__init__()
        self.config = config or QH001Config()
        self.down = nn.Linear(self.config.hidden_size, self.config.n_qubits, bias=True)
        self.vqc = NativeStatevectorVQC(self.config.n_qubits, self.config.depth)
        self.up = nn.Linear(self.config.n_qubits, self.config.hidden_size, bias=False)
        self.gamma = nn.Parameter(torch.tensor(self.config.gamma_init, dtype=torch.float32))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, mean=0.0, std=1.0e-3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        if original_shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected hidden size {self.config.hidden_size}, got {original_shape[-1]}"
            )
        flat = hidden_states.reshape(-1, original_shape[-1])
        normalized = F.layer_norm(flat.float(), (self.config.hidden_size,)).to(flat.dtype)
        encoded_angles = torch.pi * torch.tanh(self.down(normalized).float())
        expectations = self.vqc(encoded_angles)
        residual = self.up(expectations.to(self.up.weight.dtype)).to(flat.dtype)
        scaled = residual * self.gamma.to(residual.dtype)
        return (flat + scaled).reshape(original_shape)

    def audit(self) -> dict:
        return {
            "config": asdict(self.config),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }


class QuantumResidualMergerWrapper(nn.Module):
    def __init__(self, frozen_merger: nn.Module, adapter: QuantumResidualAdapter) -> None:
        super().__init__()
        self.frozen_merger = frozen_merger
        self.adapter = adapter
        for parameter in self.frozen_merger.parameters():
            parameter.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        merged = self.frozen_merger(hidden_states)
        return self.adapter(merged)


def freeze_and_inject_qh001(model: nn.Module, config: QH001Config | None = None) -> dict:
    """Freeze Qwen3.8 and wrap its visual PatchMerger with the QH-001 quantum residual."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    try:
        visual = model.model.visual
        old_merger = visual.merger
    except AttributeError as error:
        raise AttributeError("expected model.model.visual.merger in Qwen3.8 multimodal model") from error

    adapter = QuantumResidualAdapter(config)
    reference = next(old_merger.parameters())
    adapter.to(device=reference.device, dtype=reference.dtype)
    visual.merger = QuantumResidualMergerWrapper(old_merger, adapter)

    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    return {
        "candidate": "QH-001",
        "injection_path": "model.model.visual.merger",
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }

