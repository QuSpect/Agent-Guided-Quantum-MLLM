"""QH-022: simulator-resident quantum parameter generator for one frozen Qwen layer.

The circuit generates the coefficients of a fixed low-rank update once per model
forward.  Unlike the QPA paper, inference deliberately keeps the exact CUDA
statevector call so the project's train-and-inference simulator constraint is met.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .quantum_residual import NativeStatevectorVQC


@dataclass(frozen=True)
class QH022Config:
    hidden_size: int = 5120
    rank: int = 12
    n_qubits: int = 6
    depth: int = 2
    layer_index: int = 7
    target_projection: str = "v_proj"
    gamma_init: float = 0.05
    basis_seed: int = 20260829


class GlobalQuantumCoefficientCore(NativeStatevectorVQC):
    """Generate local-Z and ring-ZZ coefficients from one exact statevector."""

    def __init__(self, n_qubits: int, depth: int, entangle: bool = True) -> None:
        super().__init__(n_qubits=n_qubits, depth=depth)
        self.entangle = entangle
        zz = [
            self.z_signs[:, wire] * self.z_signs[:, (wire + 1) % n_qubits]
            for wire in range(n_qubits)
        ]
        self.register_buffer("zz_ring_signs", torch.stack(zz, dim=1), persistent=False)
        self.circuit_calls = 0

    def forward(self) -> torch.Tensor:
        if not self.theta.is_cuda:
            raise RuntimeError("QH-022 exact statevector simulation is CUDA-only")
        dimension = 1 << self.n_qubits
        state = torch.full(
            (1, dimension),
            1.0 / math.sqrt(dimension),
            dtype=torch.complex64,
            device=self.theta.device,
        )
        for layer in range(self.depth):
            for wire in range(self.n_qubits):
                state = self._apply_ry(state, self.theta[layer, wire, 0], wire)
                state = self._apply_rz(state, self.theta[layer, wire, 1], wire)
            if self.entangle:
                for control in range(self.n_qubits):
                    target = (control + 1) % self.n_qubits
                    state = state.index_select(1, getattr(self, f"cnot_{control}_{target}"))
        probabilities = state.abs().square()
        self.circuit_calls += 1
        return torch.cat(
            [probabilities @ self.z_signs, probabilities @ self.zz_ring_signs], dim=-1
        ).squeeze(0)


class GlobalClassicalCoefficientCore(nn.Module):
    """Strictly parameter-matched classical coefficient generator."""

    def __init__(self, n_qubits: int, depth: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.theta = nn.Parameter(torch.empty(depth, n_qubits, 2, dtype=torch.float32))
        nn.init.uniform_(self.theta, -0.05, 0.05)
        rows = 2 * n_qubits
        columns = 2 * depth * n_qubits
        row = torch.arange(rows, dtype=torch.float32).unsqueeze(1) + 1.0
        column = torch.arange(columns, dtype=torch.float32).unsqueeze(0) + 0.5
        projection = torch.cos(math.pi * row * column / columns)
        projection = F.normalize(projection, dim=1)
        self.register_buffer("projection", projection, persistent=False)
        self.circuit_calls = 0

    def forward(self) -> torch.Tensor:
        if not self.theta.is_cuda:
            raise RuntimeError("QH-022 matched controls are GPU-only")
        self.circuit_calls += 1
        return torch.tanh(self.projection @ self.theta.flatten())


class SimulatorParameterGeneratedAdapter(nn.Module):
    """A 25-parameter circuit-generated rank-12 residual update."""

    def __init__(
        self,
        config: QH022Config | None = None,
        core_kind: str = "quantum",
    ) -> None:
        super().__init__()
        self.config = config or QH022Config()
        if self.config.rank != 2 * self.config.n_qubits:
            raise ValueError("rank must equal the local-Z plus ring-ZZ readout width")
        if core_kind == "quantum":
            self.core = GlobalQuantumCoefficientCore(
                self.config.n_qubits, self.config.depth, entangle=True
            )
        elif core_kind == "no_entanglement":
            self.core = GlobalQuantumCoefficientCore(
                self.config.n_qubits, self.config.depth, entangle=False
            )
        elif core_kind == "classical":
            self.core = GlobalClassicalCoefficientCore(
                self.config.n_qubits, self.config.depth
            )
        else:
            raise ValueError(f"unsupported core_kind: {core_kind}")
        self.core_kind = core_kind
        self.gamma = nn.Parameter(torch.tensor(self.config.gamma_init, dtype=torch.float32))

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.basis_seed)
        left = torch.randn(
            self.config.rank, self.config.hidden_size, generator=generator, dtype=torch.float32
        )
        right = torch.randn(
            self.config.hidden_size, self.config.rank, generator=generator, dtype=torch.float32
        )
        self.register_buffer("left_basis", F.normalize(left, dim=1), persistent=True)
        self.register_buffer("right_basis", F.normalize(right, dim=0), persistent=True)
        self.last_coefficients: torch.Tensor | None = None

    def residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hidden_states.is_cuda:
            raise RuntimeError("QH-022 adapter and controls are GPU-only")
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected hidden size {self.config.hidden_size}, got {hidden_states.shape[-1]}"
            )
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.config.hidden_size)
        normalized = F.layer_norm(flat.float(), (self.config.hidden_size,))
        coefficients = self.core().float()
        self.last_coefficients = coefficients.detach()
        low_rank = F.linear(normalized, self.left_basis)
        update = F.linear(low_rank * coefficients, self.right_basis)
        update = update * (self.gamma / math.sqrt(self.config.rank))
        return update.reshape(original_shape).to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.residual(hidden_states)

    def audit(self) -> dict:
        return {
            "candidate": {
                "quantum": "QH-022",
                "classical": "CC-022",
                "no_entanglement": "QH-022-no-ent",
            }[self.core_kind],
            "config": asdict(self.config),
            "core_kind": self.core_kind,
            "trainable_parameters": sum(p.numel() for p in self.parameters()),
            "fixed_basis_parameters": self.left_basis.numel() + self.right_basis.numel(),
            "simulator_execution": "one 6q complex64 CUDA statevector per model forward",
            "inference_simulator_required": self.core_kind != "classical",
            "claim_limit": "classically simulable parameterization; no speedup claim",
        }


class ProjectionInputParameterGenerator(nn.Module):
    def __init__(self, frozen_projection: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.frozen_projection = frozen_projection
        self.adapter = adapter
        for parameter in self.frozen_projection.parameters():
            parameter.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.frozen_projection(self.adapter(hidden_states))


def freeze_and_inject_qh022(
    model: nn.Module,
    core_kind: str = "quantum",
    config: QH022Config | None = None,
) -> dict:
    """Freeze Qwen3.8 and inject QH-022 before one full-attention v_proj."""
    config = config or QH022Config()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = model.model.language_model.layers[config.layer_index]
    if getattr(layer, "block_type", None) != "full_attention":
        raise ValueError(f"layer {config.layer_index} is not a full-attention layer")
    projection = getattr(layer.self_attn, config.target_projection)
    reference = next(projection.parameters())
    adapter = SimulatorParameterGeneratedAdapter(config, core_kind=core_kind).to(reference.device)
    setattr(
        layer.self_attn,
        config.target_projection,
        ProjectionInputParameterGenerator(projection, adapter),
    )
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "candidate": adapter.audit()["candidate"],
        "parent": "QPA arXiv:2410.09846 adapted to retain simulator inference",
        "injection_path": (
            f"model.model.language_model.layers.{config.layer_index}.self_attn."
            f"{config.target_projection}.input"
        ),
        "execution_scope": "once per prefill forward and once per decode-step forward",
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
