"""QH-023 static prototype: simulator-resident Pauli/Stiefel PEFT.

This implements the real RY/CZ Pauli pattern from Quantum-PEFT Eq. (2).
Four basis states are propagated through each unitary, so the returned columns
are orthonormal by construction.  Qwen's width 5120 is covered exactly by
4096- and 1024-dimensional registers; no padded coordinate is truncated.

The module is intentionally CUDA-only.  It is a gate-level exact statevector
simulator, not evidence of quantum speedup and not yet a performance candidate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class QH023Config:
    hidden_size: int = 5120
    register_widths: tuple[int, int] = (4096, 1024)
    rank: int = 4
    depth: int = 2
    layer_index: int = 7
    target_projection: str = "v_proj"
    gamma_init: float = 0.05
    singular_init: float = 0.01
    angle_seed: int = 20260835

    def __post_init__(self) -> None:
        if sum(self.register_widths) != self.hidden_size:
            raise ValueError("register widths must exactly cover hidden size")
        if self.rank < 1 or self.rank > min(self.register_widths):
            raise ValueError("rank must fit in every statevector register")
        if self.depth < 1:
            raise ValueError("depth must be positive")
        for width in self.register_widths:
            if width < 2 or width & (width - 1):
                raise ValueError("register widths must be powers of two")


class PauliStiefelCircuit(nn.Module):
    """Apply the formula-matched alternating RY/CZ circuit to basis columns."""

    def __init__(
        self,
        n_qubits: int,
        depth: int,
        *,
        seed: int,
        entangle: bool = True,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.width = 1 << n_qubits
        self.entangle = entangle
        self.angle_count = (2 * depth + 1) * n_qubits - 2 * depth
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        angles = torch.empty(self.angle_count, dtype=torch.float32)
        angles.uniform_(-0.02, 0.02, generator=generator)
        self.theta = nn.Parameter(angles)
        basis = torch.arange(self.width, dtype=torch.long)
        for wire in range(n_qubits):
            zero = basis[(basis & (1 << wire)) == 0]
            self.register_buffer(f"zero_{wire}", zero, persistent=False)
            self.register_buffer(f"one_{wire}", zero | (1 << wire), persistent=False)
        for left in range(n_qubits - 1):
            sign = torch.ones(self.width, dtype=torch.float32)
            active = ((basis & (1 << left)) != 0) & ((basis & (1 << (left + 1))) != 0)
            sign[active] = -1.0
            self.register_buffer(f"cz_{left}_{left + 1}", sign, persistent=False)
        self.circuit_calls = 0

    def _apply_ry(self, state: torch.Tensor, angle: torch.Tensor, wire: int) -> torch.Tensor:
        zero = getattr(self, f"zero_{wire}")
        one = getattr(self, f"one_{wire}")
        a0 = state.index_select(1, zero)
        a1 = state.index_select(1, one)
        cosine = torch.cos(angle * 0.5)
        sine = torch.sin(angle * 0.5)
        output = state.clone()
        output[:, zero] = cosine * a0 - sine * a1
        output[:, one] = sine * a0 + cosine * a1
        return output

    def _apply_cz_pairs(self, state: torch.Tensor, offset: int) -> torch.Tensor:
        if not self.entangle:
            return state
        for left in range(offset, self.n_qubits - 1, 2):
            state = state * getattr(self, f"cz_{left}_{left + 1}")
        return state

    def forward(self, rank: int) -> torch.Tensor:
        if not self.theta.is_cuda:
            raise RuntimeError("QH-023 exact statevector simulation is CUDA-only")
        if rank < 1 or rank > self.width:
            raise ValueError("invalid Stiefel rank")
        state = torch.zeros(rank, self.width, device=self.theta.device, dtype=torch.float32)
        rows = torch.arange(rank, device=self.theta.device)
        state[rows, rows] = 1.0
        cursor = 0
        for wire in range(self.n_qubits):
            state = self._apply_ry(state, self.theta[cursor], wire)
            cursor += 1
        for _ in range(self.depth):
            for wire in range(self.n_qubits - 1):
                state = self._apply_ry(state, self.theta[cursor], wire)
                cursor += 1
            state = self._apply_cz_pairs(state, offset=0)
            for wire in range(1, self.n_qubits):
                state = self._apply_ry(state, self.theta[cursor], wire)
                cursor += 1
            state = self._apply_cz_pairs(state, offset=1)
        if cursor != self.angle_count:
            raise AssertionError("Pauli angle accounting mismatch")
        self.circuit_calls += 1
        return state.transpose(0, 1).contiguous()


class ClassicalStiefelCore(nn.Module):
    """Equal-parameter classical Stiefel generator using a dense tangent dictionary."""

    def __init__(self, width: int, angle_count: int, rank: int, *, seed: int) -> None:
        super().__init__()
        self.width = width
        self.angle_count = angle_count
        self.rank = rank
        self.theta = nn.Parameter(torch.zeros(angle_count, dtype=torch.float32))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        base = torch.randn(width, rank, generator=generator, dtype=torch.float32)
        base = torch.linalg.qr(base, mode="reduced").Q
        directions = torch.randn(
            angle_count, width, rank, generator=generator, dtype=torch.float32
        )
        directions = directions / directions.flatten(1).norm(dim=1).clamp_min(1e-8)[:, None, None]
        self.register_buffer("base", base, persistent=True)
        self.register_buffer("directions", directions, persistent=True)
        self.circuit_calls = 0

    def forward(self, rank: int) -> torch.Tensor:
        if not self.theta.is_cuda:
            raise RuntimeError("CC-023 matched Stiefel control is GPU-only")
        if rank != self.rank:
            raise ValueError("classical Stiefel rank mismatch")
        displacement = torch.einsum("a,awr->wr", torch.tanh(self.theta), self.directions)
        raw = self.base + displacement / math.sqrt(self.angle_count)
        q, r = torch.linalg.qr(raw, mode="reduced")
        orientation = torch.where(
            torch.diagonal(r).detach() < 0,
            -torch.ones(rank, device=q.device),
            torch.ones(rank, device=q.device),
        )
        self.circuit_calls += 1
        return q * orientation


class TensorContractionPauliCircuit(PauliStiefelCircuit):
    """Independent exact DQ realization using tensor-axis gate contraction."""

    def _apply_ry(self, state: torch.Tensor, angle: torch.Tensor, wire: int) -> torch.Tensor:
        cosine = torch.cos(angle * 0.5)
        sine = torch.sin(angle * 0.5)
        matrix = torch.stack(
            [torch.stack([cosine, -sine]), torch.stack([sine, cosine])]
        )
        tensor = state.reshape(state.shape[0], *([2] * self.n_qubits))
        axis = 1 + (self.n_qubits - 1 - wire)
        moved = tensor.movedim(axis, -1)
        contracted = torch.einsum("...j,ij->...i", moved, matrix)
        return contracted.movedim(-1, axis).reshape_as(state)

    def _apply_cz_pairs(self, state: torch.Tensor, offset: int) -> torch.Tensor:
        if not self.entangle:
            return state
        # Diagonal tensor contraction is deliberately reconstructed here rather
        # than calling the gate-level implementation used by QH-023.
        basis = torch.arange(self.width, device=state.device)
        sign = torch.ones(self.width, device=state.device, dtype=state.dtype)
        for left in range(offset, self.n_qubits - 1, 2):
            active = ((basis >> left) & 1) * ((basis >> (left + 1)) & 1)
            sign = sign * (1.0 - 2.0 * active.to(state.dtype))
        return torch.einsum("bi,i->bi", state, sign)


class PauliStiefelAdapter(nn.Module):
    """A 209-parameter rank-4 residual update generated every forward."""

    def __init__(
        self,
        config: QH023Config | None = None,
        *,
        entangle: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or QH023Config()
        qubits = [int(math.log2(width)) for width in self.config.register_widths]
        names = ("u_high", "u_low", "v_high", "v_low")
        seeds = [self.config.angle_seed + offset for offset in range(4)]
        self.circuits = nn.ModuleDict({
            name: PauliStiefelCircuit(
                qubits[index % 2], self.config.depth, seed=seeds[index], entangle=entangle
            )
            for index, name in enumerate(names)
        })
        self.singular_values = nn.Parameter(
            torch.full((self.config.rank,), self.config.singular_init, dtype=torch.float32)
        )
        self.gamma = nn.Parameter(torch.tensor(self.config.gamma_init, dtype=torch.float32))
        self.entangle = entangle
        self.last_orthogonality_error: dict[str, float] = {}

    def factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        rank = self.config.rank
        scale = 1.0 / math.sqrt(len(self.config.register_widths))
        u = torch.cat(
            [self.circuits["u_high"](rank), self.circuits["u_low"](rank)], dim=0
        ) * scale
        v = torch.cat(
            [self.circuits["v_high"](rank), self.circuits["v_low"](rank)], dim=0
        ) * scale
        identity = torch.eye(rank, device=u.device, dtype=u.dtype)
        self.last_orthogonality_error = {
            "u": float((u.transpose(0, 1) @ u - identity).detach().abs().max()),
            "v": float((v.transpose(0, 1) @ v - identity).detach().abs().max()),
        }
        return u, v

    @property
    def circuit_calls(self) -> int:
        return sum(circuit.circuit_calls for circuit in self.circuits.values())

    def residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hidden_states.is_cuda:
            raise RuntimeError("QH-023 adapter is CUDA-only")
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("hidden size mismatch")
        original_shape = hidden_states.shape
        values = F.layer_norm(
            hidden_states.reshape(-1, self.config.hidden_size).float(),
            (self.config.hidden_size,),
        )
        u, v = self.factors()
        coordinates = values @ v
        update = (coordinates * self.singular_values) @ u.transpose(0, 1)
        return (self.gamma * update).reshape(original_shape).to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.residual(hidden_states)

    def audit(self) -> dict:
        return {
            "candidate": "QH-023" if self.entangle else "QH-023-no-ent",
            "config": asdict(self.config),
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "angle_parameters": sum(circuit.theta.numel() for circuit in self.circuits.values()),
            "factorization": "delta_W = U diag(lambda) V^T",
            "simulator_calls_per_forward": 4,
            "statevector_registers": ["12q", "10q", "12q", "10q"],
            "inference_simulator_required": True,
            "claim_limit": "static simulator prototype; no task benefit or speedup claim",
        }


class ClassicalStiefelAdapter(PauliStiefelAdapter):
    """CC-023: 209 trainable parameters with a non-circuit Stiefel generator."""

    def __init__(self, config: QH023Config | None = None) -> None:
        nn.Module.__init__(self)
        self.config = config or QH023Config()
        qubits = [int(math.log2(width)) for width in self.config.register_widths]
        angle_counts = [
            (2 * self.config.depth + 1) * q - 2 * self.config.depth for q in qubits
        ]
        names = ("u_high", "u_low", "v_high", "v_low")
        seeds = [self.config.angle_seed + 100 + offset for offset in range(4)]
        self.circuits = nn.ModuleDict({
            name: ClassicalStiefelCore(
                self.config.register_widths[index % 2],
                angle_counts[index % 2],
                self.config.rank,
                seed=seeds[index],
            )
            for index, name in enumerate(names)
        })
        self.singular_values = nn.Parameter(
            torch.full((self.config.rank,), self.config.singular_init, dtype=torch.float32)
        )
        self.gamma = nn.Parameter(torch.tensor(self.config.gamma_init, dtype=torch.float32))
        self.entangle = False
        self.last_orthogonality_error: dict[str, float] = {}

    def audit(self) -> dict:
        value = super().audit()
        value.update({
            "candidate": "CC-023",
            "generator": "dense classical tangent dictionary followed by thin QR",
            "inference_simulator_required": False,
            "claim_limit": "equal-trainable-parameter classical structural control",
        })
        return value


class DequantizedPauliStiefelAdapter(PauliStiefelAdapter):
    """DQ-023: independent tensor-contraction realization of the same map."""

    def __init__(self, config: QH023Config | None = None) -> None:
        nn.Module.__init__(self)
        self.config = config or QH023Config()
        qubits = [int(math.log2(width)) for width in self.config.register_widths]
        names = ("u_high", "u_low", "v_high", "v_low")
        seeds = [self.config.angle_seed + offset for offset in range(4)]
        self.circuits = nn.ModuleDict({
            name: TensorContractionPauliCircuit(
                qubits[index % 2], self.config.depth, seed=seeds[index], entangle=True
            )
            for index, name in enumerate(names)
        })
        self.singular_values = nn.Parameter(
            torch.full((self.config.rank,), self.config.singular_init, dtype=torch.float32)
        )
        self.gamma = nn.Parameter(torch.tensor(self.config.gamma_init, dtype=torch.float32))
        self.entangle = True
        self.last_orthogonality_error: dict[str, float] = {}

    def audit(self) -> dict:
        value = super().audit()
        value.update({
            "candidate": "DQ-023",
            "generator": "independent exact tensor-axis contraction of RY and diagonal CZ",
            "inference_simulator_required": False,
            "claim_limit": "exactly dequantized control; bounds claims to executable structure",
        })
        return value


class ProjectionInputPauliStiefel(nn.Module):
    def __init__(self, frozen_projection: nn.Module, adapter: PauliStiefelAdapter) -> None:
        super().__init__()
        self.frozen_projection = frozen_projection
        self.adapter = adapter
        for parameter in frozen_projection.parameters():
            parameter.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.frozen_projection(self.adapter(hidden_states))


def freeze_and_inject_qh023(model: nn.Module, config: QH023Config | None = None) -> dict:
    """Freeze Qwen3.8 and inject the QH-023 static prototype before one v_proj."""
    config = config or QH023Config()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = model.model.language_model.layers[config.layer_index]
    if getattr(layer, "block_type", None) != "full_attention":
        raise ValueError(f"layer {config.layer_index} is not a full-attention layer")
    projection = getattr(layer.self_attn, config.target_projection)
    reference = next(projection.parameters())
    adapter = PauliStiefelAdapter(config).to(reference.device)
    setattr(
        layer.self_attn,
        config.target_projection,
        ProjectionInputPauliStiefel(projection, adapter),
    )
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    return {
        "candidate": "QH-023-static",
        "injection_path": (
            f"model.model.language_model.layers.{config.layer_index}.self_attn."
            f"{config.target_projection}.input"
        ),
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
