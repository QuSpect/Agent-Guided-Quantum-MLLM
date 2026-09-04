"""QH-012: four-qubit brickwork Cayley adapters for visual prefill."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .cayley_two_qubit import (
    VisualMergerOutputAdapter,
    _single_qubit_zyz,
    _skew_from_six,
    _symmetric_from_six,
)


@dataclass(frozen=True)
class QH012Config:
    hidden_size: int = 5120
    register_size: int = 16
    qubits_per_register: int = 4
    gates_per_register: int = 4
    init_std: float = 0.0


class FourQubitBrickworkTransform(nn.Module):
    """Four overlapping two-qubit gates per 16-amplitude register.

    The schedule (01, 23, 12, 30) has two brickwork rounds.  Unlike QH-010,
    the second round crosses the first-round partition and can create genuine
    four-qubit correlations while retaining the same 7,680-parameter budget.
    """

    schedule = ((0, 1), (2, 3), (1, 2), (3, 0))

    def __init__(self, config: QH012Config | None = None, core_kind: str = "quantum") -> None:
        super().__init__()
        self.config = config or QH012Config()
        if self.config.register_size != 2 ** self.config.qubits_per_register:
            raise ValueError("register_size must equal 2**qubits_per_register")
        if self.config.qubits_per_register != 4 or self.config.gates_per_register != 4:
            raise ValueError("QH-012 implements the preregistered four-qubit/four-gate circuit")
        if self.config.hidden_size % self.config.register_size:
            raise ValueError("hidden_size must be divisible by register_size")
        if core_kind not in {"quantum", "dequantized", "classical", "no_entanglement"}:
            raise ValueError(f"unsupported core_kind: {core_kind}")
        self.core_kind = core_kind
        self.n_registers = self.config.hidden_size // self.config.register_size
        self.theta = nn.Parameter(
            torch.empty(self.n_registers, self.config.gates_per_register, 6, dtype=torch.float32)
        )
        nn.init.normal_(self.theta, mean=0.0, std=self.config.init_std)
        self.register_buffer("identity4", torch.eye(4, dtype=torch.float32), persistent=False)

    def _cayley_gates(self) -> torch.Tensor:
        flat_theta = self.theta.reshape(-1, 6)
        generator = _skew_from_six(flat_theta)
        identity = self.identity4.expand(len(flat_theta), -1, -1)
        left = identity - 0.5 * generator
        right = identity + 0.5 * generator
        gates = torch.linalg.solve(
            right.transpose(-1, -2), left.transpose(-1, -2)
        ).transpose(-1, -2)
        return gates.reshape(self.n_registers, self.config.gates_per_register, 4, 4)

    def _classical_gates(self) -> torch.Tensor:
        flat_theta = self.theta.reshape(-1, 6)
        identity = self.identity4.expand(len(flat_theta), -1, -1)
        gates = identity + _symmetric_from_six(flat_theta)
        return gates.reshape(self.n_registers, self.config.gates_per_register, 4, 4)

    def _separable_gates(self) -> torch.Tensor:
        flat_theta = self.theta.reshape(-1, 6)
        first = _single_qubit_zyz(flat_theta[:, :3])
        second = _single_qubit_zyz(flat_theta[:, 3:])
        gates = torch.einsum("bij,bkl->bikjl", first, second).reshape(-1, 4, 4)
        return gates.reshape(self.n_registers, self.config.gates_per_register, 4, 4)

    @staticmethod
    def _apply_two_qubit_gate(
        state: torch.Tensor, gate: torch.Tensor, wires: tuple[int, int]
    ) -> torch.Tensor:
        """Apply a register-specific 4x4 gate to two axes of [N,R,2,2,2,2]."""
        first, second = wires
        if first == second or not (0 <= first < 4 and 0 <= second < 4):
            raise ValueError(f"invalid wires: {wires}")
        remaining = [wire for wire in range(4) if wire not in wires]
        output_order = remaining + [first, second]
        permutation = [0, 1] + [2 + wire for wire in output_order]
        packed = state.permute(permutation).reshape(
            state.shape[0], state.shape[1], 4, 4
        )
        # b is the independent feature register and r is the untouched two-qubit
        # basis. They must never share an einsum label: doing so silently coupled
        # registers when n_registers happened to equal four in the toy test.
        transformed = torch.einsum("bij,nbrj->nbri", gate, packed)
        ordered = transformed.reshape(state.shape[0], state.shape[1], 2, 2, 2, 2)
        inverse = [0, 1] + [2 + output_order.index(wire) for wire in range(4)]
        return ordered.permute(inverse)

    def _run_schedule(self, state: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        for gate_index, wires in enumerate(self.schedule):
            state = self._apply_two_qubit_gate(state, gates[:, gate_index], wires)
        return state

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hidden_states.is_cuda:
            raise RuntimeError("QH-012 simulation and controls are GPU-only")
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected hidden size {self.config.hidden_size}, got {hidden_states.shape[-1]}"
            )
        original_shape = hidden_states.shape
        blocks = hidden_states.reshape(-1, self.n_registers, 16).float()
        if self.core_kind == "classical":
            state = blocks.reshape(-1, self.n_registers, 2, 2, 2, 2)
            transformed = self._run_schedule(state, self._classical_gates()).reshape_as(blocks)
        elif self.core_kind == "dequantized":
            state = blocks.reshape(-1, self.n_registers, 2, 2, 2, 2)
            amplitudes = self._run_schedule(state, self._cayley_gates()).reshape_as(blocks)
            transformed = amplitudes.abs() * torch.sign(blocks)
        else:
            norm = torch.linalg.vector_norm(blocks, dim=-1, keepdim=True)
            normalized = torch.where(norm > 0.0, blocks / norm.clamp_min(1.0e-12), blocks)
            gates = (
                self._cayley_gates().to(torch.complex64)
                if self.core_kind == "quantum"
                else self._separable_gates()
            )
            state = normalized.to(torch.complex64).reshape(
                -1, self.n_registers, 2, 2, 2, 2
            )
            amplitudes = self._run_schedule(state, gates).reshape_as(blocks)
            transformed = amplitudes.abs() * torch.sign(blocks) * norm
        return transformed.reshape(original_shape).to(hidden_states.dtype)

    def audit(self) -> dict:
        return {
            "config": asdict(self.config),
            "core_kind": self.core_kind,
            "n_four_qubit_registers": self.n_registers,
            "two_qubit_gate_schedule": [list(pair) for pair in self.schedule],
            "parameters_per_two_qubit_gate": 6,
            "trainable_parameters": self.theta.numel(),
            "simulator": "exact CUDA statevector; complex64 four-qubit registers",
            "measurement": "amplitude magnitude reconstruction with input sign correction",
        }


def freeze_and_inject_qh012(
    model: nn.Module,
    core_kind: str = "quantum",
    config: QH012Config | None = None,
) -> dict:
    """Freeze Qwen3.8 and apply QH-012 only to visual merger outputs."""
    config = config or QH012Config()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.model.visual
    merger = visual.merger
    reference = next(merger.parameters())
    transform = FourQubitBrickworkTransform(config, core_kind=core_kind).to(reference.device)
    visual.merger = VisualMergerOutputAdapter(merger, transform)
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    candidate = {
        "quantum": "QH-012",
        "dequantized": "DQ-012",
        "classical": "CC-012",
        "no_entanglement": "QH-012-no-ent",
    }[core_kind]
    return {
        "candidate": candidate,
        "parent": "QH-010/011 weak-entanglement cross-block mutation",
        "injection_path": "model.model.visual.merger.output",
        "execution_scope": "visual prefill only; no text-only or decode-token invocation",
        "adapter": transform.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
