"""QH032: shared-basis Pauli-observable nonlinear layer replacement.

QH031 showed that a shared rank-1024 basis can remove more than three million
Qwen3.8 parameters, but its input-independent unitary core failed an independent
confirmation. QH032 keeps only that compression backbone and changes the core
function class. A small circuit defines a learned observable frame; 60 Pauli
expectations gate Pauli-vector residuals, giving an input-dependent cubic map.

Train and inference both execute the exact CUDA statevector path. The module also
contains an independently coded tensor-axis audit and an equal-parameter classical
quadratic-form control. Exact simulation is classically reproducible; no compute
advantage is implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Tuple

import torch
from torch import Tensor, nn

from .qh031_shared_basis_replacement import (
    SharedBasisValueReplacement,
    SharedInputBasis,
)


@dataclass(frozen=True)
class QH032Config:
    hidden_size: int = 5120
    output_size: int = 1024
    rank: int = 1024
    layer_indices: Tuple[int, int] = (55, 59)
    num_qubits: int = 10
    observable_offsets: Tuple[int, int] = (1, 2)
    eps: float = 1.0e-12

    def validate(self) -> None:
        if self.rank != 1 << self.num_qubits:
            raise ValueError("rank must equal 2**num_qubits")
        if len(self.layer_indices) != 2:
            raise ValueError("QH032 currently requires exactly two layers")
        if len(self.observable_offsets) != 2:
            raise ValueError("two observable offsets give the locked 60 observables")
        if any(not 0 < offset < self.num_qubits for offset in self.observable_offsets):
            raise ValueError("invalid observable offset")

    @property
    def observable_count(self) -> int:
        # local X/Z plus XX/ZZ at both configured offsets.
        return self.num_qubits * (2 + 2 * len(self.observable_offsets))


def _require_cuda(x: Tensor) -> None:
    if x.device.type != "cuda":
        raise RuntimeError(f"QH032 exact simulation is GPU-only; received {x.device}")


class _ObservableCore(nn.Module):
    def __init__(self, config: QH032Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.theta = nn.Parameter(torch.zeros(config.num_qubits, 2, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.zeros(config.observable_count, dtype=torch.float32))
        self.register_buffer("circuit_calls", torch.zeros((), dtype=torch.long), persistent=False)

    @property
    def parameter_count(self) -> int:
        return self.theta.numel() + self.alpha.numel()

    def reset_circuit_calls(self) -> None:
        self.circuit_calls.zero_()


def _observable_specs(config: QH032Config):
    specs = []
    for wire in range(config.num_qubits):
        specs.extend((("x", wire, -1), ("z", wire, -1)))
    for offset in config.observable_offsets:
        for wire in range(config.num_qubits):
            other = (wire + offset) % config.num_qubits
            specs.extend((("xx", wire, other), ("zz", wire, other)))
    if len(specs) != config.observable_count:
        raise AssertionError("observable ledger mismatch")
    return tuple(specs)


class ExactPauliObservableCore(_ObservableCore):
    """Paired-index exact complex statevector implementation."""

    def __init__(self, config: QH032Config, use_entanglers: bool = True) -> None:
        super().__init__(config)
        self.use_entanglers = use_entanglers
        basis = torch.arange(config.rank, dtype=torch.long)
        for wire in range(config.num_qubits):
            zero = basis[(basis.bitwise_and(1 << wire)) == 0]
            one = zero.bitwise_or(1 << wire)
            self.register_buffer(f"zero_{wire}", zero, persistent=False)
            self.register_buffer(f"one_{wire}", one, persistent=False)
            self.register_buffer(
                f"zsign_{wire}",
                torch.where(
                    basis.bitwise_and(1 << wire) == 0,
                    torch.ones(config.rank),
                    -torch.ones(config.rank),
                ),
                persistent=False,
            )
            target = (wire + 1) % config.num_qubits
            self.register_buffer(
                f"xperm_{wire}", basis.bitwise_xor(1 << wire), persistent=False
            )
            self.register_buffer(
                f"cnotperm_{wire}",
                torch.where(
                    basis.bitwise_and(1 << wire) != 0,
                    basis.bitwise_xor(1 << target),
                    basis,
                ),
                persistent=False,
            )

    def _rotate(self, state: Tensor, wire: int, theta: Tensor, axis: str) -> Tensor:
        zero, one = getattr(self, f"zero_{wire}"), getattr(self, f"one_{wire}")
        a0, a1 = state.index_select(-1, zero), state.index_select(-1, one)
        half = theta * 0.5
        c, s = torch.cos(half), torch.sin(half)
        out = state.clone()
        if axis == "rx":
            out.index_copy_(-1, zero, c * a0 - 1j * s * a1)
            out.index_copy_(-1, one, -1j * s * a0 + c * a1)
        else:
            out.index_copy_(-1, zero, c * a0 - s * a1)
            out.index_copy_(-1, one, s * a0 + c * a1)
        return out

    def _cnot(self, state: Tensor, wire: int) -> Tensor:
        return state.index_select(-1, getattr(self, f"cnotperm_{wire}"))

    def _pauli(self, state: Tensor, spec) -> Tensor:
        kind, first, second = spec
        if kind == "x":
            return state.index_select(-1, getattr(self, f"xperm_{first}"))
        if kind == "z":
            return state * getattr(self, f"zsign_{first}")
        if kind == "xx":
            perm = getattr(self, f"xperm_{first}").bitwise_xor(1 << second)
            return state.index_select(-1, perm)
        if kind == "zz":
            return state * getattr(self, f"zsign_{first}") * getattr(self, f"zsign_{second}")
        raise AssertionError(kind)

    def _forward_frame(self, state: Tensor) -> Tensor:
        for wire in range(self.config.num_qubits):
            state = self._rotate(state, wire, self.theta[wire, 0], "rx")
            state = self._rotate(state, wire, self.theta[wire, 1], "ry")
        if self.use_entanglers:
            for wire in range(self.config.num_qubits):
                state = self._cnot(state, wire)
        return state

    def _inverse_frame(self, state: Tensor) -> Tensor:
        if self.use_entanglers:
            for wire in reversed(range(self.config.num_qubits)):
                state = self._cnot(state, wire)
        for wire in reversed(range(self.config.num_qubits)):
            state = self._rotate(state, wire, -self.theta[wire, 1], "ry")
            state = self._rotate(state, wire, -self.theta[wire, 0], "rx")
        return state

    def forward(self, values: Tensor) -> Tensor:
        _require_cuda(values)
        original_shape = values.shape
        flat = values.reshape(-1, self.config.rank).float()
        norm = flat.square().sum(-1, keepdim=True).clamp_min(self.config.eps).sqrt()
        state = torch.complex(flat / norm, torch.zeros_like(flat))
        framed = self._forward_frame(state)
        residual = torch.zeros_like(framed)
        scale = float(self.config.observable_count) ** 0.5
        for coefficient, spec in zip(torch.tanh(self.alpha), _observable_specs(self.config)):
            acted = self._pauli(framed, spec)
            expectation = (framed.conj() * acted).sum(-1, keepdim=True).real
            residual = residual + coefficient * expectation * acted / scale
        residual = self._inverse_frame(residual)
        self.circuit_calls.add_(1)
        output = flat + norm * residual.real
        return output.reshape(original_shape).to(values.dtype)


class DequantizedPauliObservableCore(_ObservableCore):
    """Independent tensor-axis implementation of QH032 for numerical audit."""

    def __init__(self, config: QH032Config, use_entanglers: bool = True) -> None:
        super().__init__(config)
        self.use_entanglers = use_entanglers

    def _single(self, state: Tensor, wire: int, theta: Tensor, axis: str) -> Tensor:
        tensor_axis = 1 + (self.config.num_qubits - 1 - wire)
        moved = state.movedim(tensor_axis, -1)
        half = theta * 0.5
        c, s = torch.cos(half), torch.sin(half)
        if axis == "rx":
            gate = torch.stack((
                torch.stack((torch.complex(c, c * 0), torch.complex(c * 0, -s))),
                torch.stack((torch.complex(c * 0, -s), torch.complex(c, c * 0))),
            ))
        else:
            gate = torch.complex(
                torch.stack((torch.stack((c, -s)), torch.stack((s, c)))),
                torch.zeros(2, 2, device=state.device),
            )
        return torch.matmul(moved, gate.transpose(0, 1)).movedim(-1, tensor_axis)

    def _cnot(self, state: Tensor, control: int) -> Tensor:
        target = (control + 1) % self.config.num_qubits
        ca = 1 + (self.config.num_qubits - 1 - control)
        ta = 1 + (self.config.num_qubits - 1 - target)
        moved = state.movedim((ca, ta), (-2, -1))
        pairs = moved.reshape(*moved.shape[:-2], 4)
        acted = pairs[..., [0, 1, 3, 2]].reshape(moved.shape)
        return acted.movedim((-2, -1), (ca, ta))

    def _pauli(self, state: Tensor, spec) -> Tensor:
        kind, first, second = spec
        result = state
        wires = (first,) if second < 0 else (first, second)
        operator = kind[0]
        for wire in wires:
            axis = 1 + (self.config.num_qubits - 1 - wire)
            if operator == "x":
                result = result.index_select(axis, torch.tensor([1, 0], device=state.device))
            else:
                shape = [1] * result.ndim
                shape[axis] = 2
                sign = torch.tensor([1.0, -1.0], device=state.device).reshape(shape)
                result = result * sign
        return result

    def _forward_frame(self, state: Tensor) -> Tensor:
        for wire in range(self.config.num_qubits):
            state = self._single(state, wire, self.theta[wire, 0], "rx")
            state = self._single(state, wire, self.theta[wire, 1], "ry")
        if self.use_entanglers:
            for wire in range(self.config.num_qubits):
                state = self._cnot(state, wire)
        return state

    def _inverse_frame(self, state: Tensor) -> Tensor:
        if self.use_entanglers:
            for wire in reversed(range(self.config.num_qubits)):
                state = self._cnot(state, wire)
        for wire in reversed(range(self.config.num_qubits)):
            state = self._single(state, wire, -self.theta[wire, 1], "ry")
            state = self._single(state, wire, -self.theta[wire, 0], "rx")
        return state

    def forward(self, values: Tensor) -> Tensor:
        _require_cuda(values)
        original_shape = values.shape
        flat = values.reshape(-1, self.config.rank).float()
        norm = flat.square().sum(-1, keepdim=True).clamp_min(self.config.eps).sqrt()
        state = torch.complex(flat / norm, torch.zeros_like(flat)).reshape(
            -1, *([2] * self.config.num_qubits)
        )
        framed = self._forward_frame(state)
        residual = torch.zeros_like(framed)
        scale = float(self.config.observable_count) ** 0.5
        batch = framed.shape[0]
        for coefficient, spec in zip(torch.tanh(self.alpha), _observable_specs(self.config)):
            acted = self._pauli(framed, spec)
            expectation = (framed.conj() * acted).reshape(batch, -1).sum(-1).real
            residual = residual + coefficient * expectation.reshape(batch, *([1] * self.config.num_qubits)) * acted / scale
        residual = self._inverse_frame(residual).reshape(-1, self.config.rank)
        self.circuit_calls.add_(1)
        return (flat + norm * residual.real).reshape(original_shape).to(values.dtype)


class EqualParameterClassicalObservableCore(_ObservableCore):
    """Matched-degree 80-parameter classical quadratic-form residual."""

    def __init__(self, config: QH032Config) -> None:
        super().__init__(config)
        basis = torch.arange(config.rank, dtype=torch.long)
        for wire in range(config.num_qubits):
            self.register_buffer(f"xperm_{wire}", basis.bitwise_xor(1 << wire), persistent=False)
            self.register_buffer(
                f"zsign_{wire}",
                torch.where(
                    basis.bitwise_and(1 << wire) == 0,
                    torch.ones(config.rank), -torch.ones(config.rank),
                ), persistent=False,
            )

    def _operator(self, state: Tensor, spec) -> Tensor:
        kind, first, second = spec
        if kind == "x":
            return state.index_select(-1, getattr(self, f"xperm_{first}"))
        if kind == "z":
            return state * getattr(self, f"zsign_{first}")
        if kind == "xx":
            perm = getattr(self, f"xperm_{first}").bitwise_xor(1 << second)
            return state.index_select(-1, perm)
        return state * getattr(self, f"zsign_{first}") * getattr(self, f"zsign_{second}")

    def forward(self, values: Tensor) -> Tensor:
        _require_cuda(values)
        original_shape = values.shape
        flat = values.reshape(-1, self.config.rank).float()
        norm = flat.square().sum(-1, keepdim=True).clamp_min(self.config.eps).sqrt()
        state = flat / norm
        pre_scale = float(2 * self.config.num_qubits) ** 0.5
        for wire in range(self.config.num_qubits):
            state = state + torch.tanh(self.theta[wire, 0]) * state * getattr(self, f"zsign_{wire}") / pre_scale
            state = state + torch.tanh(self.theta[wire, 1]) * state.index_select(-1, getattr(self, f"xperm_{wire}")) / pre_scale
        residual = torch.zeros_like(state)
        scale = float(self.config.observable_count) ** 0.5
        for coefficient, spec in zip(torch.tanh(self.alpha), _observable_specs(self.config)):
            acted = self._operator(state, spec)
            moment = (state * acted).sum(-1, keepdim=True)
            residual = residual + coefficient * moment * acted / scale
        return (flat + norm * residual).reshape(original_shape).to(values.dtype)


def build_qh032_replacements(
    dense_layers: Mapping[int, nn.Linear],
    config: QH032Config,
    *,
    branch: str = "quantum",
) -> Dict[int, SharedBasisValueReplacement]:
    config.validate()
    if tuple(dense_layers) != config.layer_indices:
        raise ValueError("dense layer order must match QH032 layer_indices")
    ordered = [dense_layers[index] for index in config.layer_indices]
    device, dtype = ordered[0].weight.device, ordered[0].weight.dtype
    _require_cuda(ordered[0].weight)
    stacked = torch.cat([layer.weight.detach().float() for layer in ordered], dim=0)
    u, s, vh = torch.linalg.svd(stacked, full_matrices=False)
    shared = SharedInputBasis(config.hidden_size, config.rank, dtype=dtype).to(device)
    shared.down.weight.copy_(vh[: config.rank].to(dtype))
    classes = {
        "quantum": ExactPauliObservableCore,
        "dq": DequantizedPauliObservableCore,
        "classical": EqualParameterClassicalObservableCore,
        "no_ent": lambda cfg: ExactPauliObservableCore(cfg, use_entanglers=False),
    }
    if branch not in classes:
        raise ValueError(f"unknown QH032 branch: {branch}")
    core = classes[branch](config).to(device)
    us = u[:, : config.rank] * s[: config.rank].unsqueeze(0)
    result = {}
    for position, layer_index in enumerate(config.layer_indices):
        replacement = SharedBasisValueReplacement(shared, core, config.output_size, dtype=dtype).to(device)
        start = position * config.output_size
        replacement.up.weight.copy_(us[start : start + config.output_size].to(dtype))
        result[layer_index] = replacement
    return result


def qh032_parameter_audit(config: QH032Config) -> MutableMapping[str, int]:
    config.validate()
    removed = len(config.layer_indices) * config.hidden_size * config.output_size
    frozen = config.rank * config.hidden_size + len(config.layer_indices) * config.output_size * config.rank
    trainable = 2 * config.num_qubits + config.observable_count + len(config.layer_indices)
    return {
        "original_parameters_removed": removed,
        "frozen_replacement_parameters": frozen,
        "trainable_parameters": trainable,
        "total_replacement_parameters": frozen + trainable,
        "net_parameter_reduction": removed - frozen - trainable,
        "observable_count": config.observable_count,
    }


def install_qh032(model: nn.Module, config: QH032Config = QH032Config(), *, branch: str = "quantum"):
    config.validate()
    model.requires_grad_(False)
    language_model = model.model.language_model
    originals = {}
    for index in config.layer_indices:
        projection = language_model.layers[index].self_attn.v_proj
        if not isinstance(projection, nn.Linear):
            raise TypeError(f"layer {index} v_proj is not dense before QH032 install")
        originals[index] = projection
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    replacements = build_qh032_replacements(originals, config, branch=branch)
    for index, replacement in replacements.items():
        language_model.layers[index].self_attn.v_proj = replacement
    deployed = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    audit = qh032_parameter_audit(config)
    if base_parameters - deployed != audit["net_parameter_reduction"] or trainable != audit["trainable_parameters"]:
        raise AssertionError("QH032 unique-parameter audit mismatch")
    return replacements, originals, {
        **audit,
        "base_model_parameters": base_parameters,
        "deployed_model_parameters": deployed,
        "net_model_parameter_reduction": audit["net_parameter_reduction"],
        "trainable_parameter_count": trainable,
        "branch": branch,
        "layer_indices": list(config.layer_indices),
        "rank": config.rank,
        "num_qubits": config.num_qubits,
    }


def unique_trainable_parameters(replacements: Mapping[int, nn.Module]):
    seen = set()
    for replacement in replacements.values():
        for parameter in replacement.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                yield parameter
