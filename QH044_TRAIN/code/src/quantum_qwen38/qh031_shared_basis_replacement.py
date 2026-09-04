"""QH031: shared-basis two-layer quantum replacement.

This module replaces two dense value projections with a shared low-rank input
basis, layer-specific output bases, and a shared amplitude-state quantum core.
The circuit is simulated exactly with PyTorch tensors on CUDA; no quantum
hardware or shot sampling is involved.

The implementation deliberately contains three independently coded branches:

* ``quantum``: index-pair statevector gate application;
* ``dq``: tensor-axis dense-gate dequantization audit;
* ``classical``: an equal-parameter structured classical control.

At zero initialization all three branches are exact identities in the shared
rank space, so installing QH031 initially reproduces the shared-SVD backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class QH031Config:
    hidden_size: int = 5120
    output_size: int = 1024
    rank: int = 1024
    layer_indices: Tuple[int, int] = (55, 59)
    num_qubits: int = 10
    depth: int = 2
    entangling_offsets: Tuple[int, ...] = (1, 2, 4)
    eps: float = 1.0e-12

    def validate(self) -> None:
        if self.rank != 1 << self.num_qubits:
            raise ValueError(
                f"rank must equal 2**num_qubits for norm-preserving amplitude "
                f"encoding; got rank={self.rank}, num_qubits={self.num_qubits}"
            )
        if len(self.layer_indices) != 2:
            raise ValueError("QH031 currently requires exactly two replaced layers")
        if self.depth < 1:
            raise ValueError("depth must be positive")
        for offset in self.entangling_offsets:
            if not 0 < offset < self.num_qubits:
                raise ValueError(f"invalid entangling offset: {offset}")


def _require_cuda(x: Tensor) -> None:
    if x.device.type != "cuda":
        raise RuntimeError(
            "QH031 exact simulation is GPU-only by experiment contract; "
            f"received device={x.device}"
        )


def _ry_matrix(theta: Tensor) -> Tensor:
    half = theta * 0.5
    c, s = torch.cos(half), torch.sin(half)
    return torch.stack((torch.stack((c, -s)), torch.stack((s, c))))


class _ParameterizedCore(nn.Module):
    """Common parameter surface shared by all causal-control branches."""

    def __init__(self, config: QH031Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.theta = nn.Parameter(
            torch.zeros(
                config.depth,
                config.num_qubits,
                1 + len(config.entangling_offsets),
                dtype=torch.float32,
            )
        )
        self.register_buffer(
            "circuit_calls", torch.zeros((), dtype=torch.long), persistent=False
        )

    @property
    def parameter_count(self) -> int:
        return self.theta.numel()

    def reset_circuit_calls(self) -> None:
        self.circuit_calls.zero_()


class ExactQuantumCore(_ParameterizedCore):
    """Exact real-amplitude statevector simulation using paired indices."""

    def __init__(self, config: QH031Config, use_entanglers: bool = True) -> None:
        super().__init__(config)
        self.use_entanglers = use_entanglers
        basis = torch.arange(config.rank, dtype=torch.long)
        for wire in range(config.num_qubits):
            zero = basis[(basis.bitwise_and(1 << wire)) == 0]
            one = zero.bitwise_or(1 << wire)
            self.register_buffer(f"ry_zero_{wire}", zero, persistent=False)
            self.register_buffer(f"ry_one_{wire}", one, persistent=False)
        for control in range(config.num_qubits):
            for offset in config.entangling_offsets:
                target = (control + offset) % config.num_qubits
                mask = ((basis.bitwise_and(1 << control)) != 0) & (
                    (basis.bitwise_and(1 << target)) == 0
                )
                zero = basis[mask]
                one = zero.bitwise_or(1 << target)
                self.register_buffer(
                    f"cry_zero_{control}_{offset}", zero, persistent=False
                )
                self.register_buffer(
                    f"cry_one_{control}_{offset}", one, persistent=False
                )

    @staticmethod
    def _rotate_pairs(state: Tensor, zero: Tensor, one: Tensor, theta: Tensor) -> Tensor:
        a0 = state.index_select(-1, zero)
        a1 = state.index_select(-1, one)
        half = theta * 0.5
        c, s = torch.cos(half), torch.sin(half)
        out = state.clone()
        out.index_copy_(-1, zero, c * a0 - s * a1)
        out.index_copy_(-1, one, s * a0 + c * a1)
        return out

    def forward(self, values: Tensor) -> Tensor:
        _require_cuda(values)
        original_shape = values.shape
        flat = values.reshape(-1, self.config.rank).float()
        norm = flat.square().sum(dim=-1, keepdim=True).clamp_min(self.config.eps).sqrt()
        state = flat / norm
        for layer in range(self.config.depth):
            for wire in range(self.config.num_qubits):
                state = self._rotate_pairs(
                    state,
                    getattr(self, f"ry_zero_{wire}"),
                    getattr(self, f"ry_one_{wire}"),
                    self.theta[layer, wire, 0],
                )
            if self.use_entanglers:
                for offset_index, offset in enumerate(self.config.entangling_offsets):
                    for control in range(self.config.num_qubits):
                        state = self._rotate_pairs(
                            state,
                            getattr(self, f"cry_zero_{control}_{offset}"),
                            getattr(self, f"cry_one_{control}_{offset}"),
                            self.theta[layer, control, 1 + offset_index],
                        )
        self.circuit_calls.add_(1)
        return (state * norm).reshape(original_shape).to(values.dtype)


class DequantizedAuditCore(_ParameterizedCore):
    """Independent tensor-axis implementation of the same RY/CRY circuit."""

    def _apply_ry_axis(self, state: Tensor, wire: int, theta: Tensor) -> Tensor:
        axis = 1 + (self.config.num_qubits - 1 - wire)
        moved = state.movedim(axis, -1)
        out = torch.matmul(moved, _ry_matrix(theta).transpose(0, 1))
        return out.movedim(-1, axis)

    def _apply_cry_axes(
        self, state: Tensor, control: int, target: int, theta: Tensor
    ) -> Tensor:
        control_axis = 1 + (self.config.num_qubits - 1 - control)
        target_axis = 1 + (self.config.num_qubits - 1 - target)
        moved = state.movedim((control_axis, target_axis), (-2, -1))
        original = moved.shape
        pairs = moved.reshape(*original[:-2], 4)
        gate = torch.eye(4, dtype=state.dtype, device=state.device)
        gate[2:, 2:] = _ry_matrix(theta).to(dtype=state.dtype, device=state.device)
        out = torch.matmul(pairs, gate.transpose(0, 1)).reshape(original)
        return out.movedim((-2, -1), (control_axis, target_axis))

    def forward(self, values: Tensor) -> Tensor:
        _require_cuda(values)
        original_shape = values.shape
        flat = values.reshape(-1, self.config.rank).float()
        norm = flat.square().sum(dim=-1, keepdim=True).clamp_min(self.config.eps).sqrt()
        state = (flat / norm).reshape(-1, *([2] * self.config.num_qubits))
        for layer in range(self.config.depth):
            for wire in range(self.config.num_qubits):
                state = self._apply_ry_axis(state, wire, self.theta[layer, wire, 0])
            for offset_index, offset in enumerate(self.config.entangling_offsets):
                for control in range(self.config.num_qubits):
                    state = self._apply_cry_axes(
                        state,
                        control,
                        (control + offset) % self.config.num_qubits,
                        self.theta[layer, control, 1 + offset_index],
                    )
        self.circuit_calls.add_(1)
        return (state.reshape(-1, self.config.rank) * norm).reshape(original_shape).to(
            values.dtype
        )


class EqualParameterClassicalCore(_ParameterizedCore):
    """Structured nonlinear classical control with exactly the same parameters.

    This is intentionally not a dense layer. Local terms provide bit-axis signed
    scalings; interaction terms couple XOR-neighbour features conditionally on a
    control bit. Zero parameters give the identity, matching the quantum arm's
    initialization and optimization budget.
    """

    def __init__(self, config: QH031Config) -> None:
        super().__init__(config)
        basis = torch.arange(config.rank, dtype=torch.long)
        signs = []
        for wire in range(config.num_qubits):
            signs.append(
                torch.where(
                    basis.bitwise_and(1 << wire) == 0,
                    torch.ones_like(basis, dtype=torch.float32),
                    -torch.ones_like(basis, dtype=torch.float32),
                )
            )
        self.register_buffer("bit_signs", torch.stack(signs), persistent=False)
        for control in range(config.num_qubits):
            control_sign = torch.where(
                basis.bitwise_and(1 << control) == 0,
                torch.zeros_like(basis, dtype=torch.float32),
                torch.ones_like(basis, dtype=torch.float32),
            )
            self.register_buffer(
                f"control_mask_{control}", control_sign, persistent=False
            )
            for offset in config.entangling_offsets:
                target = (control + offset) % config.num_qubits
                self.register_buffer(
                    f"xor_perm_{control}_{offset}",
                    basis.bitwise_xor(1 << target),
                    persistent=False,
                )

    def forward(self, values: Tensor) -> Tensor:
        _require_cuda(values)
        original_shape = values.shape
        state = values.reshape(-1, self.config.rank).float()
        scale = float(len(self.config.entangling_offsets) * self.config.num_qubits) ** 0.5
        for layer in range(self.config.depth):
            for wire in range(self.config.num_qubits):
                coefficient = torch.tanh(self.theta[layer, wire, 0])
                state = state + coefficient * state * self.bit_signs[wire]
            for offset_index, offset in enumerate(self.config.entangling_offsets):
                for control in range(self.config.num_qubits):
                    coefficient = torch.tanh(
                        self.theta[layer, control, 1 + offset_index]
                    )
                    neighbour = state.index_select(
                        -1, getattr(self, f"xor_perm_{control}_{offset}")
                    )
                    state = state + coefficient * neighbour * getattr(
                        self, f"control_mask_{control}"
                    ) / scale
        return state.reshape(original_shape).to(values.dtype)


class SharedInputBasis(nn.Module):
    def __init__(self, hidden_size: int, rank: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False, dtype=dtype)
        self.down.requires_grad_(False)


class SharedBasisValueReplacement(nn.Module):
    """One layer-specific output head attached to a shared basis and core."""

    def __init__(
        self,
        shared_basis: SharedInputBasis,
        core: _ParameterizedCore,
        output_size: int,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.shared_basis = shared_basis
        self.core = core
        self.up = nn.Linear(
            shared_basis.down.out_features, output_size, bias=False, dtype=dtype
        )
        self.up.requires_grad_(False)
        self.gamma = nn.Parameter(torch.ones((), dtype=torch.float32))
        self.branch_enabled = True
        self.capture_teacher = False
        self.register_buffer("teacher_output", torch.empty(0), persistent=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        projected = self.shared_basis.down(hidden_states)
        if self.capture_teacher:
            raise RuntimeError(
                "Teacher capture requires the original dense module wrapper; "
                "use QH031TeacherStudentController during distillation"
            )
        if self.branch_enabled:
            transformed = self.core(projected)
            projected = projected + self.gamma.to(projected.dtype) * (
                transformed - projected
            )
        return self.up(projected)


def build_shared_svd_replacements(
    dense_layers: Mapping[int, nn.Linear],
    config: QH031Config,
    *,
    branch: str = "quantum",
) -> Dict[int, SharedBasisValueReplacement]:
    """Construct QH031 modules from the joint SVD of two dense projections.

    For dense weights ``W_i`` with shape ``[output, hidden]``, QH031 computes a
    rank-``r`` SVD of ``cat([W_0, W_1], dim=0)``. The common ``Vh`` becomes the
    shared down projection and each row block of ``U*S`` becomes its frozen up
    projection. The returned modules share both the down projection and core.
    """

    config.validate()
    if tuple(sorted(dense_layers)) != tuple(sorted(config.layer_indices)):
        raise ValueError(
            f"dense layer keys {tuple(sorted(dense_layers))} do not match "
            f"configured layers {tuple(sorted(config.layer_indices))}"
        )
    ordered = [dense_layers[index] for index in config.layer_indices]
    device = ordered[0].weight.device
    _require_cuda(ordered[0].weight)
    dtype = ordered[0].weight.dtype
    for dense in ordered:
        if dense.bias is not None:
            raise ValueError("QH031 currently supports bias-free value projections")
        if dense.weight.shape != (config.output_size, config.hidden_size):
            raise ValueError(f"unexpected dense shape: {tuple(dense.weight.shape)}")
    stacked = torch.cat([dense.weight.detach().float() for dense in ordered], dim=0)
    u, s, vh = torch.linalg.svd(stacked, full_matrices=False)
    shared = SharedInputBasis(config.hidden_size, config.rank, dtype=dtype).to(device)
    shared.down.weight.copy_(vh[: config.rank].to(dtype))

    branch_classes = {
        "quantum": ExactQuantumCore,
        "dq": DequantizedAuditCore,
        "classical": EqualParameterClassicalCore,
        "no_ent": lambda cfg: ExactQuantumCore(cfg, use_entanglers=False),
    }
    if branch not in branch_classes:
        raise ValueError(f"unknown branch {branch!r}; choose from {tuple(branch_classes)}")
    core = branch_classes[branch](config).to(device)
    us = u[:, : config.rank] * s[: config.rank].unsqueeze(0)
    result: Dict[int, SharedBasisValueReplacement] = {}
    for position, layer_index in enumerate(config.layer_indices):
        replacement = SharedBasisValueReplacement(
            shared, core, config.output_size, dtype=dtype
        ).to(device)
        start = position * config.output_size
        stop = start + config.output_size
        replacement.up.weight.copy_(us[start:stop].to(dtype))
        result[layer_index] = replacement
    return result


def qh031_parameter_audit(config: QH031Config) -> MutableMapping[str, int]:
    config.validate()
    original = len(config.layer_indices) * config.hidden_size * config.output_size
    frozen_replacement = config.rank * config.hidden_size + (
        len(config.layer_indices) * config.output_size * config.rank
    )
    trainable = (
        config.depth
        * config.num_qubits
        * (1 + len(config.entangling_offsets))
        + len(config.layer_indices)
    )
    return {
        "original_parameters_removed": original,
        "frozen_replacement_parameters": frozen_replacement,
        "trainable_parameters": trainable,
        "total_replacement_parameters": frozen_replacement + trainable,
        "net_parameter_reduction": original - frozen_replacement - trainable,
    }


def install_qh031(
    model: nn.Module,
    config: QH031Config = QH031Config(),
    *,
    branch: str = "quantum",
):
    """Freeze a Qwen3.8 model and replace both configured ``v_proj`` layers.

    Returns ``(replacements, originals, audit)``.  ``originals`` is intentionally
    returned only to support teacher forwards during training and must not be
    serialized into a deployed checkpoint.
    """

    config.validate()
    model.requires_grad_(False)
    language_model = model.model.language_model
    originals: Dict[int, nn.Linear] = {}
    for layer_index in config.layer_indices:
        layer = language_model.layers[layer_index]
        if getattr(layer, "block_type", None) != "full_attention":
            raise ValueError(f"layer {layer_index} is not a full-attention layer")
        projection = layer.self_attn.v_proj
        if not isinstance(projection, nn.Linear):
            raise TypeError(
                f"layer {layer_index} v_proj must be nn.Linear before QH031 install"
            )
        originals[layer_index] = projection
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    replacements = build_shared_svd_replacements(originals, config, branch=branch)
    for layer_index, replacement in replacements.items():
        language_model.layers[layer_index].self_attn.v_proj = replacement
    deployed_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    expected = qh031_parameter_audit(config)
    if base_parameters - deployed_parameters != expected["net_parameter_reduction"]:
        raise AssertionError(
            "deployed model reduction differs from the unique-parameter audit: "
            f"observed={base_parameters - deployed_parameters}, "
            f"expected={expected['net_parameter_reduction']}"
        )
    if trainable_parameters != expected["trainable_parameters"]:
        raise AssertionError(
            f"trainable mismatch: observed={trainable_parameters}, "
            f"expected={expected['trainable_parameters']}"
        )
    audit = {
        **expected,
        "net_model_parameter_reduction": expected["net_parameter_reduction"],
        "trainable_parameter_count": trainable_parameters,
        "branch": branch,
        "layer_indices": list(config.layer_indices),
        "rank": config.rank,
        "num_qubits": config.num_qubits,
        "depth": config.depth,
        "entangling_offsets": list(config.entangling_offsets),
        "base_model_parameters": base_parameters,
        "deployed_model_parameters": deployed_parameters,
        "observed_net_model_parameter_reduction": base_parameters - deployed_parameters,
        "observed_trainable_parameter_count": trainable_parameters,
        "shared_input_basis": True,
        "shared_quantum_or_control_core": True,
        "gpu_exact_statevector_required": branch in {"quantum", "dq", "no_ent"},
        "teacher_dense_modules_are_deployment_parameters": False,
    }
    return replacements, originals, audit


def unique_trainable_parameters(
    replacements: Mapping[int, SharedBasisValueReplacement],
) -> Iterable[nn.Parameter]:
    """Yield shared parameters once, despite their appearance in two modules."""

    seen = set()
    for replacement in replacements.values():
        for parameter in replacement.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                yield parameter
