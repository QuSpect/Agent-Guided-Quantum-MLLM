"""QH034: simulator-resident quantum modulation of shared SVD coordinates.

The QH031 shared-basis replacement removes more than three million parameters
from two Qwen3.8 value projections, but its arbitrary latent rotation did not
survive an independent confirmation. QH034 keeps the frozen joint-SVD backbone
and makes a narrower intervention: a 10-qubit circuit generates 1024 positive,
mean-one scale factors for the shared singular coordinates.

The circuit starts in the uniform state. Zero angles therefore generate an
all-ones scale and exactly reproduce the shared-SVD backbone. Both training and
inference execute the exact CUDA statevector generator. The independent DQ
branch uses tensor-axis gates, while the equal-parameter classical branch uses
80 fixed Walsh spectral modes followed by the same positive mean-one constraint.
Exact simulation remains classically reproducible; no speedup is claimed.
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
class QH034Config:
    hidden_size: int = 5120
    output_size: int = 1024
    rank: int = 1024
    layer_indices: Tuple[int, int] = (55, 59)
    num_qubits: int = 10
    depth: int = 2
    entangling_offsets: Tuple[int, ...] = (1, 2, 4)

    def validate(self) -> None:
        if self.rank != 1 << self.num_qubits:
            raise ValueError("rank must equal 2**num_qubits")
        if len(self.layer_indices) != 2:
            raise ValueError("QH034 currently requires exactly two layers")
        if self.depth < 1:
            raise ValueError("depth must be positive")
        if any(not 0 < offset < self.num_qubits for offset in self.entangling_offsets):
            raise ValueError("invalid entangling offset")

    @property
    def circuit_parameter_count(self) -> int:
        return self.depth * self.num_qubits * (1 + len(self.entangling_offsets))


def _require_cuda(x: Tensor) -> None:
    if x.device.type != "cuda":
        raise RuntimeError(f"QH034 exact simulation is GPU-only; received {x.device}")


def _ry_matrix(theta: Tensor) -> Tensor:
    half = theta * 0.5
    c, s = torch.cos(half), torch.sin(half)
    return torch.stack((torch.stack((c, -s)), torch.stack((s, c))))


class _SpectralCore(nn.Module):
    def __init__(self, config: QH034Config) -> None:
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


class ExactQuantumSpectralCore(_SpectralCore):
    """Paired-index exact real-statevector spectral generator."""

    def __init__(self, config: QH034Config, use_entanglers: bool = True) -> None:
        super().__init__(config)
        self.use_entanglers = use_entanglers
        basis = torch.arange(config.rank, dtype=torch.long)
        for wire in range(config.num_qubits):
            zero = basis[(basis.bitwise_and(1 << wire)) == 0]
            one = zero.bitwise_or(1 << wire)
            self.register_buffer(f"zero_{wire}", zero, persistent=False)
            self.register_buffer(f"one_{wire}", one, persistent=False)
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
    def _rotate(state: Tensor, zero: Tensor, one: Tensor, theta: Tensor) -> Tensor:
        a0 = state.index_select(-1, zero)
        a1 = state.index_select(-1, one)
        half = theta * 0.5
        c, s = torch.cos(half), torch.sin(half)
        out = state.clone()
        out.index_copy_(-1, zero, c * a0 - s * a1)
        out.index_copy_(-1, one, s * a0 + c * a1)
        return out

    def _scales(self, device: torch.device) -> Tensor:
        state = torch.full(
            (self.config.rank,),
            self.config.rank ** -0.5,
            device=device,
            dtype=torch.float32,
        )
        for layer in range(self.config.depth):
            for wire in range(self.config.num_qubits):
                state = self._rotate(
                    state,
                    getattr(self, f"zero_{wire}"),
                    getattr(self, f"one_{wire}"),
                    self.theta[layer, wire, 0],
                )
            if self.use_entanglers:
                for offset_index, offset in enumerate(self.config.entangling_offsets):
                    for control in range(self.config.num_qubits):
                        state = self._rotate(
                            state,
                            getattr(self, f"cry_zero_{control}_{offset}"),
                            getattr(self, f"cry_one_{control}_{offset}"),
                            self.theta[layer, control, 1 + offset_index],
                        )
        return self.config.rank * state.square()

    def forward(self, values: Tensor) -> Tensor:
        _require_cuda(values)
        scale = self._scales(values.device)
        self.circuit_calls.add_(1)
        return (values.float() * scale).to(values.dtype)


class DequantizedSpectralCore(_SpectralCore):
    """Independent tensor-axis implementation of the same spectral circuit."""

    def __init__(self, config: QH034Config, use_entanglers: bool = True) -> None:
        super().__init__(config)
        self.use_entanglers = use_entanglers

    def _single(self, state: Tensor, wire: int, theta: Tensor) -> Tensor:
        axis = self.config.num_qubits - 1 - wire
        moved = state.movedim(axis, -1)
        return torch.matmul(moved, _ry_matrix(theta).transpose(0, 1)).movedim(-1, axis)

    def _controlled(
        self, state: Tensor, control: int, target: int, theta: Tensor
    ) -> Tensor:
        control_axis = self.config.num_qubits - 1 - control
        target_axis = self.config.num_qubits - 1 - target
        moved = state.movedim((control_axis, target_axis), (-2, -1))
        original_shape = moved.shape
        pairs = moved.reshape(*original_shape[:-2], 4)
        gate = torch.eye(4, dtype=state.dtype, device=state.device)
        gate[2:, 2:] = _ry_matrix(theta).to(state)
        acted = torch.matmul(pairs, gate.transpose(0, 1)).reshape(original_shape)
        return acted.movedim((-2, -1), (control_axis, target_axis))

    def _scales(self, device: torch.device) -> Tensor:
        state = torch.full(
            [2] * self.config.num_qubits,
            self.config.rank ** -0.5,
            device=device,
            dtype=torch.float32,
        )
        for layer in range(self.config.depth):
            for wire in range(self.config.num_qubits):
                state = self._single(state, wire, self.theta[layer, wire, 0])
            if self.use_entanglers:
                for offset_index, offset in enumerate(self.config.entangling_offsets):
                    for control in range(self.config.num_qubits):
                        state = self._controlled(
                            state,
                            control,
                            (control + offset) % self.config.num_qubits,
                            self.theta[layer, control, 1 + offset_index],
                        )
        return self.config.rank * state.reshape(self.config.rank).square()

    def forward(self, values: Tensor) -> Tensor:
        _require_cuda(values)
        scale = self._scales(values.device)
        self.circuit_calls.add_(1)
        return (values.float() * scale).to(values.dtype)


class EqualParameterClassicalSpectralCore(_SpectralCore):
    """80-parameter Walsh-basis positive spectral control."""

    def __init__(self, config: QH034Config) -> None:
        super().__init__(config)
        if config.circuit_parameter_count >= config.rank:
            raise ValueError("classical Walsh control requires parameters < rank")
        indices = torch.arange(config.rank, dtype=torch.long)
        modes = []
        for mode in range(1, config.circuit_parameter_count + 1):
            parity = torch.zeros(config.rank, dtype=torch.long)
            masked = indices.bitwise_and(mode)
            for bit in range(config.num_qubits):
                parity = parity.bitwise_xor(masked.bitwise_right_shift(bit).bitwise_and(1))
            modes.append(torch.where(parity == 0, 1.0, -1.0))
        self.register_buffer("walsh_modes", torch.stack(modes), persistent=False)

    def forward(self, values: Tensor) -> Tensor:
        _require_cuda(values)
        coefficients = torch.tanh(self.theta.reshape(-1))
        logits = torch.matmul(coefficients, self.walsh_modes) / (
            float(self.config.circuit_parameter_count) ** 0.5
        )
        scale = torch.softmax(logits, dim=-1) * self.config.rank
        return (values.float() * scale).to(values.dtype)


def build_qh034_replacements(
    dense_layers: Mapping[int, nn.Linear],
    config: QH034Config,
    *,
    branch: str = "quantum",
) -> Dict[int, SharedBasisValueReplacement]:
    config.validate()
    if tuple(dense_layers) != config.layer_indices:
        raise ValueError("dense layer order must match QH034 layer_indices")
    ordered = [dense_layers[index] for index in config.layer_indices]
    device, dtype = ordered[0].weight.device, ordered[0].weight.dtype
    _require_cuda(ordered[0].weight)
    for dense in ordered:
        if dense.bias is not None:
            raise ValueError("QH034 supports bias-free value projections")
        if dense.weight.shape != (config.output_size, config.hidden_size):
            raise ValueError(f"unexpected dense shape: {tuple(dense.weight.shape)}")
    stacked = torch.cat([layer.weight.detach().float() for layer in ordered], dim=0)
    u, s, vh = torch.linalg.svd(stacked, full_matrices=False)
    shared = SharedInputBasis(config.hidden_size, config.rank, dtype=dtype).to(device)
    shared.down.weight.copy_(vh[: config.rank].to(dtype))
    classes = {
        "quantum": ExactQuantumSpectralCore,
        "dq": DequantizedSpectralCore,
        "classical": EqualParameterClassicalSpectralCore,
        "no_ent": lambda cfg: ExactQuantumSpectralCore(cfg, use_entanglers=False),
    }
    if branch not in classes:
        raise ValueError(f"unknown QH034 branch: {branch}")
    core = classes[branch](config).to(device)
    us = u[:, : config.rank] * s[: config.rank].unsqueeze(0)
    result: Dict[int, SharedBasisValueReplacement] = {}
    for position, layer_index in enumerate(config.layer_indices):
        replacement = SharedBasisValueReplacement(
            shared, core, config.output_size, dtype=dtype
        ).to(device)
        start = position * config.output_size
        replacement.up.weight.copy_(
            us[start : start + config.output_size].to(dtype)
        )
        result[layer_index] = replacement
    return result


def qh034_parameter_audit(config: QH034Config) -> MutableMapping[str, int]:
    config.validate()
    removed = len(config.layer_indices) * config.hidden_size * config.output_size
    frozen = config.rank * config.hidden_size + (
        len(config.layer_indices) * config.output_size * config.rank
    )
    trainable = config.circuit_parameter_count + len(config.layer_indices)
    return {
        "original_parameters_removed": removed,
        "frozen_replacement_parameters": frozen,
        "trainable_parameters": trainable,
        "total_replacement_parameters": frozen + trainable,
        "net_parameter_reduction": removed - frozen - trainable,
        "spectral_scale_count": config.rank,
    }


def install_qh034(
    model: nn.Module,
    config: QH034Config = QH034Config(),
    *,
    branch: str = "quantum",
):
    config.validate()
    model.requires_grad_(False)
    language_model = model.model.language_model
    originals = {}
    for index in config.layer_indices:
        projection = language_model.layers[index].self_attn.v_proj
        if not isinstance(projection, nn.Linear):
            raise TypeError(f"layer {index} v_proj is not dense before QH034 install")
        originals[index] = projection
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    replacements = build_qh034_replacements(originals, config, branch=branch)
    for index, replacement in replacements.items():
        language_model.layers[index].self_attn.v_proj = replacement
    deployed = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    audit = qh034_parameter_audit(config)
    if base_parameters - deployed != audit["net_parameter_reduction"]:
        raise AssertionError("QH034 deployed-parameter audit mismatch")
    if trainable != audit["trainable_parameters"]:
        raise AssertionError("QH034 trainable-parameter audit mismatch")
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
        "depth": config.depth,
        "entangling_offsets": list(config.entangling_offsets),
        "runtime_scale_generation": True,
        "gpu_exact_statevector_required": branch in {"quantum", "dq", "no_ent"},
    }


def unique_trainable_parameters(replacements: Mapping[int, nn.Module]):
    seen = set()
    for replacement in replacements.values():
        for parameter in replacement.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                yield parameter
