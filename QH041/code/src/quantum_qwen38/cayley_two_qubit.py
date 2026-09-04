"""QH-010: exact GPU-simulated two-qubit Cayley adapters for Qwen3.8."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class QH010Config:
    hidden_size: int = 5120
    block_size: int = 4
    layer_index: int = 7
    target_projection: str = "v_proj"
    # Exact identity is essential: before training the wrapped Qwen3.8 must be bitwise-stable.
    init_std: float = 0.0


def _skew_from_six(theta: torch.Tensor) -> torch.Tensor:
    """Map six coordinates per block to a 4x4 real skew-symmetric generator."""
    matrix = theta.new_zeros(theta.shape[0], 4, 4)
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for coordinate, (row, column) in enumerate(pairs):
        matrix[:, row, column] = theta[:, coordinate]
        matrix[:, column, row] = -theta[:, coordinate]
    return matrix


def _symmetric_from_six(theta: torch.Tensor) -> torch.Tensor:
    """Equal-parameter non-unitary classical control."""
    matrix = theta.new_zeros(theta.shape[0], 4, 4)
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for coordinate, (row, column) in enumerate(pairs):
        value = theta[:, coordinate]
        matrix[:, row, column] = value
        matrix[:, column, row] = value
    return matrix


def _single_qubit_zyz(angles: torch.Tensor) -> torch.Tensor:
    """Return batched SU(2) matrices in complex64 from Z-Y-Z Euler angles."""
    alpha, beta, gamma = angles.unbind(dim=-1)
    half_beta = beta * 0.5
    cosine = torch.cos(half_beta)
    sine = torch.sin(half_beta)
    phase_sum = torch.polar(torch.ones_like(alpha), -0.5 * (alpha + gamma))
    phase_diff = torch.polar(torch.ones_like(alpha), -0.5 * (alpha - gamma))
    output = torch.empty(*angles.shape[:-1], 2, 2, dtype=torch.complex64, device=angles.device)
    output[..., 0, 0] = phase_sum * cosine
    output[..., 0, 1] = -phase_diff * sine
    output[..., 1, 0] = phase_diff.conj() * sine
    output[..., 1, 1] = phase_sum.conj() * cosine
    return output


class TwoQubitBlockTransform(nn.Module):
    """Blockwise simulator/control sharing exactly six trainable values per block."""

    def __init__(self, config: QH010Config | None = None, core_kind: str = "quantum") -> None:
        super().__init__()
        self.config = config or QH010Config()
        if self.config.block_size != 4:
            raise ValueError("QH-010 currently implements exact two-qubit (4-amplitude) blocks")
        if self.config.hidden_size % self.config.block_size:
            raise ValueError("hidden_size must be divisible by block_size")
        if core_kind not in {"quantum", "dequantized", "classical", "no_entanglement"}:
            raise ValueError(f"unsupported core_kind: {core_kind}")
        self.core_kind = core_kind
        self.n_blocks = self.config.hidden_size // self.config.block_size
        self.theta = nn.Parameter(torch.empty(self.n_blocks, 6, dtype=torch.float32))
        nn.init.normal_(self.theta, mean=0.0, std=self.config.init_std)
        self.register_buffer("identity4", torch.eye(4, dtype=torch.float32), persistent=False)

    def _quantum_unitary(self) -> torch.Tensor:
        generator = _skew_from_six(self.theta)
        identity = self.identity4.expand(self.n_blocks, -1, -1)
        left = identity - 0.5 * generator
        right = identity + 0.5 * generator
        return torch.linalg.solve(right.transpose(-1, -2), left.transpose(-1, -2)).transpose(-1, -2)

    def _classical_matrix(self) -> torch.Tensor:
        identity = self.identity4.expand(self.n_blocks, -1, -1)
        return identity + _symmetric_from_six(self.theta)

    def _separable_unitary(self) -> torch.Tensor:
        first = _single_qubit_zyz(self.theta[:, :3])
        second = _single_qubit_zyz(self.theta[:, 3:])
        return torch.einsum("bij,bkl->bikjl", first, second).reshape(self.n_blocks, 4, 4)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hidden_states.is_cuda:
            raise RuntimeError("QH-010 simulation and controls are GPU-only")
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected hidden size {self.config.hidden_size}, got {hidden_states.shape[-1]}"
            )
        original_shape = hidden_states.shape
        blocks = hidden_states.reshape(-1, self.n_blocks, 4).float()
        if self.core_kind == "classical":
            transformed = torch.einsum("bij,nbj->nbi", self._classical_matrix(), blocks)
        elif self.core_kind == "dequantized":
            # Exact classical realization of the same two-qubit Cayley map. This is
            # a mandatory dequantization control, not a separate hypothesis class.
            amplitudes = torch.einsum("bij,nbj->nbi", self._quantum_unitary(), blocks)
            transformed = amplitudes.abs() * torch.sign(blocks)
        else:
            norm = torch.linalg.vector_norm(blocks, dim=-1, keepdim=True)
            normalized = torch.where(norm > 0.0, blocks / norm.clamp_min(1.0e-12), blocks)
            unitary = (
                self._quantum_unitary().to(torch.complex64)
                if self.core_kind == "quantum"
                else self._separable_unitary()
            )
            state = torch.einsum("bij,nbj->nbi", unitary, normalized.to(torch.complex64))
            # Exact statevector measurement reconstruction used by the cited CUA method.
            transformed = state.abs() * torch.sign(blocks) * norm
        return transformed.reshape(original_shape).to(hidden_states.dtype)

    def audit(self) -> dict:
        return {
            "config": asdict(self.config),
            "core_kind": self.core_kind,
            "n_two_qubit_blocks": self.n_blocks,
            "parameters_per_block": 6,
            "trainable_parameters": self.theta.numel(),
            "simulator": "exact CUDA statevector; complex64 amplitudes, analytic Cayley SO(4)",
            "measurement": "amplitude magnitude reconstruction with input sign correction",
        }


class ProjectionInputAdapter(nn.Module):
    """Apply a trainable block transform immediately before a frozen projection."""

    def __init__(self, frozen_projection: nn.Module, transform: TwoQubitBlockTransform) -> None:
        super().__init__()
        self.frozen_projection = frozen_projection
        self.transform = transform
        for parameter in self.frozen_projection.parameters():
            parameter.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.frozen_projection(self.transform(hidden_states))


class VisualMergerOutputAdapter(nn.Module):
    """Apply a block transform only to visual-prefill features after the frozen merger."""

    def __init__(self, frozen_merger: nn.Module, transform: TwoQubitBlockTransform) -> None:
        super().__init__()
        self.frozen_merger = frozen_merger
        self.transform = transform
        for parameter in self.frozen_merger.parameters():
            parameter.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.transform(self.frozen_merger(hidden_states))


def freeze_and_inject_qh010(
    model: nn.Module,
    core_kind: str = "quantum",
    config: QH010Config | None = None,
) -> dict:
    """Freeze Qwen3.8 and insert QH-010 before layer-7 full-attention v_proj."""
    config = config or QH010Config()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    language_model = model.model.language_model
    layer = language_model.layers[config.layer_index]
    if getattr(layer, "block_type", None) != "full_attention":
        raise ValueError(f"layer {config.layer_index} is not a full-attention layer")
    projection = getattr(layer.self_attn, config.target_projection)
    reference = next(projection.parameters())
    transform = TwoQubitBlockTransform(config, core_kind=core_kind).to(reference.device)
    setattr(
        layer.self_attn,
        config.target_projection,
        ProjectionInputAdapter(projection, transform),
    )
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    candidate = {
        "quantum": "QH-010",
        "dequantized": "DQ-010",
        "classical": "CC-010",
        "no_entanglement": "QH-010-no-ent",
    }[core_kind]
    return {
        "candidate": candidate,
        "parent": "Cayley Unitery Adapter, arXiv:2605.05914; adapted to Qwen3.8",
        "injection_path": (
            f"model.model.language_model.layers.{config.layer_index}.self_attn."
            f"{config.target_projection}.input"
        ),
        "adapter": transform.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }


def freeze_and_inject_qh011(
    model: nn.Module,
    core_kind: str = "quantum",
    config: QH010Config | None = None,
) -> dict:
    """Freeze Qwen3.8 and transform visual merger outputs only during prefill."""
    config = config or QH010Config()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.model.visual
    merger = visual.merger
    reference = next(merger.parameters())
    transform = TwoQubitBlockTransform(config, core_kind=core_kind).to(reference.device)
    visual.merger = VisualMergerOutputAdapter(merger, transform)
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    candidate = {
        "quantum": "QH-011",
        "dequantized": "DQ-011",
        "classical": "CC-011",
        "no_entanglement": "QH-011-no-ent",
    }[core_kind]
    return {
        "candidate": candidate,
        "parent": "QH-010 placement mutation",
        "injection_path": "model.model.visual.merger.output",
        "execution_scope": "visual prefill only; no text-only or decode-token invocation",
        "adapter": transform.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
