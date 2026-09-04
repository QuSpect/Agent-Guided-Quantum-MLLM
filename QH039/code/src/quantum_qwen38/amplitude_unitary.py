"""Real-amplitude quantum unitary adapters evolved for Qwen3.8's visual merger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log2

import torch
from torch import nn


@dataclass(frozen=True)
class AmplitudeUnitaryConfig:
    width: int
    register_widths: tuple[int, ...]
    depth: int = 2
    permutation_seed: int = 20260828

    def __post_init__(self) -> None:
        if sum(self.register_widths) != self.width:
            raise ValueError("register widths must exactly cover width")
        if self.depth < 1:
            raise ValueError("depth must be positive")
        for value in self.register_widths:
            if value < 4 or value & (value - 1):
                raise ValueError("each register width must be a power of two >= 4")


class RealAmplitudeRegister(nn.Module):
    """Exact differentiable statevector circuit restricted to real amplitudes.

    Each layer applies trainable local RY gates followed by identity-initialized
    CNOT-RY(phi)-CNOT entanglers on a ring.  Both angle sets are zero-initialized,
    so inserting the layer cannot perturb the frozen backbone before training.
    """

    def __init__(self, width: int, depth: int = 2) -> None:
        super().__init__()
        if width < 4 or width & (width - 1):
            raise ValueError("width must be a power of two >= 4")
        self.width = width
        self.n_qubits = int(log2(width))
        self.depth = depth
        self.local_angles = nn.Parameter(torch.zeros(depth, self.n_qubits))
        self.entangling_angles = nn.Parameter(torch.zeros(depth, self.n_qubits))

        basis = torch.arange(width, dtype=torch.long)
        for wire in range(self.n_qubits):
            self.register_buffer(
                f"wire_{wire}_zero",
                basis[(basis & (1 << wire)) == 0],
                persistent=False,
            )
            self.register_buffer(
                f"wire_{wire}_one",
                basis[(basis & (1 << wire)) != 0],
                persistent=False,
            )
        for control in range(self.n_qubits):
            target = (control + 1) % self.n_qubits
            permutation = basis.clone()
            active = (basis & (1 << control)) != 0
            permutation[active] ^= 1 << target
            self.register_buffer(
                f"cnot_{control}_{target}",
                permutation,
                persistent=False,
            )

    @staticmethod
    def _expand_angle(angle: torch.Tensor, target_ndim: int) -> torch.Tensor:
        while angle.ndim < target_ndim:
            angle = angle.unsqueeze(-1)
        return angle

    def _apply_ry(
        self,
        state: torch.Tensor,
        angle: torch.Tensor,
        wire: int,
    ) -> torch.Tensor:
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

    def _apply_cnot(self, state: torch.Tensor, control: int) -> torch.Tensor:
        target = (control + 1) % self.n_qubits
        permutation = getattr(self, f"cnot_{control}_{target}")
        return state.index_select(1, permutation)

    def forward(self, amplitudes: torch.Tensor) -> torch.Tensor:
        if not amplitudes.is_cuda:
            raise RuntimeError("quantum statevector simulation must run on CUDA")
        if amplitudes.ndim != 2 or amplitudes.shape[-1] != self.width:
            raise ValueError(f"expected [batch, {self.width}], got {tuple(amplitudes.shape)}")
        values = amplitudes.float()
        norm = values.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        state = values / norm
        for layer in range(self.depth):
            for wire in range(self.n_qubits):
                state = self._apply_ry(state, self.local_angles[layer, wire], wire)
            for control in range(self.n_qubits):
                target = (control + 1) % self.n_qubits
                state = self._apply_cnot(state, control)
                state = self._apply_ry(
                    state,
                    self.entangling_angles[layer, control],
                    target,
                )
                state = self._apply_cnot(state, control)
        return state * norm


class BlockAmplitudeUnitary(nn.Module):
    """Full-width unitary transform assembled from exact power-of-two registers."""

    def __init__(self, config: AmplitudeUnitaryConfig) -> None:
        super().__init__()
        self.config = config
        self.registers = nn.ModuleList(
            RealAmplitudeRegister(width, config.depth)
            for width in config.register_widths
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.permutation_seed)
        permutation = torch.randperm(config.width, generator=generator)
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", torch.argsort(permutation))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        if original_shape[-1] != self.config.width:
            raise ValueError(
                f"expected hidden width {self.config.width}, got {original_shape[-1]}"
            )
        flat = hidden_states.reshape(-1, self.config.width)
        permuted = flat.index_select(-1, self.permutation)
        pieces = torch.split(permuted, self.config.register_widths, dim=-1)
        transformed = torch.cat(
            [register(piece) for register, piece in zip(self.registers, pieces, strict=True)],
            dim=-1,
        )
        restored = transformed.index_select(-1, self.inverse_permutation)
        return restored.reshape(original_shape).to(hidden_states.dtype)

    def audit(self) -> dict:
        return {
            "config": asdict(self.config),
            "register_qubits": [register.n_qubits for register in self.registers],
            "state_dtype": "torch.float32 real-amplitude statevector",
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
            "entangler": "CNOT-RY(phi)-CNOT ring",
            "identity_initialized": True,
        }


class QuantumUnitaryMergerWrapper(nn.Module):
    """Optionally apply quantum unitaries before and/or after Qwen's frozen merger."""

    def __init__(
        self,
        frozen_merger: nn.Module,
        pre_unitary: BlockAmplitudeUnitary | None,
        post_unitary: BlockAmplitudeUnitary,
    ) -> None:
        super().__init__()
        self.frozen_merger = frozen_merger
        self.pre_unitary = pre_unitary
        self.post_unitary = post_unitary
        for parameter in frozen_merger.parameters():
            parameter.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.pre_unitary is not None:
            hidden_states = self.pre_unitary(hidden_states)
        merged = self.frozen_merger(hidden_states)
        return self.post_unitary(merged)


def freeze_and_inject_qh006(
    model: nn.Module,
    *,
    mode: str = "post",
    depth: int = 2,
) -> dict:
    """Inject QH-006e (post) or QH-006d (two-sided visual sandwich)."""
    if mode not in {"post", "sandwich"}:
        raise ValueError("mode must be 'post' or 'sandwich'")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.model.visual
    old_merger = visual.merger
    reference = next(old_merger.parameters())
    pre = None
    if mode == "sandwich":
        pre = BlockAmplitudeUnitary(
            AmplitudeUnitaryConfig(
                width=1152,
                register_widths=(1024, 128),
                depth=depth,
                permutation_seed=1152,
            )
        ).to(reference.device)
    post = BlockAmplitudeUnitary(
        AmplitudeUnitaryConfig(
            width=5120,
            register_widths=(4096, 1024),
            depth=depth,
            permutation_seed=20260828,
        )
    ).to(reference.device)
    visual.merger = QuantumUnitaryMergerWrapper(old_merger, pre, post)
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    return {
        "candidate": "QH-006d" if mode == "sandwich" else "QH-006e",
        "mode": mode,
        "injection_path": "model.model.visual.merger",
        "pre_unitary": None if pre is None else pre.audit(),
        "post_unitary": post.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
