"""QH-024 true low-rank-plus-VQC replacement for Qwen3.8 ``v_proj``.

The pretrained dense projection is removed from the deployed parameter set. A
frozen SVD backbone preserves most of the map and a HyQuT-inspired circuit
learns a nonlinear residual. The dense teacher is only a non-persistent buffer
for distillation and can be stripped before evaluation/export.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class QH024Config:
    hidden_size: int = 5120
    output_size: int = 1024
    svd_rank: int = 512
    n_qubits: int = 10
    depth: int = 2
    layer_index: int = 7
    target_projection: str = "v_proj"
    gamma_init: float = 0.1
    parameter_seed: int = 20260838

    def __post_init__(self) -> None:
        if self.svd_rank < 1 or self.svd_rank > min(self.hidden_size, self.output_size):
            raise ValueError("invalid SVD rank")
        if self.n_qubits < 2 or self.depth < 1:
            raise ValueError("invalid quantum circuit dimensions")


class HyQuTStatevectorCore(nn.Module):
    """Batched H + data RY/RZ + trainable RZ/RY/RZ + circular CNOT."""

    def __init__(self, n_qubits: int, depth: int, *, seed: int, entangle: bool = True) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.width = 1 << n_qubits
        self.entangle = entangle
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        theta = torch.empty(depth, n_qubits, 3, dtype=torch.float32)
        theta.uniform_(-0.02, 0.02, generator=generator)
        self.theta = nn.Parameter(theta)
        basis = torch.arange(self.width, dtype=torch.long)
        z_signs = []
        for wire in range(n_qubits):
            zero = basis[(basis & (1 << wire)) == 0]
            self.register_buffer(f"zero_{wire}", zero, persistent=False)
            self.register_buffer(f"one_{wire}", zero | (1 << wire), persistent=False)
            z_signs.append(1.0 - 2.0 * ((basis >> wire) & 1).to(torch.float32))
        self.register_buffer("z_signs", torch.stack(z_signs, dim=1), persistent=False)
        for control in range(n_qubits):
            target = (control + 1) % n_qubits
            active = ((basis >> control) & 1).to(torch.long)
            self.register_buffer(
                f"cnot_{control}_{target}", basis ^ (active << target), persistent=False
            )
        self.circuit_calls = 0

    @staticmethod
    def _columns(angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cosine, sine = torch.cos(angle * 0.5), torch.sin(angle * 0.5)
        return (cosine, sine) if angle.ndim == 0 else (cosine[:, None], sine[:, None])

    def _apply_ry(self, state: torch.Tensor, angle: torch.Tensor, wire: int) -> torch.Tensor:
        zero, one = getattr(self, f"zero_{wire}"), getattr(self, f"one_{wire}")
        a0, a1 = state.index_select(1, zero), state.index_select(1, one)
        cosine, sine = self._columns(angle)
        output = state.clone()
        output[:, zero] = cosine * a0 - sine * a1
        output[:, one] = sine * a0 + cosine * a1
        return output

    def _apply_rz(self, state: torch.Tensor, angle: torch.Tensor, wire: int) -> torch.Tensor:
        zero, one = getattr(self, f"zero_{wire}"), getattr(self, f"one_{wire}")
        if angle.ndim == 0:
            p0, p1 = torch.exp(-0.5j * angle), torch.exp(0.5j * angle)
        else:
            p0, p1 = torch.exp(-0.5j * angle)[:, None], torch.exp(0.5j * angle)[:, None]
        output = state.clone()
        output[:, zero] = state.index_select(1, zero) * p0
        output[:, one] = state.index_select(1, one) * p1
        return output

    def _apply_cnot(self, state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        if not self.entangle:
            return state
        return state.index_select(1, getattr(self, f"cnot_{control}_{target}"))

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        if not encoded.is_cuda or not self.theta.is_cuda:
            raise RuntimeError("QH-024 exact statevector simulation is CUDA-only")
        if encoded.ndim != 2 or encoded.shape[1] != 2 * self.n_qubits:
            raise ValueError("encoded feature shape mismatch")
        state = torch.full(
            (encoded.shape[0], self.width),
            1.0 / math.sqrt(self.width),
            device=encoded.device,
            dtype=torch.complex64,
        )
        data_angles = math.pi * torch.sigmoid(encoded.float())
        for wire in range(self.n_qubits):
            state = self._apply_ry(state, data_angles[:, wire], wire)
            state = self._apply_rz(state, data_angles[:, self.n_qubits + wire], wire)
        for layer in range(self.depth):
            for wire in range(self.n_qubits):
                state = self._apply_rz(state, self.theta[layer, wire, 0], wire)
                state = self._apply_ry(state, self.theta[layer, wire, 1], wire)
                state = self._apply_rz(state, self.theta[layer, wire, 2], wire)
            for control in range(self.n_qubits):
                state = self._apply_cnot(state, control, (control + 1) % self.n_qubits)
        probabilities = state.real.square() + state.imag.square()
        self.circuit_calls += 1
        return probabilities @ self.z_signs


class TensorContractionHyQuTCore(HyQuTStatevectorCore):
    """Independent exact tensor-axis realization for dequantization audit."""

    def _apply_ry(self, state: torch.Tensor, angle: torch.Tensor, wire: int) -> torch.Tensor:
        cosine, sine = torch.cos(angle * 0.5), torch.sin(angle * 0.5)
        if angle.ndim == 0:
            matrix = torch.stack([
                torch.stack([cosine, -sine]), torch.stack([sine, cosine])
            ]).to(torch.complex64)
        else:
            matrix = torch.stack([
                torch.stack([cosine, -sine], dim=-1),
                torch.stack([sine, cosine], dim=-1),
            ], dim=1).to(torch.complex64)
        tensor = state.reshape(state.shape[0], *([2] * self.n_qubits))
        axis = 1 + self.n_qubits - 1 - wire
        moved = tensor.movedim(axis, -1)
        updated = (
            torch.einsum("b...j,ij->b...i", moved, matrix)
            if angle.ndim == 0
            else torch.einsum("b...j,bij->b...i", moved, matrix)
        )
        return updated.movedim(-1, axis).reshape_as(state)

    def _apply_rz(self, state: torch.Tensor, angle: torch.Tensor, wire: int) -> torch.Tensor:
        diagonal = (
            torch.stack([torch.exp(-0.5j * angle), torch.exp(0.5j * angle)])
            if angle.ndim == 0
            else torch.stack([torch.exp(-0.5j * angle), torch.exp(0.5j * angle)], dim=1)
        )
        tensor = state.reshape(state.shape[0], *([2] * self.n_qubits))
        axis = 1 + self.n_qubits - 1 - wire
        moved = tensor.movedim(axis, -1)
        updated = (
            torch.einsum("b...j,j->b...j", moved, diagonal)
            if angle.ndim == 0
            else torch.einsum("b...j,bj->b...j", moved, diagonal)
        )
        return updated.movedim(-1, axis).reshape_as(state)

    def _apply_cnot(self, state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        if not self.entangle:
            return state
        matrix = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            device=state.device,
            dtype=state.dtype,
        )
        tensor = state.reshape(state.shape[0], *([2] * self.n_qubits))
        control_axis = 1 + self.n_qubits - 1 - control
        target_axis = 1 + self.n_qubits - 1 - target
        moved = tensor.movedim((control_axis, target_axis), (-2, -1))
        flat = moved.reshape(*moved.shape[:-2], 4)
        updated = torch.einsum("b...j,ij->b...i", flat, matrix).reshape_as(moved)
        return updated.movedim((-2, -1), (control_axis, target_axis)).reshape_as(state)


class ClassicalMatchedCore(nn.Module):
    """A 6*nq-parameter classical core matching the VQC parameter budget."""

    def __init__(self, n_qubits: int, depth: int, *, seed: int) -> None:
        super().__init__()
        self.n_qubits, self.depth = n_qubits, depth
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        theta = torch.empty(depth, n_qubits, 3, dtype=torch.float32)
        theta.uniform_(-0.02, 0.02, generator=generator)
        self.theta = nn.Parameter(theta)
        self.circuit_calls = 0

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        if not encoded.is_cuda or not self.theta.is_cuda:
            raise RuntimeError("CC-024 matched control is GPU-only")
        left, right = encoded.float().chunk(2, dim=-1)
        values = torch.tanh(left + right)
        for layer in range(self.depth):
            a, b, c = self.theta[layer].unbind(dim=-1)
            local = values * torch.cos(a) + torch.roll(values, 1, -1) * torch.sin(a)
            values = torch.tanh(local + b + c * left * right)
        return values


class HybridValueReplacement(nn.Module):
    """Frozen SVD backbone plus a trainable quantum or matched residual."""

    def __init__(
        self,
        config: QH024Config,
        down_svd: torch.Tensor,
        up_svd: torch.Tensor,
        teacher_weight: torch.Tensor,
        *,
        core_kind: str = "quantum",
    ) -> None:
        super().__init__()
        self.config, self.core_kind = config, core_kind
        self.svd_down = nn.Linear(
            config.hidden_size, config.svd_rank, bias=False,
            device=down_svd.device, dtype=down_svd.dtype,
        )
        self.svd_up = nn.Linear(
            config.svd_rank, config.output_size, bias=False,
            device=up_svd.device, dtype=up_svd.dtype,
        )
        with torch.no_grad():
            self.svd_down.weight.copy_(down_svd)
            self.svd_up.weight.copy_(up_svd)
        self.svd_down.requires_grad_(False)
        self.svd_up.requires_grad_(False)
        self.down = nn.Linear(
            config.hidden_size, 2 * config.n_qubits, bias=False,
            device=teacher_weight.device, dtype=torch.float32,
        )
        self.up = nn.Linear(
            config.n_qubits, config.output_size, bias=False,
            device=teacher_weight.device, dtype=torch.float32,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.parameter_seed)
        with torch.no_grad():
            self.down.weight.copy_(
                (torch.randn(self.down.weight.shape, generator=generator) * 0.005).to(
                    self.down.weight.device
                )
            )
            self.up.weight.copy_(
                (torch.randn(self.up.weight.shape, generator=generator) * 0.005).to(
                    self.up.weight.device
                )
            )
        core_types = {
            "quantum": (HyQuTStatevectorCore, True),
            "dequantized": (TensorContractionHyQuTCore, True),
            "no_entanglement": (HyQuTStatevectorCore, False),
        }
        if core_kind == "classical":
            self.core = ClassicalMatchedCore(
                config.n_qubits, config.depth, seed=config.parameter_seed + 1
            ).to(teacher_weight.device)
        elif core_kind in core_types:
            core_class, entangle = core_types[core_kind]
            self.core = core_class(
                config.n_qubits, config.depth, seed=config.parameter_seed + 1, entangle=entangle
            ).to(teacher_weight.device)
        else:
            raise ValueError(f"unknown QH-024 core kind: {core_kind}")
        self.gamma = nn.Parameter(torch.tensor(config.gamma_init, device=teacher_weight.device))
        self.register_buffer("teacher_weight", teacher_weight.detach().clone(), persistent=False)
        self.capture_teacher_only = False
        self.residual_enabled = True
        self.captured_input: torch.Tensor | None = None
        self.captured_target: torch.Tensor | None = None

    @classmethod
    def from_dense(
        cls, projection: nn.Linear, config: QH024Config, *, core_kind: str = "quantum"
    ) -> "HybridValueReplacement":
        if projection.bias is not None:
            raise ValueError("QH-024 requires a bias-free dense projection")
        weight = projection.weight.detach()
        if tuple(weight.shape) != (config.output_size, config.hidden_size):
            raise ValueError("QH-024 dense projection shape mismatch")
        u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
        result = cls(
            config,
            vh[: config.svd_rank].to(weight.dtype),
            (u[:, : config.svd_rank] * s[: config.svd_rank]).to(weight.dtype),
            weight,
            core_kind=core_kind,
        )
        del u, s, vh
        torch.cuda.empty_cache()
        return result

    def backbone(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.svd_up(self.svd_down(hidden_states))

    def quantum_or_control_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape[:-1]
        flat = hidden_states.reshape(-1, self.config.hidden_size).float()
        features = self.core(torch.tanh(self.down(flat)))
        return self.up(features).reshape(*shape, self.config.output_size)

    def student_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = self.backbone(hidden_states)
        if not self.residual_enabled:
            return base
        residual = self.quantum_or_control_residual(hidden_states)
        return base + (self.gamma * residual).to(base.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.capture_teacher_only:
            if self.teacher_weight is None:
                raise RuntimeError("teacher buffer was stripped")
            teacher = F.linear(hidden_states, self.teacher_weight)
            self.captured_input, self.captured_target = hidden_states.detach(), teacher.detach()
            return teacher
        return self.student_forward(hidden_states)

    def set_capture_teacher_only(self, enabled: bool) -> None:
        if enabled and self.teacher_weight is None:
            raise RuntimeError("cannot capture after stripping the teacher")
        self.capture_teacher_only = enabled
        self.captured_input = self.captured_target = None

    def set_residual_enabled(self, enabled: bool) -> None:
        self.residual_enabled = enabled

    def take_capture(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.captured_input is None or self.captured_target is None:
            raise RuntimeError("no teacher capture is available")
        values = self.captured_input, self.captured_target
        self.captured_input = self.captured_target = None
        return values

    def strip_teacher(self) -> None:
        self.capture_teacher_only = False
        self.captured_input = self.captured_target = None
        self._buffers["teacher_weight"] = None

    @property
    def inference_simulator_required(self) -> bool:
        return self.core_kind in {"quantum", "no_entanglement"}

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def deployed_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def audit(self) -> dict:
        original = self.config.hidden_size * self.config.output_size
        deployed = self.deployed_parameter_count()
        candidate = {
            "quantum": "QH-024", "dequantized": "DQ-024",
            "no_entanglement": "QH-024-no-ent", "classical": "CC-024",
        }[self.core_kind]
        return {
            "candidate": candidate,
            "config": asdict(self.config),
            "architecture": "frozen rank-r SVD backbone + trainable HyQuT residual",
            "original_projection_parameters": original,
            "deployed_replacement_parameters": deployed,
            "removed_projection_parameters": original - deployed,
            "trainable_parameters": self.trainable_parameter_count(),
            "teacher_is_nonpersistent_training_buffer": self.teacher_weight is not None,
            "original_dense_projection_in_deployed_parameter_set": False,
            "simulator_calls_per_student_forward": 1 if self.inference_simulator_required else 0,
            "residual_enabled": self.residual_enabled,
            "inference_simulator_required": self.inference_simulator_required,
            "statevector_qubits": self.config.n_qubits if self.inference_simulator_required else 0,
            "claim_limit": "GPU-simulated executable structure; no hardware or speed advantage claim",
        }


def freeze_and_replace_qh024(
    model: nn.Module,
    config: QH024Config | None = None,
    *,
    core_kind: str = "quantum",
) -> dict:
    """Freeze Qwen3.8 and delete/replace one full-attention ``v_proj``."""
    config = config or QH024Config()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base_parameters = sum(p.numel() for p in model.parameters())
    layer = model.model.language_model.layers[config.layer_index]
    if getattr(layer, "block_type", None) != "full_attention":
        raise ValueError(f"layer {config.layer_index} is not full_attention")
    projection = getattr(layer.self_attn, config.target_projection)
    replacement = HybridValueReplacement.from_dense(projection, config, core_kind=core_kind)
    setattr(layer.self_attn, config.target_projection, replacement)
    deployed_parameters = sum(p.numel() for p in model.parameters())
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    return {
        "candidate": replacement.audit()["candidate"],
        "replacement_path": f"model.model.language_model.layers.{config.layer_index}.self_attn.{config.target_projection}",
        "base_model_parameters": base_parameters,
        "deployed_model_parameters": deployed_parameters,
        "net_model_parameter_reduction": base_parameters - deployed_parameters,
        "replacement": replacement.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
