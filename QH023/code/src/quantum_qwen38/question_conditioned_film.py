"""QH-015: question-conditioned quantum FiLM over token-specific visual features."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .correlation_gated_low_rank import (
    CorrelatorClassicalGateCore,
    CorrelatorNoEntanglementVQC,
    CorrelatorStatevectorVQC,
)


@dataclass(frozen=True)
class QH015Config:
    hidden_size: int = 5120
    rank: int = 64
    n_qubits: int = 8
    depth: int = 2
    gate_strength: float = 1.0
    scale_init: float = 0.05
    scale_max: float = 1.0
    token_up_init_std: float = 0.001


class QuestionConditionedFiLMAdapter(nn.Module):
    """Use one cross-modal circuit per image to modulate token-local features."""

    def __init__(
        self,
        config: QH015Config | None = None,
        core_kind: str = "quantum",
    ) -> None:
        super().__init__()
        self.config = config or QH015Config()
        self.core_kind = core_kind
        self.token_down = nn.Linear(self.config.hidden_size, self.config.rank, bias=True)
        self.visual_down = nn.Linear(self.config.hidden_size, self.config.n_qubits, bias=True)
        self.text_down = nn.Linear(self.config.hidden_size, self.config.n_qubits, bias=True)
        if core_kind == "quantum":
            self.core = CorrelatorStatevectorVQC(self.config.n_qubits, self.config.depth)
        elif core_kind == "classical":
            self.core = CorrelatorClassicalGateCore(self.config.n_qubits, self.config.depth)
        elif core_kind == "no_entanglement":
            self.core = CorrelatorNoEntanglementVQC(
                self.config.n_qubits, self.config.depth
            )
        else:
            raise ValueError(f"unsupported core_kind: {core_kind}")
        self.gate_up = nn.Linear(
            2 * self.config.n_qubits, self.config.rank, bias=True
        )
        self.token_up = nn.Linear(self.config.rank, self.config.hidden_size, bias=False)
        probability = self.config.scale_init / self.config.scale_max
        self.raw_scale = nn.Parameter(
            torch.tensor(
                math.log(probability / (1.0 - probability)), dtype=torch.float32
            )
        )
        for projection in (self.token_down, self.visual_down, self.text_down):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        nn.init.xavier_uniform_(self.gate_up.weight)
        nn.init.zeros_(self.gate_up.bias)
        nn.init.normal_(
            self.token_up.weight, mean=0.0, std=self.config.token_up_init_std
        )
        self._text_anchor: torch.Tensor | None = None
        self.last_circuit_calls = 0

    @property
    def scale(self) -> torch.Tensor:
        return self.config.scale_max * torch.sigmoid(self.raw_scale)

    def set_text_anchor(self, anchor: torch.Tensor) -> None:
        if anchor.ndim != 2 or anchor.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected [batch, {self.config.hidden_size}] text anchor, "
                f"got {tuple(anchor.shape)}"
            )
        self._text_anchor = anchor.detach().to(self.text_down.weight.device)

    def residual(self, merged_tokens: torch.Tensor, split_sizes: list[int]) -> torch.Tensor:
        if self._text_anchor is None:
            raise RuntimeError("QH-015 requires a prompt-only text anchor before visual forward")
        if merged_tokens.ndim != 2 or merged_tokens.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected [visual_tokens, {self.config.hidden_size}], "
                f"got {tuple(merged_tokens.shape)}"
            )
        if sum(split_sizes) != merged_tokens.shape[0]:
            raise ValueError("split_sizes must exactly cover merged visual tokens")

        normalized = F.layer_norm(
            merged_tokens.float(), (self.config.hidden_size,)
        )
        images = torch.split(normalized, split_sizes)
        summaries = torch.stack([
            F.layer_norm(image.mean(dim=0), (self.config.hidden_size,))
            for image in images
        ])
        text = self._text_anchor.float()
        if text.shape[0] == 1 and len(images) > 1:
            text = text.expand(len(images), -1)
        if text.shape[0] != len(images):
            raise ValueError("text-anchor batch must be one or match the number of images")
        text = F.layer_norm(text, (self.config.hidden_size,))

        visual_latent = self.visual_down(
            summaries.to(self.visual_down.weight.dtype)
        ).float()
        text_latent = self.text_down(text.to(self.text_down.weight.dtype)).float()
        joint_angles = torch.pi * torch.tanh(0.5 * (visual_latent + text_latent))
        observables = self.core(joint_angles)
        image_gates = torch.tanh(
            self.gate_up(observables.to(self.gate_up.weight.dtype)).float()
        )
        token_gates = torch.cat([
            gate.unsqueeze(0).expand(size, -1)
            for gate, size in zip(image_gates, split_sizes, strict=True)
        ])
        token_features = F.gelu(
            self.token_down(normalized.to(self.token_down.weight.dtype)).float()
        )
        modulated = token_features * (
            1.0 + self.config.gate_strength * token_gates
        )
        residual = self.token_up(
            modulated.to(self.token_up.weight.dtype)
        ).to(merged_tokens.dtype)
        self.last_circuit_calls = len(images)
        return residual * self.scale.to(residual.dtype)

    def forward(self, merged_tokens: torch.Tensor, split_sizes: list[int]) -> torch.Tensor:
        return merged_tokens + self.residual(merged_tokens, split_sizes)

    def audit(self) -> dict:
        candidate = {
            "quantum": "QH-015",
            "classical": "CC-015",
            "no_entanglement": "QH-015-no-ent",
        }[self.core_kind]
        return {
            "candidate": candidate,
            "config": asdict(self.config),
            "core_kind": self.core_kind,
            "text_anchor": "mean frozen prompt-token embedding; answer excluded",
            "image_anchor": "mean normalized visual tokens per image",
            "readout": "8 local-Z + 8 adjacent-ZZ observables",
            "modulation": "one image-level gate FiLM-modulates token-specific rank features",
            "execution": "one circuit call per image during visual prefill",
            "current_scale": float(self.scale.detach().cpu()),
            "trainable_parameters": sum(p.numel() for p in self.parameters()),
        }


class QuestionConditionedFiLMVisionWrapper(nn.Module):
    def __init__(self, frozen_visual: nn.Module, adapter: QuestionConditionedFiLMAdapter) -> None:
        super().__init__()
        self.frozen_visual = frozen_visual
        self.adapter = adapter
        for parameter in self.frozen_visual.parameters():
            parameter.requires_grad_(False)

    @property
    def dtype(self) -> torch.dtype:
        return self.frozen_visual.dtype

    @property
    def device(self) -> torch.device:
        return self.frozen_visual.device

    @property
    def spatial_merge_size(self) -> int:
        return self.frozen_visual.spatial_merge_size

    @property
    def config(self):
        return self.frozen_visual.config

    def forward(self, *args, grid_thw: torch.Tensor | None = None, **kwargs):
        if grid_thw is None:
            raise ValueError("grid_thw is required for per-image QH-015 FiLM")
        output = self.frozen_visual(*args, grid_thw=grid_thw, **kwargs)
        split_sizes = (
            grid_thw.prod(-1) // self.spatial_merge_size**2
        ).detach().cpu().tolist()
        output.pooler_output = self.adapter(output.pooler_output, split_sizes)
        return output


def freeze_and_inject_qh015(
    model: nn.Module,
    core_kind: str = "quantum",
    config: QH015Config | None = None,
) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.model.visual
    reference = next(visual.merger.parameters())
    adapter = QuestionConditionedFiLMAdapter(
        config, core_kind=core_kind
    ).to(device=reference.device)
    for projection in (
        adapter.token_down,
        adapter.visual_down,
        adapter.text_down,
        adapter.gate_up,
        adapter.token_up,
    ):
        projection.to(dtype=reference.dtype)
    adapter.core.to(dtype=torch.float32)
    model.model.visual = QuestionConditionedFiLMVisionWrapper(visual, adapter)
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    candidate = {
        "quantum": "QH-015",
        "classical": "CC-015",
        "no_entanglement": "QH-015-no-ent",
    }[core_kind]
    return {
        "candidate": candidate,
        "parent": "QH-013 token content path + QH-014 prompt semantic anchor",
        "injection_path": "model.model.visual.pooler_output",
        "execution_scope": "one circuit per image during visual prefill",
        "quantum_compute_dtype": (
            "torch.float32/torch.complex64" if core_kind != "classical" else None
        ),
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
