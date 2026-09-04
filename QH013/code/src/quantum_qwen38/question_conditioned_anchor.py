"""QH-014: one-circuit-per-image question-conditioned visual anchor."""

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
class QH014Config:
    hidden_size: int = 5120
    n_qubits: int = 8
    depth: int = 2
    scale_init: float = 0.05
    scale_max: float = 1.0
    up_init_std: float = 0.001


class QuestionConditionedAnchorAdapter(nn.Module):
    """Fuse a prompt-only text anchor with one pooled vector per image."""

    def __init__(
        self,
        config: QH014Config | None = None,
        core_kind: str = "quantum",
    ) -> None:
        super().__init__()
        self.config = config or QH014Config()
        self.core_kind = core_kind
        self.visual_down = nn.Linear(self.config.hidden_size, self.config.n_qubits, bias=True)
        self.text_down = nn.Linear(self.config.hidden_size, self.config.n_qubits, bias=True)
        if core_kind == "quantum":
            self.core = CorrelatorStatevectorVQC(self.config.n_qubits, self.config.depth)
        elif core_kind == "classical":
            self.core = CorrelatorClassicalGateCore(self.config.n_qubits, self.config.depth)
        elif core_kind == "no_entanglement":
            self.core = CorrelatorNoEntanglementVQC(self.config.n_qubits, self.config.depth)
        else:
            raise ValueError(f"unsupported core_kind: {core_kind}")
        self.up = nn.Linear(2 * self.config.n_qubits, self.config.hidden_size, bias=False)
        probability = self.config.scale_init / self.config.scale_max
        self.raw_scale = nn.Parameter(
            torch.tensor(math.log(probability / (1.0 - probability)), dtype=torch.float32)
        )
        nn.init.xavier_uniform_(self.visual_down.weight)
        nn.init.zeros_(self.visual_down.bias)
        nn.init.xavier_uniform_(self.text_down.weight)
        nn.init.zeros_(self.text_down.bias)
        nn.init.normal_(self.up.weight, mean=0.0, std=self.config.up_init_std)
        self._text_anchor: torch.Tensor | None = None
        self.last_circuit_calls = 0

    @property
    def scale(self) -> torch.Tensor:
        return self.config.scale_max * torch.sigmoid(self.raw_scale)

    def set_text_anchor(self, anchor: torch.Tensor) -> None:
        if anchor.ndim != 2 or anchor.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected [batch, {self.config.hidden_size}] text anchor, got {tuple(anchor.shape)}"
            )
        self._text_anchor = anchor.detach().to(self.text_down.weight.device)

    def residual(self, merged_tokens: torch.Tensor, split_sizes: list[int]) -> torch.Tensor:
        if self._text_anchor is None:
            raise RuntimeError("QH-014 requires a prompt-only text anchor before visual forward")
        if sum(split_sizes) != merged_tokens.shape[0]:
            raise ValueError("split_sizes must exactly cover merged visual tokens")
        images = torch.split(merged_tokens, split_sizes)
        summaries = torch.stack([
            F.layer_norm(
                F.layer_norm(image.float(), (self.config.hidden_size,)).mean(dim=0),
                (self.config.hidden_size,),
            )
            for image in images
        ])
        text = self._text_anchor.float()
        if text.shape[0] == 1 and len(images) > 1:
            text = text.expand(len(images), -1)
        if text.shape[0] != len(images):
            raise ValueError("text-anchor batch must be one or match the number of images")
        text = F.layer_norm(text, (self.config.hidden_size,))
        visual_latent = self.visual_down(summaries.to(self.visual_down.weight.dtype)).float()
        text_latent = self.text_down(text.to(self.text_down.weight.dtype)).float()
        joint_angles = torch.pi * torch.tanh(0.5 * (visual_latent + text_latent))
        observables = self.core(joint_angles)
        image_residuals = self.up(observables.to(self.up.weight.dtype)).to(merged_tokens.dtype)
        image_residuals = image_residuals * self.scale.to(image_residuals.dtype)
        self.last_circuit_calls = len(images)
        return torch.cat([
            residual.unsqueeze(0).expand(image.shape[0], -1)
            for image, residual in zip(images, image_residuals, strict=True)
        ])

    def forward(self, merged_tokens: torch.Tensor, split_sizes: list[int]) -> torch.Tensor:
        return merged_tokens + self.residual(merged_tokens, split_sizes)

    def audit(self) -> dict:
        candidate = {
            "quantum": "QH-014",
            "classical": "CC-014",
            "no_entanglement": "QH-014-no-ent",
        }[self.core_kind]
        return {
            "candidate": candidate,
            "config": asdict(self.config),
            "core_kind": self.core_kind,
            "text_anchor": "mean frozen prompt-token embedding; answer excluded during training",
            "image_anchor": "mean normalized visual tokens per image",
            "readout": "8 local-Z + 8 adjacent-ZZ observables",
            "execution": "one circuit call per image",
            "current_scale": float(self.scale.detach().cpu()),
            "trainable_parameters": sum(p.numel() for p in self.parameters()),
        }


class QuestionConditionedVisionWrapper(nn.Module):
    def __init__(self, frozen_visual: nn.Module, adapter: QuestionConditionedAnchorAdapter) -> None:
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
            raise ValueError("grid_thw is required for per-image question anchors")
        output = self.frozen_visual(*args, grid_thw=grid_thw, **kwargs)
        split_sizes = (grid_thw.prod(-1) // self.spatial_merge_size**2).detach().cpu().tolist()
        output.pooler_output = self.adapter(output.pooler_output, split_sizes)
        return output


def freeze_and_inject_qh014(
    model: nn.Module,
    core_kind: str = "quantum",
    config: QH014Config | None = None,
) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.model.visual
    reference = next(visual.merger.parameters())
    adapter = QuestionConditionedAnchorAdapter(config, core_kind=core_kind).to(
        device=reference.device
    )
    adapter.visual_down.to(dtype=reference.dtype)
    adapter.text_down.to(dtype=reference.dtype)
    adapter.up.to(dtype=reference.dtype)
    adapter.core.to(dtype=torch.float32)
    model.model.visual = QuestionConditionedVisionWrapper(visual, adapter)
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    candidate = {
        "quantum": "QH-014",
        "classical": "CC-014",
        "no_entanglement": "QH-014-no-ent",
    }[core_kind]
    return {
        "candidate": candidate,
        "parent": "QH-003b + MQAdapter-inspired prompt semantic anchor",
        "injection_path": "model.model.visual.pooler_output",
        "execution_scope": "one circuit per image during visual prefill",
        "quantum_compute_dtype": (
            "torch.float32/torch.complex64" if core_kind != "classical" else None
        ),
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
