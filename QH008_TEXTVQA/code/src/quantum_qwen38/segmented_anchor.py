from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .quantum_residual import NativeStatevectorVQC


@dataclass(frozen=True)
class QH003AConfig:
    hidden_size: int = 5120
    n_qubits: int = 8
    depth: int = 2
    anchors_per_image: int = 4
    scale_init: float = 0.3
    scale_max: float = 1.0
    up_init_std: float = 0.01


class SegmentedQuantumAnchorAdapter(nn.Module):
    """Image-boundary-aware regional quantum anchors broadcast back to visual tokens."""

    def __init__(self, config: QH003AConfig | None = None) -> None:
        super().__init__()
        self.config = config or QH003AConfig()
        self.down = nn.Linear(self.config.hidden_size, self.config.n_qubits, bias=True)
        self.vqc = NativeStatevectorVQC(self.config.n_qubits, self.config.depth)
        self.up = nn.Linear(self.config.n_qubits, self.config.hidden_size, bias=False)
        probability = self.config.scale_init / self.config.scale_max
        self.raw_scale = nn.Parameter(torch.tensor(math.log(probability / (1.0 - probability))))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, mean=0.0, std=self.config.up_init_std)
        self.last_num_anchors = 0

    @property
    def scale(self) -> torch.Tensor:
        return self.config.scale_max * torch.sigmoid(self.raw_scale)

    def forward(self, merged_tokens: torch.Tensor, split_sizes: list[int]) -> torch.Tensor:
        if sum(split_sizes) != merged_tokens.shape[0]:
            raise ValueError("split_sizes must exactly cover merged visual tokens")
        images = torch.split(merged_tokens, split_sizes)
        segments: list[torch.Tensor] = []
        segment_counts: list[int] = []
        for image_tokens in images:
            count = min(self.config.anchors_per_image, image_tokens.shape[0])
            image_segments = list(torch.tensor_split(image_tokens, count, dim=0))
            segments.extend(image_segments)
            segment_counts.append(count)

        summaries = torch.stack(
            [
                F.layer_norm(segment.float(), (self.config.hidden_size,)).mean(dim=0)
                for segment in segments
            ],
            dim=0,
        )
        angles = torch.pi * torch.tanh(self.down(summaries.to(self.down.weight.dtype)).float())
        expectations = self.vqc(angles)
        anchor_residuals = self.up(expectations.to(self.up.weight.dtype)).to(merged_tokens.dtype)
        anchor_residuals = anchor_residuals * self.scale.to(anchor_residuals.dtype)

        reconstructed = []
        cursor = 0
        for image_tokens, count in zip(images, segment_counts, strict=True):
            image_segments = torch.tensor_split(image_tokens, count, dim=0)
            for segment, residual in zip(
                image_segments, anchor_residuals[cursor : cursor + count], strict=True
            ):
                reconstructed.append(segment + residual.unsqueeze(0))
            cursor += count
        self.last_num_anchors = len(segments)
        return torch.cat(reconstructed, dim=0)

    def audit(self) -> dict:
        return {
            "candidate": "QH-003a",
            "config": asdict(self.config),
            "current_scale": float(self.scale.detach().cpu()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


class QuantumAnchorVisionWrapper(nn.Module):
    """Wrap Qwen3.8's visual tower without modifying Transformers source."""

    def __init__(self, frozen_visual: nn.Module, adapter: SegmentedQuantumAnchorAdapter) -> None:
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
            raise ValueError("grid_thw is required for image-boundary-aware anchors")
        output = self.frozen_visual(*args, grid_thw=grid_thw, **kwargs)
        split_sizes = (grid_thw.prod(-1) // self.spatial_merge_size**2).detach().cpu().tolist()
        output.pooler_output = self.adapter(output.pooler_output, split_sizes)
        return output


def freeze_and_inject_qh003a(model: nn.Module, config: QH003AConfig | None = None) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.model.visual
    reference = next(visual.merger.parameters())
    adapter = SegmentedQuantumAnchorAdapter(config).to(device=reference.device)
    adapter.down.to(dtype=reference.dtype)
    adapter.up.to(dtype=reference.dtype)
    adapter.vqc.to(dtype=torch.float32)
    model.model.visual = QuantumAnchorVisionWrapper(visual, adapter)
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    return {
        "candidate": "QH-003a",
        "injection_path": "model.model.visual",
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
        "adapter": adapter.audit(),
    }

