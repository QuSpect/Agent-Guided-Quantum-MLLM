from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .segmented_anchor import (
    QH003AConfig,
    QuantumAnchorVisionWrapper,
    SegmentedQuantumAnchorAdapter,
)


class StableSegmentedQuantumAnchorAdapter(SegmentedQuantumAnchorAdapter):
    """QH-003b: normalize regional means to remove visual-token-count scale drift."""

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
        summaries = F.layer_norm(summaries, (self.config.hidden_size,))
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
        result = super().audit()
        result["candidate"] = "QH-003b"
        result["post_mean_layer_norm"] = True
        return result


def freeze_and_inject_qh003b(model: nn.Module, config: QH003AConfig | None = None) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.model.visual
    reference = next(visual.merger.parameters())
    adapter = StableSegmentedQuantumAnchorAdapter(config).to(device=reference.device)
    adapter.down.to(dtype=reference.dtype)
    adapter.up.to(dtype=reference.dtype)
    adapter.vqc.to(dtype=torch.float32)
    model.model.visual = QuantumAnchorVisionWrapper(visual, adapter)
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    return {
        "candidate": "QH-003b",
        "injection_path": "model.model.visual",
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
        "adapter": adapter.audit(),
    }

