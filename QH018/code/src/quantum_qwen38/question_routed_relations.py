"""QH-016: question-routed quantum relations over frozen spatial regions."""

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
class QH016Config:
    hidden_size: int = 5120
    token_rank: int = 64
    object_rank: int = 16
    n_qubits: int = 6
    depth: int = 2
    regions_per_side: int = 2
    routing_temperature: float = 0.5
    gate_strength: float = 1.0
    scale_init: float = 0.05
    scale_max: float = 1.0
    token_up_init_std: float = 0.001


class QuestionRoutedRelationAdapter(nn.Module):
    """Model six directed relations among four frozen spatial region summaries.

    The Qwen vision backbone remains frozen.  Each image is reduced to a 2x2
    grid of region summaries, producing the deterministic directed pairs
    (0,1), (0,2), (0,3), (1,2), (1,3), and (2,3).  A prompt-only question
    anchor routes the pair outputs back to token-local low-rank features.
    """

    def __init__(
        self,
        config: QH016Config | None = None,
        core_kind: str = "quantum",
    ) -> None:
        super().__init__()
        self.config = config or QH016Config()
        if self.config.regions_per_side != 2:
            raise ValueError("QH-016 currently requires a 2x2 region grid")
        if not 0.0 < self.config.scale_init < self.config.scale_max:
            raise ValueError("scale_init must lie strictly between zero and scale_max")
        self.core_kind = core_kind
        self.token_down = nn.Linear(
            self.config.hidden_size, self.config.token_rank, bias=True
        )
        self.object_down = nn.Linear(
            self.config.hidden_size, self.config.object_rank, bias=True
        )
        self.text_down = nn.Linear(
            self.config.hidden_size, self.config.object_rank, bias=True
        )
        relation_width = 5 * self.config.object_rank
        self.relation_down = nn.Linear(relation_width, self.config.n_qubits, bias=True)
        if core_kind == "quantum":
            self.core = CorrelatorStatevectorVQC(
                self.config.n_qubits, self.config.depth
            )
        elif core_kind == "classical":
            self.core = CorrelatorClassicalGateCore(
                self.config.n_qubits, self.config.depth
            )
        elif core_kind == "no_entanglement":
            self.core = CorrelatorNoEntanglementVQC(
                self.config.n_qubits, self.config.depth
            )
        else:
            raise ValueError(f"unsupported core_kind: {core_kind}")
        self.pair_position = nn.Parameter(
            torch.zeros(6, self.config.n_qubits, dtype=torch.float32)
        )
        self.gate_up = nn.Linear(
            2 * self.config.n_qubits, self.config.token_rank, bias=True
        )
        self.token_up = nn.Linear(
            self.config.token_rank, self.config.hidden_size, bias=False
        )
        probability = self.config.scale_init / self.config.scale_max
        self.raw_scale = nn.Parameter(
            torch.tensor(
                math.log(probability / (1.0 - probability)), dtype=torch.float32
            )
        )
        for projection in (
            self.token_down,
            self.object_down,
            self.text_down,
            self.relation_down,
        ):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        nn.init.normal_(self.pair_position, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.gate_up.weight)
        nn.init.zeros_(self.gate_up.bias)
        nn.init.normal_(
            self.token_up.weight, mean=0.0, std=self.config.token_up_init_std
        )
        self.register_buffer(
            "pair_indices",
            torch.tensor(
                [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self._text_anchor: torch.Tensor | None = None
        self.last_circuit_evaluations = 0
        self.last_routing_weights: torch.Tensor | None = None

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

    def _region_summaries(
        self, image: torch.Tensor, grid_shape: tuple[int, int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temporal, height, width = grid_shape
        expected = temporal * height * width
        if image.shape[0] != expected:
            raise ValueError(
                f"grid {grid_shape} covers {expected} tokens, got {image.shape[0]}"
            )
        normalized = F.layer_norm(image.float(), (self.config.hidden_size,))
        grid = normalized.reshape(temporal, height, width, self.config.hidden_size)
        grid = grid.mean(dim=0).permute(2, 0, 1).unsqueeze(0)
        pooled = F.adaptive_avg_pool2d(
            grid, (self.config.regions_per_side, self.config.regions_per_side)
        )
        regions = pooled.squeeze(0).permute(1, 2, 0).reshape(4, -1)
        y = torch.arange(height, device=image.device).mul(2).div(
            height, rounding_mode="floor"
        )
        x = torch.arange(width, device=image.device).mul(2).div(
            width, rounding_mode="floor"
        )
        region_ids = (y[:, None] * 2 + x[None, :]).reshape(-1).repeat(temporal)
        return regions, region_ids

    def residual(
        self,
        merged_tokens: torch.Tensor,
        grid_shapes: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        if self._text_anchor is None:
            raise RuntimeError("QH-016 requires a prompt-only text anchor")
        split_sizes = [t * h * w for t, h, w in grid_shapes]
        if sum(split_sizes) != merged_tokens.shape[0]:
            raise ValueError("grid_shapes must exactly cover merged visual tokens")
        images = torch.split(merged_tokens, split_sizes)
        text = self._text_anchor.float()
        if text.shape[0] == 1 and len(images) > 1:
            text = text.expand(len(images), -1)
        if text.shape[0] != len(images):
            raise ValueError("text-anchor batch must be one or match image count")
        text = F.layer_norm(text, (self.config.hidden_size,))

        residuals: list[torch.Tensor] = []
        route_audit: list[torch.Tensor] = []
        pair_indices = self.pair_indices
        for image_index, (image, grid_shape) in enumerate(
            zip(images, grid_shapes, strict=True)
        ):
            regions, region_ids = self._region_summaries(image, grid_shape)
            objects = self.object_down(
                regions.to(self.object_down.weight.dtype)
            ).float()
            question = self.text_down(
                text[image_index].to(self.text_down.weight.dtype)
            ).float()
            left = objects.index_select(0, pair_indices[:, 0])
            right = objects.index_select(0, pair_indices[:, 1])
            question_pairs = question.unsqueeze(0).expand(left.shape[0], -1)
            relation_features = torch.cat(
                [left, right, left - right, left * right, question_pairs], dim=-1
            )
            raw_angles = self.relation_down(
                relation_features.to(self.relation_down.weight.dtype)
            ).float()
            angles = torch.pi * torch.tanh(raw_angles + self.pair_position)
            observables = self.core(angles)
            pair_gates = torch.tanh(
                self.gate_up(observables.to(self.gate_up.weight.dtype)).float()
            )
            routing_logits = (
                (0.5 * (left + right) * question_pairs).sum(dim=-1)
                / math.sqrt(self.config.object_rank)
            )
            routing = torch.softmax(
                routing_logits / self.config.routing_temperature, dim=0
            )
            region_gates = torch.zeros(
                4,
                self.config.token_rank,
                device=image.device,
                dtype=torch.float32,
            )
            weighted = pair_gates * routing.unsqueeze(-1)
            region_gates.index_add_(0, pair_indices[:, 0], weighted)
            region_gates.index_add_(0, pair_indices[:, 1], -weighted)
            token_gates = region_gates.index_select(0, region_ids)
            normalized = F.layer_norm(image.float(), (self.config.hidden_size,))
            token_features = F.gelu(
                self.token_down(normalized.to(self.token_down.weight.dtype)).float()
            )
            modulated = token_features * (
                1.0 + self.config.gate_strength * token_gates
            )
            residuals.append(
                self.token_up(modulated.to(self.token_up.weight.dtype)).to(image.dtype)
            )
            route_audit.append(routing.detach())
        self.last_circuit_evaluations = 6 * len(images)
        self.last_routing_weights = torch.stack(route_audit)
        result = torch.cat(residuals, dim=0)
        return result * self.scale.to(result.dtype)

    def forward(
        self,
        merged_tokens: torch.Tensor,
        grid_shapes: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        return merged_tokens + self.residual(merged_tokens, grid_shapes)

    def audit(self) -> dict:
        candidate = {
            "quantum": "QH-016",
            "classical": "CC-016",
            "no_entanglement": "QH-016-no-ent",
        }[self.core_kind]
        return {
            "candidate": candidate,
            "config": asdict(self.config),
            "core_kind": self.core_kind,
            "frozen_objects": "2x2 pooled regions from frozen Qwen vision tokens",
            "relations": "six deterministic directed spatial-region pairs",
            "routing": "prompt-only question anchor softmax over relation pairs",
            "readout": "6 local-Z + 6 adjacent-ZZ observables per relation",
            "execution": "six relation states batched per image on GPU simulator",
            "current_scale": float(self.scale.detach().cpu()),
            "trainable_parameters": sum(p.numel() for p in self.parameters()),
        }


class QuestionRoutedRelationVisionWrapper(nn.Module):
    def __init__(
        self, frozen_visual: nn.Module, adapter: QuestionRoutedRelationAdapter
    ) -> None:
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
            raise ValueError("grid_thw is required for QH-016 spatial relations")
        output = self.frozen_visual(*args, grid_thw=grid_thw, **kwargs)
        merge = self.spatial_merge_size
        grid_shapes = [
            (int(t), int(h) // merge, int(w) // merge)
            for t, h, w in grid_thw.detach().cpu().tolist()
        ]
        output.pooler_output = self.adapter(output.pooler_output, grid_shapes)
        return output


def freeze_and_inject_qh016(
    model: nn.Module,
    core_kind: str = "quantum",
    config: QH016Config | None = None,
) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.model.visual
    reference = next(visual.merger.parameters())
    adapter = QuestionRoutedRelationAdapter(config, core_kind=core_kind).to(
        device=reference.device
    )
    for projection in (
        adapter.token_down,
        adapter.object_down,
        adapter.text_down,
        adapter.relation_down,
        adapter.gate_up,
        adapter.token_up,
    ):
        projection.to(dtype=reference.dtype)
    adapter.core.to(dtype=torch.float32)
    model.model.visual = QuestionRoutedRelationVisionWrapper(visual, adapter)
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    candidate = {
        "quantum": "QH-016",
        "classical": "CC-016",
        "no_entanglement": "QH-016-no-ent",
    }[core_kind]
    return {
        "candidate": candidate,
        "parent": "QH-015 + CCG-VQC/staged compositional relation hypothesis",
        "injection_path": "model.model.visual.pooler_output",
        "execution_scope": "six batched relation states per image during visual prefill",
        "quantum_compute_dtype": (
            "torch.float32/torch.complex64" if core_kind != "classical" else None
        ),
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
