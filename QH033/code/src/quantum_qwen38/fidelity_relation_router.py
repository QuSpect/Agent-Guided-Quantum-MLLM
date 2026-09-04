"""QH-017: quantum-state fidelity as the causal relation-routing bottleneck."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .quantum_residual import NativeStatevectorVQC


@dataclass(frozen=True)
class QH017Config:
    hidden_size: int = 5120
    token_rank: int = 64
    object_rank: int = 16
    n_qubits: int = 4
    depth: int = 2
    regions_per_side: int = 2
    routing_temperature: float = 0.25
    gate_strength: float = 1.0
    scale_init: float = 0.05
    scale_max: float = 1.0
    token_up_init_std: float = 0.001


class FidelityStatevectorRouter(NativeStatevectorVQC):
    """Two trainable state encoders whose squared overlap is the routing score."""

    def __init__(self, n_qubits: int, depth: int, entangle: bool = True) -> None:
        super().__init__(n_qubits=n_qubits, depth=depth)
        self.theta = nn.Parameter(
            torch.empty(2, depth, n_qubits, 2, dtype=torch.float32)
        )
        nn.init.uniform_(self.theta, -0.05, 0.05)
        self.entangle = entangle

    def _encode(self, angles: torch.Tensor, branch: int) -> torch.Tensor:
        if not angles.is_cuda:
            raise RuntimeError("QH-017 quantum simulation is GPU-only")
        state = torch.zeros(
            angles.shape[0],
            1 << self.n_qubits,
            dtype=torch.complex64,
            device=angles.device,
        )
        state[:, 0] = 1.0 + 0.0j
        for wire in range(self.n_qubits):
            state = self._apply_ry(state, angles[:, wire], wire)
        for layer in range(self.depth):
            for wire in range(self.n_qubits):
                state = self._apply_ry(
                    state, self.theta[branch, layer, wire, 0], wire
                )
                state = self._apply_rz(
                    state, self.theta[branch, layer, wire, 1], wire
                )
            if self.entangle:
                for control in range(self.n_qubits):
                    target = (control + 1) % self.n_qubits
                    state = state.index_select(
                        1, getattr(self, f"cnot_{control}_{target}")
                    )
        return state

    def forward(
        self, query_angles: torch.Tensor, key_angles: torch.Tensor
    ) -> torch.Tensor:
        if query_angles.ndim != 2 or query_angles.shape != (1, self.n_qubits):
            raise ValueError(
                f"expected one query [1, {self.n_qubits}], got {tuple(query_angles.shape)}"
            )
        if key_angles.ndim != 2 or key_angles.shape[-1] != self.n_qubits:
            raise ValueError(
                f"expected keys [pairs, {self.n_qubits}], got {tuple(key_angles.shape)}"
            )
        query = self._encode(query_angles.float(), branch=0)
        keys = self._encode(key_angles.float(), branch=1)
        overlap = (query.conj() * keys).sum(dim=-1)
        return overlap.abs().square().clamp(0.0, 1.0)


class ClassicalSimilarityRouter(nn.Module):
    """Equal-parameter non-quantum similarity control for the fidelity router."""

    def __init__(self, n_features: int, depth: int) -> None:
        super().__init__()
        self.affine = nn.Parameter(
            torch.empty(2, depth, n_features, 2, dtype=torch.float32)
        )
        nn.init.uniform_(self.affine, -0.05, 0.05)

    def _encode(self, angles: torch.Tensor, branch: int) -> torch.Tensor:
        features = torch.cos(angles.float())
        for layer in range(self.affine.shape[1]):
            gain = 1.0 + self.affine[branch, layer, :, 0]
            bias = self.affine[branch, layer, :, 1]
            features = torch.tanh(features * gain + bias)
        return F.normalize(features, dim=-1, eps=1.0e-6)

    def forward(
        self, query_angles: torch.Tensor, key_angles: torch.Tensor
    ) -> torch.Tensor:
        query = self._encode(query_angles, branch=0)
        keys = self._encode(key_angles, branch=1)
        return (query * keys).sum(dim=-1).square().clamp(0.0, 1.0)


class FidelityRelationRouterAdapter(nn.Module):
    """Route six frozen spatial relations by quantum state fidelity."""

    def __init__(
        self,
        config: QH017Config | None = None,
        core_kind: str = "quantum",
    ) -> None:
        super().__init__()
        self.config = config or QH017Config()
        if self.config.regions_per_side != 2:
            raise ValueError("QH-017 currently requires a 2x2 region grid")
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
        relation_width = 4 * self.config.object_rank
        self.key_down = nn.Linear(relation_width, self.config.n_qubits, bias=True)
        self.query_down = nn.Linear(
            self.config.object_rank, self.config.n_qubits, bias=True
        )
        self.pair_content = nn.Linear(
            relation_width, self.config.token_rank, bias=True
        )
        if core_kind == "quantum":
            self.router = FidelityStatevectorRouter(
                self.config.n_qubits, self.config.depth, entangle=True
            )
        elif core_kind == "no_entanglement":
            self.router = FidelityStatevectorRouter(
                self.config.n_qubits, self.config.depth, entangle=False
            )
        elif core_kind == "classical":
            self.router = ClassicalSimilarityRouter(
                self.config.n_qubits, self.config.depth
            )
        else:
            raise ValueError(f"unsupported core_kind: {core_kind}")
        self.pair_position = nn.Parameter(
            torch.zeros(6, self.config.n_qubits, dtype=torch.float32)
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
            self.key_down,
            self.query_down,
            self.pair_content,
        ):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        nn.init.normal_(self.pair_position, mean=0.0, std=0.02)
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
        self.last_fidelities: torch.Tensor | None = None

    @property
    def scale(self) -> torch.Tensor:
        return self.config.scale_max * torch.sigmoid(self.raw_scale)

    def _routing_weights(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.softmax(
            (2.0 * scores - 1.0) / self.config.routing_temperature,
            dim=0,
        )

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
            raise RuntimeError("QH-017 requires a prompt-only text anchor")
        if not merged_tokens.is_cuda:
            raise RuntimeError("QH-017 adapter execution is GPU-only")
        counts = [t * h * w for t, h, w in grid_shapes]
        if sum(counts) != merged_tokens.shape[0]:
            raise ValueError("grid shapes do not cover merged visual tokens")
        images = list(torch.split(merged_tokens, counts, dim=0))
        text = self._text_anchor
        if text.shape[0] == 1 and len(images) > 1:
            text = text.expand(len(images), -1)
        if text.shape[0] != len(images):
            raise ValueError("text-anchor batch must be one or match image count")
        text = F.layer_norm(text.float(), (self.config.hidden_size,))

        residuals: list[torch.Tensor] = []
        routing_audit: list[torch.Tensor] = []
        fidelity_audit: list[torch.Tensor] = []
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
            relation_features = torch.cat(
                [left, right, left - right, left * right], dim=-1
            )
            key_angles = torch.pi * torch.tanh(
                self.key_down(relation_features.to(self.key_down.weight.dtype)).float()
                + self.pair_position
            )
            query_angles = torch.pi * torch.tanh(
                self.query_down(question.to(self.query_down.weight.dtype)).float()
            ).unsqueeze(0)
            fidelities = self.router(query_angles, key_angles)
            routing = self._routing_weights(fidelities)
            pair_gates = torch.tanh(
                self.pair_content(
                    relation_features.to(self.pair_content.weight.dtype)
                ).float()
            )
            weighted = pair_gates * routing.unsqueeze(-1)
            region_gates = torch.zeros(
                4,
                self.config.token_rank,
                device=image.device,
                dtype=torch.float32,
            )
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
            routing_audit.append(routing.detach())
            fidelity_audit.append(fidelities.detach())
        self.last_circuit_evaluations = 7 * len(images)
        self.last_routing_weights = torch.stack(routing_audit)
        self.last_fidelities = torch.stack(fidelity_audit)
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
            "quantum": "QH-017",
            "classical": "CC-017",
            "no_entanglement": "QH-017-no-ent",
        }[self.core_kind]
        return {
            "candidate": candidate,
            "config": asdict(self.config),
            "core_kind": self.core_kind,
            "routing": "question-to-relation state fidelity is the causal softmax bottleneck",
            "relations": "six deterministic directed 2x2 spatial-region pairs",
            "execution": "one query plus six key states per image on GPU simulator",
            "current_scale": float(self.scale.detach().cpu()),
            "trainable_parameters": sum(p.numel() for p in self.parameters()),
        }


class FidelityRelationVisionWrapper(nn.Module):
    def __init__(self, frozen_visual: nn.Module, adapter: nn.Module) -> None:
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
            raise ValueError("grid_thw is required for QH-017 spatial relations")
        output = self.frozen_visual(*args, grid_thw=grid_thw, **kwargs)
        merge = self.spatial_merge_size
        grid_shapes = [
            (int(t), int(h) // merge, int(w) // merge)
            for t, h, w in grid_thw.detach().cpu().tolist()
        ]
        output.pooler_output = self.adapter(output.pooler_output, grid_shapes)
        return output


def freeze_and_inject_qh017(
    model: nn.Module,
    core_kind: str = "quantum",
    config: QH017Config | None = None,
) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.model.visual
    reference = next(visual.merger.parameters())
    adapter = FidelityRelationRouterAdapter(config, core_kind=core_kind).to(
        device=reference.device
    )
    for projection in (
        adapter.token_down,
        adapter.object_down,
        adapter.text_down,
        adapter.key_down,
        adapter.query_down,
        adapter.pair_content,
        adapter.token_up,
    ):
        projection.to(dtype=reference.dtype)
    adapter.router.to(dtype=torch.float32)
    model.model.visual = FidelityRelationVisionWrapper(visual, adapter)
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    candidate = {
        "quantum": "QH-017",
        "classical": "CC-017",
        "no_entanglement": "QH-017-no-ent",
    }[core_kind]
    return {
        "candidate": candidate,
        "parent": "QH-016 causal-bottleneck mutation",
        "injection_path": "model.model.visual.pooler_output",
        "execution_scope": "seven batched states per image during visual prefill",
        "quantum_compute_dtype": (
            "torch.float32/torch.complex64" if core_kind != "classical" else None
        ),
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
