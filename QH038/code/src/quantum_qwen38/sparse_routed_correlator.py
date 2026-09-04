"""QH-013: sparse novelty-routed quantum correlator for visual prefill.

The classical low-rank content path processes every visual token.  A frozen,
parameter-free router selects a small number of tokens per image for the
quantum/classical/no-entanglement gate, so statevector simulation is sparse.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .correlation_gated_low_rank import CorrelationGatedLowRankAdapter, QH009Config


@dataclass(frozen=True)
class QH013Config(QH009Config):
    tokens_per_image: int = 8
    scale_init: float = 0.05
    local_up_init_std: float = 0.001


class SparseRoutedCorrelationAdapter(CorrelationGatedLowRankAdapter):
    """Run the expensive gate only on frozen-router top-k visual tokens."""

    def __init__(
        self,
        config: QH013Config | None = None,
        core_kind: str = "quantum",
    ) -> None:
        super().__init__(config or QH013Config(), core_kind=core_kind)
        self.last_selected_tokens = 0
        self.last_total_tokens = 0

    @staticmethod
    def _novelty_indices(
        normalized: torch.Tensor,
        split_sizes: list[int],
        tokens_per_image: int,
    ) -> torch.Tensor:
        """Select tokens farthest from their image mean; no trainable router."""
        selected: list[torch.Tensor] = []
        cursor = 0
        for size in split_sizes:
            image = normalized[cursor : cursor + size]
            count = min(tokens_per_image, size)
            score = (image.float() - image.float().mean(dim=0, keepdim=True)).square().mean(-1)
            local = torch.topk(score, k=count, largest=True, sorted=True).indices
            selected.append(local + cursor)
            cursor += size
        return torch.cat(selected, dim=0)

    def residual(self, hidden_states: torch.Tensor, split_sizes: list[int]) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError(f"expected [visual_tokens, hidden], got {tuple(hidden_states.shape)}")
        if sum(split_sizes) != hidden_states.shape[0]:
            raise ValueError("split_sizes must exactly cover merged visual tokens")
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected hidden size {self.config.hidden_size}, got {hidden_states.shape[-1]}"
            )

        normalized = F.layer_norm(
            hidden_states.float(), (self.config.hidden_size,)
        ).to(hidden_states.dtype)
        local_features = F.gelu(self.local_down(normalized).float())
        indices = self._novelty_indices(
            normalized.detach(), split_sizes, self.config.tokens_per_image
        )
        routed = normalized.index_select(0, indices)
        angles = torch.pi * torch.tanh(self.quantum_down(routed).float())
        observables = self.core(angles)
        selected_gate = torch.tanh(
            self.gate_up(observables.to(self.gate_up.weight.dtype)).float()
        )
        gate = torch.zeros_like(local_features)
        gate = gate.index_copy(0, indices, selected_gate)
        modulated = local_features * (1.0 + self.config.gate_strength * gate)
        residual = self.local_up(modulated.to(self.local_up.weight.dtype)).to(hidden_states.dtype)
        self.last_selected_tokens = int(indices.numel())
        self.last_total_tokens = int(hidden_states.shape[0])
        return residual * self.scale.to(residual.dtype)

    def forward(self, hidden_states: torch.Tensor, split_sizes: list[int]) -> torch.Tensor:
        return hidden_states + self.residual(hidden_states, split_sizes)

    def audit(self) -> dict:
        candidate = {
            "quantum": "QH-013",
            "classical": "CC-013",
            "no_entanglement": "QH-013-no-ent",
        }[self.core_kind]
        return {
            "candidate": candidate,
            "config": asdict(self.config),
            "core_kind": self.core_kind,
            "router": "frozen per-image top-k squared distance from normalized image mean",
            "readout": "8 local-Z + 8 adjacent-ZZ observables",
            "current_scale": float(self.scale.detach().cpu()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }


class SparseRoutedVisionWrapper(nn.Module):
    def __init__(self, frozen_visual: nn.Module, adapter: SparseRoutedCorrelationAdapter) -> None:
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
            raise ValueError("grid_thw is required for per-image sparse routing")
        output = self.frozen_visual(*args, grid_thw=grid_thw, **kwargs)
        split_sizes = (grid_thw.prod(-1) // self.spatial_merge_size**2).detach().cpu().tolist()
        output.pooler_output = self.adapter(output.pooler_output, split_sizes)
        return output


def freeze_and_inject_qh013(
    model: nn.Module,
    core_kind: str = "quantum",
    config: QH013Config | None = None,
) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.model.visual
    reference = next(visual.merger.parameters())
    adapter = SparseRoutedCorrelationAdapter(config, core_kind=core_kind).to(
        device=reference.device
    )
    adapter.local_down.to(dtype=reference.dtype)
    adapter.quantum_down.to(dtype=reference.dtype)
    adapter.gate_up.to(dtype=reference.dtype)
    adapter.local_up.to(dtype=reference.dtype)
    adapter.core.to(dtype=torch.float32)
    model.model.visual = SparseRoutedVisionWrapper(visual, adapter)
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    candidate = {
        "quantum": "QH-013",
        "classical": "CC-013",
        "no_entanglement": "QH-013-no-ent",
    }[core_kind]
    return {
        "candidate": candidate,
        "parent": "QH-009 + MEDQUA-inspired sparse routing",
        "injection_path": "model.model.visual.pooler_output",
        "execution_scope": "visual prefill only; routed top-k tokens per image",
        "quantum_compute_dtype": (
            "torch.float32/torch.complex64" if core_kind != "classical" else None
        ),
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
