"""QH-018: conservative quantum-fidelity correction to a classical router."""

from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from .fidelity_relation_router import (
    ClassicalSimilarityRouter,
    FidelityRelationRouterAdapter,
    FidelityRelationVisionWrapper,
    FidelityStatevectorRouter,
    QH017Config,
)


class ResidualFidelityRouter(nn.Module):
    """A stable classical route plus a small centered residual similarity route."""

    def __init__(self, n_qubits: int, depth: int, residual_kind: str) -> None:
        super().__init__()
        self.base = ClassicalSimilarityRouter(n_qubits, depth)
        if residual_kind == "quantum":
            self.residual = FidelityStatevectorRouter(n_qubits, depth, entangle=True)
        elif residual_kind == "no_entanglement":
            self.residual = FidelityStatevectorRouter(n_qubits, depth, entangle=False)
        elif residual_kind == "classical":
            self.residual = ClassicalSimilarityRouter(n_qubits, depth)
        else:
            raise ValueError(f"unsupported residual_kind: {residual_kind}")
        self.residual_kind = residual_kind
        self.mix = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(
        self, query_angles: torch.Tensor, key_angles: torch.Tensor
    ) -> torch.Tensor:
        base_probability = self.base(query_angles, key_angles).clamp(1.0e-4, 1.0 - 1.0e-4)
        correction = self.residual(query_angles, key_angles)
        correction = correction - correction.mean(dim=0, keepdim=True)
        base_logit = torch.logit(base_probability)
        return torch.sigmoid(base_logit + self.mix * correction)


class ResidualFidelityRelationAdapter(FidelityRelationRouterAdapter):
    def __init__(
        self,
        config: QH017Config | None = None,
        residual_kind: str = "quantum",
    ) -> None:
        super().__init__(config=config, core_kind="classical")
        self.core_kind = residual_kind
        self.router = ResidualFidelityRouter(
            self.config.n_qubits, self.config.depth, residual_kind
        )

    def audit(self) -> dict:
        candidate = {
            "quantum": "QH-018",
            "classical": "CC-018",
            "no_entanglement": "QH-018-no-ent",
        }[self.core_kind]
        return {
            "candidate": candidate,
            "config": asdict(self.config),
            "core_kind": self.core_kind,
            "routing": "classical similarity plus centered learnable residual route",
            "quantum_role": "small fidelity-logit correction, initialized at 0.05",
            "relations": "six deterministic directed 2x2 spatial-region pairs",
            "execution": "base classical route plus one query and six residual key states",
            "current_mix": float(self.router.mix.detach().cpu()),
            "current_scale": float(self.scale.detach().cpu()),
            "trainable_parameters": sum(p.numel() for p in self.parameters()),
        }


def freeze_and_inject_qh018(
    model: nn.Module,
    residual_kind: str = "quantum",
    config: QH017Config | None = None,
) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.model.visual
    reference = next(visual.merger.parameters())
    adapter = ResidualFidelityRelationAdapter(
        config=config, residual_kind=residual_kind
    ).to(device=reference.device)
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
        "quantum": "QH-018",
        "classical": "CC-018",
        "no_entanglement": "QH-018-no-ent",
    }[residual_kind]
    return {
        "candidate": candidate,
        "parent": "QH-017 preserving the effective CC-017 base route",
        "injection_path": "model.model.visual.pooler_output",
        "execution_scope": "classical base plus seven residual states during visual prefill",
        "quantum_compute_dtype": (
            "torch.float32/torch.complex64"
            if residual_kind != "classical"
            else None
        ),
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
