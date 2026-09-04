"""QH-019: coherent amplitude interference across six relation candidates."""

from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from .fidelity_relation_router import (
    ClassicalSimilarityRouter,
    FidelityRelationRouterAdapter,
    FidelityRelationVisionWrapper,
    QH017Config,
)
from .quantum_residual import NativeStatevectorVQC


class CoherentAmplitudeMixer(NativeStatevectorVQC):
    """Map six prior probabilities through one identity-near 3q state."""

    def __init__(self, depth: int, entangle: bool = True) -> None:
        super().__init__(n_qubits=3, depth=depth)
        self.theta = nn.Parameter(torch.empty(depth, 3, 2, dtype=torch.float32))
        nn.init.uniform_(self.theta, -0.02, 0.02)
        self.entangle = entangle
        self.last_padding_probability: torch.Tensor | None = None

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        if probabilities.shape != (6,):
            raise ValueError(f"expected six probabilities, got {tuple(probabilities.shape)}")
        if not probabilities.is_cuda:
            raise RuntimeError("QH-019 quantum simulation is GPU-only")
        normalized = probabilities.float().clamp_min(1.0e-8)
        normalized = normalized / normalized.sum()
        state = torch.zeros(1, 8, dtype=torch.complex64, device=normalized.device)
        state[0, :6] = normalized.sqrt().to(torch.complex64)
        for layer in range(self.depth):
            for wire in range(3):
                phase = self.theta[layer, wire, 0]
                if self.entangle:
                    target = (wire + 1) % 3
                    state = state.index_select(1, getattr(self, f"cnot_{wire}_{target}"))
                    state = self._apply_rz(state, phase, target)
                    state = state.index_select(1, getattr(self, f"cnot_{wire}_{target}"))
                else:
                    state = self._apply_rz(state, phase, wire)
            for wire in range(3):
                state = self._apply_ry(state, self.theta[layer, wire, 1], wire)
        born = state.abs().square().squeeze(0)
        self.last_padding_probability = born[6:].sum().detach()
        selected = born[:6].clamp_min(1.0e-8)
        return selected / selected.sum()


class ClassicalProbabilityMixer(nn.Module):
    """Equal-parameter classical probability mixer, identity at zero."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.affine = nn.Parameter(torch.empty(depth, 3, 2, dtype=torch.float32))
        nn.init.uniform_(self.affine, -0.02, 0.02)

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        if probabilities.shape != (6,):
            raise ValueError(f"expected six probabilities, got {tuple(probabilities.shape)}")
        logits = probabilities.float().clamp_min(1.0e-8).log()
        for layer in range(self.affine.shape[0]):
            coefficient = self.affine[layer].reshape(6)
            contrast = torch.tanh(torch.roll(logits, shifts=1) - logits)
            logits = logits + coefficient * contrast
        return torch.softmax(logits, dim=0)


class CoherentRelationRouter(nn.Module):
    def __init__(self, depth: int, core_kind: str, temperature: float) -> None:
        super().__init__()
        self.base = ClassicalSimilarityRouter(n_features=3, depth=depth)
        if core_kind == "quantum":
            self.mixer = CoherentAmplitudeMixer(depth=depth, entangle=True)
        elif core_kind == "no_entanglement":
            self.mixer = CoherentAmplitudeMixer(depth=depth, entangle=False)
        elif core_kind == "classical":
            self.mixer = ClassicalProbabilityMixer(depth=depth)
        else:
            raise ValueError(f"unsupported core_kind: {core_kind}")
        self.core_kind = core_kind
        self.temperature = temperature

    def forward(self, query_angles: torch.Tensor, key_angles: torch.Tensor) -> torch.Tensor:
        similarities = self.base(query_angles, key_angles)
        prior = torch.softmax((2.0 * similarities - 1.0) / self.temperature, dim=0)
        return self.mixer(prior)


class CoherentRelationMixerAdapter(FidelityRelationRouterAdapter):
    def __init__(self, config: QH017Config | None = None, core_kind: str = "quantum") -> None:
        config = config or QH017Config(n_qubits=3)
        if config.n_qubits != 3:
            raise ValueError("QH-019 requires exactly three qubits for eight amplitudes")
        super().__init__(config=config, core_kind="classical")
        self.core_kind = core_kind
        self.router = CoherentRelationRouter(
            depth=config.depth,
            core_kind=core_kind,
            temperature=config.routing_temperature,
        )

    def _routing_weights(self, scores: torch.Tensor) -> torch.Tensor:
        return scores / scores.sum().clamp_min(1.0e-8)

    def audit(self) -> dict:
        candidate = {
            "quantum": "QH-019",
            "classical": "CC-019",
            "no_entanglement": "QH-019-no-ent",
        }[self.core_kind]
        return {
            "candidate": candidate,
            "config": asdict(self.config),
            "core_kind": self.core_kind,
            "routing": "classical relation prior encoded as one 3q amplitude state",
            "quantum_role": "coherent interference directly emits final six-way Born route",
            "identity_control": "set all mixer theta to zero",
            "execution": "one eight-amplitude state per image during visual prefill",
            "current_scale": float(self.scale.detach().cpu()),
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters()),
        }


def freeze_and_inject_qh019(
    model: nn.Module,
    core_kind: str = "quantum",
    config: QH017Config | None = None,
) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.model.visual
    reference = next(visual.merger.parameters())
    adapter = CoherentRelationMixerAdapter(config=config, core_kind=core_kind).to(
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
        "quantum": "QH-019",
        "classical": "CC-019",
        "no_entanglement": "QH-019-no-ent",
    }[core_kind]
    return {
        "candidate": candidate,
        "parent": "QH-018 classical relation trunk with a new coherent causal role",
        "injection_path": "model.model.visual.pooler_output",
        "execution_scope": "one 3q state during visual prefill",
        "quantum_compute_dtype": (
            "torch.float32/torch.complex64" if core_kind != "classical" else None
        ),
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
