from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .quantum_residual import QuantumResidualMergerWrapper


@dataclass(frozen=True)
class ClassicalControlConfig:
    hidden_size: int = 5120
    bottleneck_size: int = 8
    depth: int = 2
    scale_init: float = 0.3
    scale_max: float = 1.0
    up_init_std: float = 0.01


class MatchedClassicalResidualAdapter(nn.Module):
    """Exact trainable-parameter-count control for QH-001b (81,961 parameters)."""

    def __init__(self, config: ClassicalControlConfig | None = None) -> None:
        super().__init__()
        self.config = config or ClassicalControlConfig()
        self.down = nn.Linear(self.config.hidden_size, self.config.bottleneck_size, bias=True)
        self.core_affine = nn.Parameter(
            torch.zeros(self.config.depth, self.config.bottleneck_size, 2, dtype=torch.float32)
        )
        self.up = nn.Linear(self.config.bottleneck_size, self.config.hidden_size, bias=False)
        probability = self.config.scale_init / self.config.scale_max
        self.raw_scale = nn.Parameter(torch.tensor(math.log(probability / (1.0 - probability))))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, mean=0.0, std=self.config.up_init_std)

    @property
    def scale(self) -> torch.Tensor:
        return self.config.scale_max * torch.sigmoid(self.raw_scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1])
        normalized = F.layer_norm(flat.float(), (self.config.hidden_size,)).to(flat.dtype)
        features = torch.tanh(self.down(normalized).float())
        for layer in range(self.config.depth):
            gain = 1.0 + 0.05 * self.core_affine[layer, :, 0]
            bias = 0.05 * self.core_affine[layer, :, 1]
            features = torch.tanh(features * gain + bias)
        residual = self.up(features.to(self.up.weight.dtype)).to(flat.dtype)
        output = flat + residual * self.scale.to(residual.dtype)
        return output.reshape(original_shape)

    def audit(self) -> dict:
        return {
            "candidate": "CC-001",
            "config": asdict(self.config),
            "current_scale": float(self.scale.detach().cpu()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def freeze_and_inject_cc001(model: nn.Module, config: ClassicalControlConfig | None = None) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual = model.model.visual
    old_merger = visual.merger
    reference = next(old_merger.parameters())
    adapter = MatchedClassicalResidualAdapter(config).to(device=reference.device)
    adapter.down.to(dtype=reference.dtype)
    adapter.up.to(dtype=reference.dtype)
    adapter.core_affine.data = adapter.core_affine.data.float()
    visual.merger = QuantumResidualMergerWrapper(old_merger, adapter)
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    return {
        "candidate": "CC-001",
        "injection_path": "model.model.visual.merger",
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }

