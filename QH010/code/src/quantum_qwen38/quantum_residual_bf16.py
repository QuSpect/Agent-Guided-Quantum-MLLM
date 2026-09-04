from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .quantum_residual import NativeStatevectorVQC, QuantumResidualMergerWrapper


@dataclass(frozen=True)
class QH001BF16Config:
    hidden_size: int = 5120
    n_qubits: int = 8
    depth: int = 2
    scale_init: float = 0.3
    scale_max: float = 1.0
    up_init_std: float = 0.01


class BF16QuantumResidualAdapter(nn.Module):
    """QH-001b: numerically visible, bounded quantum residual for a BF16 backbone."""

    def __init__(self, config: QH001BF16Config | None = None) -> None:
        super().__init__()
        self.config = config or QH001BF16Config()
        if not 0.0 < self.config.scale_init < self.config.scale_max:
            raise ValueError("scale_init must lie strictly between zero and scale_max")
        self.down = nn.Linear(self.config.hidden_size, self.config.n_qubits, bias=True)
        self.vqc = NativeStatevectorVQC(self.config.n_qubits, self.config.depth)
        self.up = nn.Linear(self.config.n_qubits, self.config.hidden_size, bias=False)
        initial_probability = self.config.scale_init / self.config.scale_max
        initial_logit = math.log(initial_probability / (1.0 - initial_probability))
        self.raw_scale = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, mean=0.0, std=self.config.up_init_std)

    @property
    def scale(self) -> torch.Tensor:
        return self.config.scale_max * torch.sigmoid(self.raw_scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        if original_shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected hidden size {self.config.hidden_size}, got {original_shape[-1]}"
            )
        flat = hidden_states.reshape(-1, original_shape[-1])
        normalized = F.layer_norm(flat.float(), (self.config.hidden_size,)).to(flat.dtype)
        encoded_angles = torch.pi * torch.tanh(self.down(normalized).float())
        expectations = self.vqc(encoded_angles)
        residual = self.up(expectations.to(self.up.weight.dtype)).to(flat.dtype)
        scaled = residual * self.scale.to(residual.dtype)
        return (flat + scaled).reshape(original_shape)

    def audit(self) -> dict:
        return {
            "candidate": "QH-001b",
            "config": asdict(self.config),
            "current_scale": float(self.scale.detach().cpu()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }


def freeze_and_inject_qh001b(model: nn.Module, config: QH001BF16Config | None = None) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        visual = model.model.visual
        old_merger = visual.merger
    except AttributeError as error:
        raise AttributeError("expected model.model.visual.merger in Qwen3.8 multimodal model") from error

    reference = next(old_merger.parameters())
    adapter = BF16QuantumResidualAdapter(config).to(device=reference.device)
    adapter.down.to(dtype=reference.dtype)
    adapter.up.to(dtype=reference.dtype)
    adapter.vqc.to(dtype=torch.float32)
    visual.merger = QuantumResidualMergerWrapper(old_merger, adapter)

    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    return {
        "candidate": "QH-001b",
        "injection_path": "model.model.visual.merger",
        "quantum_compute_dtype": "torch.float32/torch.complex64",
        "projection_dtype": str(reference.dtype),
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }

