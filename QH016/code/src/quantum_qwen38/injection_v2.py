from __future__ import annotations

import torch
from torch import nn

from .quantum_residual import QH001Config, QuantumResidualAdapter, QuantumResidualMergerWrapper


def freeze_and_inject_qh001_v2(model: nn.Module, config: QH001Config | None = None) -> dict:
    """Freeze Qwen3.8 and inject QH-001 while retaining FP32 statevector arithmetic."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    try:
        visual = model.model.visual
        old_merger = visual.merger
    except AttributeError as error:
        raise AttributeError("expected model.model.visual.merger in Qwen3.8 multimodal model") from error

    reference = next(old_merger.parameters())
    adapter = QuantumResidualAdapter(config).to(device=reference.device)
    adapter.down.to(dtype=reference.dtype)
    adapter.up.to(dtype=reference.dtype)
    adapter.vqc.to(dtype=torch.float32)
    visual.merger = QuantumResidualMergerWrapper(old_merger, adapter)

    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    return {
        "candidate": "QH-001-v2",
        "injection_path": "model.model.visual.merger",
        "quantum_compute_dtype": "torch.float32/torch.complex64",
        "projection_dtype": str(reference.dtype),
        "adapter": adapter.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }

