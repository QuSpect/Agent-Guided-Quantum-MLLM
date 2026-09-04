"""QH-026 depth-3 re-uploading butterfly mutation of the QH-024 replacement."""

from __future__ import annotations

import math

import torch
from torch import nn

from .hyqut_replacement import (
    ClassicalMatchedCore,
    HybridValueReplacement,
    HyQuTStatevectorCore,
    QH024Config,
    TensorContractionHyQuTCore,
)


class QH026StatevectorCore(HyQuTStatevectorCore):
    """Depth-3 data re-uploading with log-distance butterfly CNOT mixing."""

    def __init__(self, n_qubits: int, depth: int, *, seed: int, entangle: bool = True) -> None:
        super().__init__(n_qubits, depth, seed=seed, entangle=entangle)
        basis = torch.arange(self.width, dtype=torch.long)
        self.offsets = tuple(value for value in (1, 2, 4) if value < n_qubits)
        for offset in self.offsets:
            for control in range(n_qubits):
                target = (control + offset) % n_qubits
                name = f"cnot_{control}_{target}"
                if not hasattr(self, name):
                    active = ((basis >> control) & 1).to(torch.long)
                    self.register_buffer(name, basis ^ (active << target), persistent=False)

    @staticmethod
    def _rms(values: torch.Tensor) -> torch.Tensor:
        return values / torch.sqrt(values.float().square().mean(dim=-1, keepdim=True) + 1.0e-6)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        if not encoded.is_cuda or not self.theta.is_cuda:
            raise RuntimeError("QH-026 exact statevector simulation is CUDA-only")
        if encoded.ndim != 2 or encoded.shape[1] != 2 * self.n_qubits:
            raise ValueError("encoded feature shape mismatch")
        state = torch.full(
            (encoded.shape[0], self.width),
            1.0 / math.sqrt(self.width),
            device=encoded.device,
            dtype=torch.complex64,
        )
        data_angles = math.pi * torch.tanh(self._rms(encoded.float()))
        for layer in range(self.depth):
            # Re-upload data before every variational block.
            for wire in range(self.n_qubits):
                state = self._apply_ry(state, data_angles[:, wire], wire)
                state = self._apply_rz(state, data_angles[:, self.n_qubits + wire], wire)
            for wire in range(self.n_qubits):
                state = self._apply_rz(state, self.theta[layer, wire, 0], wire)
                state = self._apply_ry(state, self.theta[layer, wire, 1], wire)
                state = self._apply_rz(state, self.theta[layer, wire, 2], wire)
            for offset in self.offsets:
                for control in range(self.n_qubits):
                    state = self._apply_cnot(state, control, (control + offset) % self.n_qubits)
        probabilities = state.real.square() + state.imag.square()
        self.circuit_calls += 1
        return self._rms(probabilities @ self.z_signs)


class QH026TensorCore(QH026StatevectorCore, TensorContractionHyQuTCore):
    """Same circuit with independent tensor-axis gate application for DQ audit."""

    def _apply_ry(self, state: torch.Tensor, angle: torch.Tensor, wire: int) -> torch.Tensor:
        return TensorContractionHyQuTCore._apply_ry(self, state, angle, wire)

    def _apply_rz(self, state: torch.Tensor, angle: torch.Tensor, wire: int) -> torch.Tensor:
        return TensorContractionHyQuTCore._apply_rz(self, state, angle, wire)

    def _apply_cnot(self, state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        return TensorContractionHyQuTCore._apply_cnot(self, state, control, target)


class QH026ClassicalCore(ClassicalMatchedCore):
    """Equal-parameter classical re-uploading/butterfly control."""

    @staticmethod
    def _rms(values: torch.Tensor) -> torch.Tensor:
        return values / torch.sqrt(values.float().square().mean(dim=-1, keepdim=True) + 1.0e-6)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        if not encoded.is_cuda or not self.theta.is_cuda:
            raise RuntimeError("CC-026 matched control is GPU-only")
        left, right = self._rms(encoded.float()).chunk(2, dim=-1)
        values = torch.zeros_like(left)
        for layer in range(self.depth):
            a, b, c = self.theta[layer].unbind(dim=-1)
            data = torch.tanh(left * torch.cos(a) + right * torch.sin(a))
            mixed = data
            for offset in (1, 2, 4):
                if offset < self.n_qubits:
                    mixed = mixed + torch.roll(values + data, offset, dims=-1) / 3.0
            values = torch.tanh(mixed + b + c * left * right)
        return self._rms(values)


class QH026ValueReplacement(HybridValueReplacement):
    def __init__(self, *args, core_kind: str = "quantum", **kwargs) -> None:
        super().__init__(*args, core_kind="classical", **kwargs)
        config = self.config
        device = self.gamma.device
        if core_kind == "quantum":
            core: nn.Module = QH026StatevectorCore(
                config.n_qubits, config.depth, seed=config.parameter_seed + 1, entangle=True
            )
        elif core_kind == "dequantized":
            core = QH026TensorCore(
                config.n_qubits, config.depth, seed=config.parameter_seed + 1, entangle=True
            )
        elif core_kind == "no_entanglement":
            core = QH026StatevectorCore(
                config.n_qubits, config.depth, seed=config.parameter_seed + 1, entangle=False
            )
        elif core_kind == "classical":
            core = QH026ClassicalCore(
                config.n_qubits, config.depth, seed=config.parameter_seed + 1
            )
        else:
            raise ValueError(core_kind)
        self.core = core.to(device)
        self.core_kind = core_kind

    @classmethod
    def from_dense(
        cls, projection: nn.Linear, config: QH024Config, *, core_kind: str = "quantum"
    ) -> "QH026ValueReplacement":
        if projection.bias is not None:
            raise ValueError("QH-026 requires a bias-free dense projection")
        weight = projection.weight.detach()
        u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
        result = cls(
            config,
            vh[: config.svd_rank].to(weight.dtype),
            (u[:, : config.svd_rank] * s[: config.svd_rank]).to(weight.dtype),
            weight,
            core_kind=core_kind,
        )
        del u, s, vh
        torch.cuda.empty_cache()
        return result

    def audit(self) -> dict:
        payload = super().audit()
        payload.update({
            "candidate": {
                "quantum": "QH-026",
                "dequantized": "DQ-026",
                "no_entanglement": "QH-026-no-ent",
                "classical": "CC-026",
            }[self.core_kind],
            "architecture": (
                "frozen rank-512 SVD backbone + depth-3 parameter-free RMS-normalized "
                "data-reuploading log-distance butterfly residual"
            ),
            "data_reuploads": self.config.depth,
            "entanglement_offsets": list(getattr(self.core, "offsets", ())),
        })
        return payload


def qh026_config() -> QH024Config:
    return QH024Config(depth=3, parameter_seed=20260840)


def freeze_and_replace_qh026(
    model: nn.Module, *, core_kind: str = "quantum"
) -> dict:
    config = qh026_config()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    layer = model.model.language_model.layers[config.layer_index]
    projection = getattr(layer.self_attn, config.target_projection)
    replacement = QH026ValueReplacement.from_dense(projection, config, core_kind=core_kind)
    setattr(layer.self_attn, config.target_projection, replacement)
    deployed_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
    return {
        "candidate": replacement.audit()["candidate"],
        "base_model_parameters": base_parameters,
        "deployed_model_parameters": deployed_parameters,
        "net_model_parameter_reduction": base_parameters - deployed_parameters,
        "replacement": replacement.audit(),
        "trainable_parameter_count": sum(count for _, count in trainable),
        "trainable_tensors": trainable,
    }
