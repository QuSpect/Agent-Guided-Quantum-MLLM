"""Paired GPU causal-control check for QH-001b without entanglement."""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

import torch

from quantum_qwen38.quantum_no_entanglement import NoEntanglementQuantumResidualAdapter
from quantum_qwen38.quantum_residual_bf16 import (
    BF16QuantumResidualAdapter,
    QH001BF16Config,
)


def relative_delta(model: torch.nn.Module, x: torch.Tensor) -> tuple[float, torch.Tensor]:
    with torch.no_grad():
        y = model(x)
    value = (y.float() - x.float()).norm() / x.float().norm().clamp_min(1e-12)
    return float(value), y


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260828)
    device = torch.device("cuda:0")
    config = QH001BF16Config(
        hidden_size=5120,
        n_qubits=8,
        depth=2,
        scale_init=0.3,
        up_init_std=0.01,
    )
    full = BF16QuantumResidualAdapter(config).to(device)
    control = NoEntanglementQuantumResidualAdapter(config).to(device)
    control.load_state_dict(full.state_dict(), strict=True)
    for model in (full, control):
        model.down.to(dtype=torch.bfloat16)
        model.up.to(dtype=torch.bfloat16)

    calibration = torch.randn(256, 5120, device=device, dtype=torch.bfloat16)
    heldout = torch.randn(256, 5120, device=device, dtype=torch.bfloat16)
    target = torch.randn_like(heldout)
    full_calibration_delta, _ = relative_delta(full, calibration)
    raw_control_delta, _ = relative_delta(control, calibration)
    old_scale = float(control.scale.detach())
    matched_scale = old_scale * full_calibration_delta / raw_control_delta
    matched_scale = max(1e-6, min(config.scale_max - 1e-6, matched_scale))
    with torch.no_grad():
        control.raw_scale.copy_(
            torch.tensor(
                math.log(matched_scale / (config.scale_max - matched_scale)),
                device=device,
            )
        )

    timings: dict[str, float] = {}
    gradients: dict[str, dict[str, float]] = {}
    outputs: dict[str, torch.Tensor] = {}
    for label, model in (("full_ring_cnot", full), ("rms_matched_no_entanglement", control)):
        samples = []
        for iteration in range(8):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            y = model(heldout)
            torch.nn.functional.mse_loss(y.float(), target.float()).backward()
            torch.cuda.synchronize()
            if iteration >= 3:
                samples.append((time.perf_counter() - started) * 1000)
        timings[label] = statistics.median(samples)
        gradients[label] = {
            name: float(parameter.grad.norm()) if parameter.grad is not None else 0.0
            for name, parameter in model.named_parameters()
        }
        with torch.no_grad():
            outputs[label] = model(heldout)

    full_delta, full_output = relative_delta(full, heldout)
    control_delta, control_output = relative_delta(control, heldout)
    result = {
        "control_id": "QH-001b-no-ent-rms-matched",
        "intervention": "delete ring-CNOT operations; calibrate one scalar gain",
        "trainable_parameters_each": sum(p.numel() for p in full.parameters()),
        "identical_state_dict_before_gain_matching": True,
        "calibration_batch": {
            "full_relative_delta": full_calibration_delta,
            "raw_control_relative_delta": raw_control_delta,
            "old_control_scale": old_scale,
            "calibrated_control_scale": float(control.scale.detach()),
        },
        "heldout_batch": {
            "full_relative_delta": full_delta,
            "control_relative_delta": control_delta,
            "relative_delta_ratio": control_delta / full_delta,
            "relative_output_difference": float(
                (full_output.float() - control_output.float()).norm()
                / full_output.float().norm()
            ),
        },
        "median_forward_backward_ms": timings,
        "gradient_norms": gradients,
        "all_parameter_groups_receive_gradient": all(
            value > 0 for group in gradients.values() for value in group.values()
        ),
        "device": torch.cuda.get_device_name(device),
    }
    output = Path("artifacts/qh001b-no-entanglement-control.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
