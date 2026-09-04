#!/usr/bin/env python3
"""Static CUDA validation for QH-008 and its matched controls."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.quantum_gated_low_rank import (
    GPUStatevectorVQC,
    QH008Config,
    QuantumGatedLowRankAdapter,
)


def gradient_groups(module: QuantumGatedLowRankAdapter) -> dict[str, float]:
    groups = {}
    for name, parameter in module.named_parameters():
        group = name.split(".")[0]
        value = 0.0 if parameter.grad is None else float(parameter.grad.float().norm())
        groups[group] = groups.get(group, 0.0) + value
    return groups


def benchmark(module, x, repeats: int = 7) -> tuple[float, int]:
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        module.zero_grad(set_to_none=True)
        started = time.perf_counter()
        output = module(x)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        times.append(1000.0 * (time.perf_counter() - started))
    return statistics.median(times[2:]), torch.cuda.max_memory_allocated()


def finite_difference_check(device: torch.device) -> dict:
    """Check the complete FP32 quantum-gate-to-low-rank computation graph."""

    torch.manual_seed(911)
    config = QH008Config(hidden_size=16, rank=8, n_qubits=8, depth=2)
    module = QuantumGatedLowRankAdapter(config, core_kind="quantum").to(device).float()
    x = torch.randn(3, 16, device=device)
    probe = torch.randn(3, 16, device=device)

    def objective() -> torch.Tensor:
        # Measure before the residual is added to the O(1) identity path. Tiny
        # perturbations can otherwise be rounded away by the FP32 addition.
        return (module.residual(x) * probe).sum()

    module.zero_grad(set_to_none=True)
    objective().backward()
    targets = {
        "theta": (module.core.theta, (0, 0, 0)),
        "quantum_down": (module.quantum_down.weight, (0, 0)),
    }
    epsilon = 1.0e-3
    results = {}
    with torch.no_grad():
        for name, (parameter, index) in targets.items():
            analytic = float(parameter.grad[index])
            original = float(parameter[index])
            parameter[index] = original + epsilon
            positive = float(objective())
            parameter[index] = original - epsilon
            negative = float(objective())
            parameter[index] = original
            numeric = (positive - negative) / (2.0 * epsilon)
            relative_error = abs(analytic - numeric) / max(
                1.0e-7, abs(analytic) + abs(numeric)
            )
            results[name] = {
                "analytic": analytic,
                "numeric": numeric,
                "relative_error": relative_error,
            }
    return results


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH-008 validation is GPU-only")
    torch.manual_seed(20260829)
    torch.cuda.manual_seed_all(20260829)
    device = torch.device("cuda:0")
    x = torch.randn(256, 5120, device=device, dtype=torch.bfloat16)

    torch.manual_seed(71)
    quantum = QuantumGatedLowRankAdapter(core_kind="quantum").to(device)
    quantum.local_down.to(dtype=torch.bfloat16)
    quantum.quantum_down.to(dtype=torch.bfloat16)
    quantum.gate_up.to(dtype=torch.bfloat16)
    quantum.local_up.to(dtype=torch.bfloat16)
    torch.manual_seed(71)
    classical = QuantumGatedLowRankAdapter(core_kind="classical").to(device)
    classical.local_down.to(dtype=torch.bfloat16)
    classical.quantum_down.to(dtype=torch.bfloat16)
    classical.gate_up.to(dtype=torch.bfloat16)
    classical.local_up.to(dtype=torch.bfloat16)
    no_ent = QuantumGatedLowRankAdapter(core_kind="no_entanglement").to(device)
    no_ent.local_down.to(dtype=torch.bfloat16)
    no_ent.quantum_down.to(dtype=torch.bfloat16)
    no_ent.gate_up.to(dtype=torch.bfloat16)
    no_ent.local_up.to(dtype=torch.bfloat16)

    parameter_counts = {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in {
            "quantum": quantum,
            "classical": classical,
            "no_entanglement": no_ent,
        }.items()
    }
    outputs = {}
    gradients = {}
    for name, module in {
        "quantum": quantum,
        "classical": classical,
        "no_entanglement": no_ent,
    }.items():
        module.zero_grad(set_to_none=True)
        output = module(x)
        loss = output.float().square().mean() + 0.01 * output.float().mean()
        loss.backward()
        residual = output.float() - x.float()
        outputs[name] = {
            "changed_fraction": float((output != x).float().mean()),
            "relative_residual_rms": float(
                (
                    residual.square().mean().sqrt()
                    / x.float().square().mean().sqrt()
                ).detach()
            ),
            "finite": bool(torch.isfinite(output).all()),
        }
        gradients[name] = gradient_groups(module)

    quantum_ms, quantum_peak = benchmark(quantum, x)
    classical_ms, classical_peak = benchmark(classical, x)
    finite_difference = finite_difference_check(device)
    cpu_rejected = False
    try:
        GPUStatevectorVQC()(torch.zeros(1, 8))
    except RuntimeError:
        cpu_rejected = True

    required_groups = {"local_down", "quantum_down", "core", "gate_up", "local_up", "raw_scale"}
    nonzero_groups = {
        name: sorted(group for group, value in values.items() if value > 0.0)
        for name, values in gradients.items()
    }
    result = {
        "status": "pass"
        if len(set(parameter_counts.values())) == 1
        and cpu_rejected
        and all(outputs[name]["finite"] for name in outputs)
        and required_groups.issubset(nonzero_groups["quantum"])
        and required_groups.issubset(nonzero_groups["classical"])
        and all(value["relative_error"] <= 0.02 for value in finite_difference.values())
        else "fail",
        "candidate": "QH-008",
        "parameter_counts": parameter_counts,
        "outputs": outputs,
        "gradient_group_norm_sums": gradients,
        "nonzero_gradient_groups": nonzero_groups,
        "median_forward_backward_ms": {
            "quantum": quantum_ms,
            "classical": classical_ms,
        },
        "peak_allocated_bytes": {
            "quantum": quantum_peak,
            "classical": classical_peak,
        },
        "cpu_quantum_path_rejected": cpu_rejected,
        "finite_difference": finite_difference,
        "finite_difference_relative_error_max": 0.02,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    output_path = PROJECT_DIR / "artifacts" / "qh008-gpu-validation.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
