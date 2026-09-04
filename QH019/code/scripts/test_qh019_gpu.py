#!/usr/bin/env python3
"""CUDA-only static gate for QH-019 and its matched controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code/src"))

from quantum_qwen38.coherent_relation_mixer import CoherentRelationMixerAdapter
from quantum_qwen38.fidelity_relation_router import QH017Config


def finite_difference(adapter, tokens, grids, anchor) -> float:
    parameter = adapter.router.mixer.theta if hasattr(adapter.router.mixer, "theta") else adapter.router.mixer.affine
    adapter.zero_grad(set_to_none=True)
    adapter.set_text_anchor(anchor)
    residual = adapter.residual(tokens, grids).float()
    probe = torch.linspace(-0.3, 0.4, residual.numel(), device=residual.device).reshape_as(residual)
    loss = (residual * probe).sum()
    loss.backward()
    analytic = float(parameter.grad.flatten()[0])
    epsilon = 1.0e-3
    with torch.no_grad():
        original = float(parameter.flatten()[0])
        parameter.flatten()[0] = original + epsilon
        adapter.set_text_anchor(anchor)
        plus = float((adapter.residual(tokens, grids).float() * probe).sum())
        parameter.flatten()[0] = original - epsilon
        adapter.set_text_anchor(anchor)
        minus = float((adapter.residual(tokens, grids).float() * probe).sum())
        parameter.flatten()[0] = original
    return abs(analytic - (plus - minus) / (2.0 * epsilon))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH-019 static validation is CUDA-only")
    torch.manual_seed(20260829)
    device = torch.device("cuda:0")
    config = QH017Config(
        hidden_size=16,
        token_rank=8,
        object_rank=4,
        n_qubits=3,
        depth=2,
    )
    adapters = {
        kind: CoherentRelationMixerAdapter(config=config, core_kind=kind).to(device)
        for kind in ("quantum", "classical", "no_entanglement")
    }
    tokens = torch.randn(16, 16, device=device)
    anchor = torch.randn(1, 16, device=device)
    grids = [(1, 4, 4)]
    records = {}
    for kind, adapter in adapters.items():
        adapter.set_text_anchor(anchor)
        output = adapter(tokens, grids)
        output.square().mean().backward()
        missing_grad = [name for name, p in adapter.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        routing = adapter.last_routing_weights
        records[kind] = {
            "parameters": sum(p.numel() for p in adapter.parameters()),
            "missing_or_nonfinite_gradient_groups": missing_grad,
            "routing_min": float(routing.min()),
            "routing_max": float(routing.max()),
            "routing_sum": float(routing.sum()),
            "finite_difference_absolute_error": finite_difference(adapter, tokens, grids, anchor),
        }
    counts = {record["parameters"] for record in records.values()}
    cpu_rejected = False
    try:
        cpu_adapter = CoherentRelationMixerAdapter(config=config, core_kind="quantum")
        cpu_adapter.set_text_anchor(torch.randn(1, 16))
        cpu_adapter(torch.randn(16, 16), grids)
    except RuntimeError:
        cpu_rejected = True
    passed = (
        len(counts) == 1
        and cpu_rejected
        and all(not record["missing_or_nonfinite_gradient_groups"] for record in records.values())
        and all(abs(record["routing_sum"] - 1.0) < 1.0e-5 for record in records.values())
        and all(record["finite_difference_absolute_error"] < 0.02 for record in records.values())
    )
    result = {
        "status": "pass" if passed else "fail",
        "candidate": "QH-019",
        "device": str(device),
        "equal_parameter_count": len(counts) == 1,
        "cpu_path_rejected": cpu_rejected,
        "records": records,
    }
    output = PROJECT / "artifacts" / "qh019-gpu-validation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
