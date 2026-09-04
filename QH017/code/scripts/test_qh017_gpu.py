from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from quantum_qwen38.fidelity_relation_router import (
    FidelityRelationRouterAdapter,
    QH017Config,
)


def run(kind: str, config: QH017Config) -> dict:
    module = FidelityRelationRouterAdapter(config, core_kind=kind).cuda()
    for name in (
        "token_down",
        "object_down",
        "text_down",
        "key_down",
        "query_down",
        "pair_content",
        "token_up",
    ):
        getattr(module, name).to(dtype=torch.float32)
    module.set_text_anchor(torch.randn(2, config.hidden_size, device="cuda"))
    grid_shapes = [(1, 4, 4), (1, 2, 4)]
    x = torch.randn(24, config.hidden_size, device="cuda", requires_grad=True)
    probe = torch.linspace(-1.0, 1.0, x.numel(), device="cuda").reshape_as(x)

    def objective() -> torch.Tensor:
        return (module.residual(x, grid_shapes).float() * probe).sum()

    output = module(x, grid_shapes)
    loss = objective()
    loss.backward()
    gradients = {
        name: float(parameter.grad.detach().norm())
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    target_name = "router.affine" if kind == "classical" else "router.theta"
    target = dict(module.named_parameters())[target_name]
    flat_index = int(target.grad.detach().abs().argmax())
    analytic = float(target.grad.detach().reshape(-1)[flat_index])
    epsilon = 1.0e-2
    with torch.no_grad():
        original = target.reshape(-1)[flat_index].item()
        target.reshape(-1)[flat_index] = original + epsilon
        plus = float(objective())
        target.reshape(-1)[flat_index] = original - epsilon
        minus = float(objective())
        target.reshape(-1)[flat_index] = original
    finite_difference = (plus - minus) / (2.0 * epsilon)
    routing = module.last_routing_weights
    fidelities = module.last_fidelities
    return {
        "kind": kind,
        "parameter_count": sum(p.numel() for p in module.parameters()),
        "finite": bool(torch.isfinite(output).all()),
        "circuit_evaluations": module.last_circuit_evaluations,
        "routing_row_sums": routing.sum(dim=-1).detach().cpu().tolist(),
        "routing_min": float(routing.min()),
        "fidelity_min": float(fidelities.min()),
        "fidelity_max": float(fidelities.max()),
        "nonzero_gradient_fraction": sum(value > 0 for value in gradients.values())
        / len(gradients),
        "gradient_norms": gradients,
        "finite_difference": finite_difference,
        "analytic_gradient": analytic,
        "finite_difference_absolute_error": abs(finite_difference - analytic),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH-017 test requires CUDA")
    torch.manual_seed(20260829)
    config = QH017Config(
        hidden_size=64,
        token_rank=12,
        object_rank=8,
        n_qubits=4,
        depth=2,
    )
    results = [run(kind, config) for kind in ("quantum", "classical", "no_entanglement")]
    equal_parameters = len({row["parameter_count"] for row in results}) == 1
    gradients_ok = all(row["nonzero_gradient_fraction"] == 1.0 for row in results)
    finite_difference_ok = all(
        row["finite_difference_absolute_error"] < 1.0e-3 for row in results
    )
    routing_ok = all(
        row["routing_min"] >= 0.0
        and max(abs(value - 1.0) for value in row["routing_row_sums"]) < 1.0e-6
        and 0.0 <= row["fidelity_min"] <= row["fidelity_max"] <= 1.0
        for row in results
    )
    calls_ok = all(row["circuit_evaluations"] == 14 for row in results)
    missing_anchor_rejected = False
    try:
        FidelityRelationRouterAdapter(config).cuda()(
            torch.randn(24, 64, device="cuda"), [(1, 4, 4), (1, 2, 4)]
        )
    except RuntimeError:
        missing_anchor_rejected = True
    cpu_rejected = False
    try:
        cpu_module = FidelityRelationRouterAdapter(config)
        cpu_module.set_text_anchor(torch.randn(2, 64))
        cpu_module(torch.randn(24, 64), [(1, 4, 4), (1, 2, 4)])
    except RuntimeError:
        cpu_rejected = True
    artifact = {
        "status": "pass"
        if all(
            (
                equal_parameters,
                gradients_ok,
                finite_difference_ok,
                routing_ok,
                calls_ok,
                missing_anchor_rejected,
                cpu_rejected,
            )
        )
        else "fail",
        "device": torch.cuda.get_device_name(),
        "config": asdict(config),
        "equal_parameter_count": equal_parameters,
        "all_gradient_groups_nonzero": gradients_ok,
        "finite_difference_ok": finite_difference_ok,
        "routing_and_fidelity_valid": routing_ok,
        "seven_states_per_image": calls_ok,
        "missing_anchor_rejected": missing_anchor_rejected,
        "cpu_path_rejected": cpu_rejected,
        "results": results,
    }
    output = Path("artifacts/qh017-gpu-validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
