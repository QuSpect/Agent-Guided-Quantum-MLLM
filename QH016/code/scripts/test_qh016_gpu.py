from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from quantum_qwen38.question_routed_relations import (
    QH016Config,
    QuestionRoutedRelationAdapter,
)


def run(
    kind: str,
    x: torch.Tensor,
    grid_shapes: list[tuple[int, int, int]],
    config: QH016Config,
) -> dict:
    module = QuestionRoutedRelationAdapter(config, core_kind=kind).cuda()
    for name in (
        "token_down",
        "object_down",
        "text_down",
        "relation_down",
        "gate_up",
        "token_up",
    ):
        getattr(module, name).to(dtype=torch.float32)
    module.set_text_anchor(torch.randn(2, config.hidden_size, device="cuda"))
    y = module(x, grid_shapes)
    probe = torch.linspace(-1.0, 1.0, y.numel(), device=y.device).reshape_as(y)

    def objective() -> torch.Tensor:
        return (module.residual(x, grid_shapes).float() * probe).sum()

    loss = objective()
    loss.backward()
    gradients = {
        name: float(parameter.grad.detach().norm())
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    target_name = "core.affine" if kind == "classical" else "core.theta"
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
    return {
        "kind": kind,
        "parameter_count": sum(p.numel() for p in module.parameters()),
        "finite": bool(torch.isfinite(y).all()),
        "circuit_evaluations": module.last_circuit_evaluations,
        "routing_row_sums": routing.sum(dim=-1).detach().cpu().tolist(),
        "routing_min": float(routing.min()),
        "nonzero_gradient_fraction": sum(v > 0 for v in gradients.values()) / len(gradients),
        "gradient_norms": gradients,
        "finite_difference": finite_difference,
        "analytic_gradient": analytic,
        "finite_difference_absolute_error": abs(finite_difference - analytic),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH-016 test requires CUDA")
    torch.manual_seed(20260829)
    config = QH016Config(
        hidden_size=64,
        token_rank=12,
        object_rank=8,
        n_qubits=4,
        depth=2,
    )
    grid_shapes = [(1, 4, 4), (1, 2, 4)]
    results = [
        run(
            kind,
            torch.randn(24, 64, device="cuda", requires_grad=True),
            grid_shapes,
            config,
        )
        for kind in ("quantum", "classical", "no_entanglement")
    ]
    equal_parameters = len({row["parameter_count"] for row in results}) == 1
    gradients_ok = all(row["nonzero_gradient_fraction"] == 1.0 for row in results)
    finite_difference_ok = all(
        row["finite_difference_absolute_error"] < 1.0e-3 for row in results
    )
    calls_ok = all(row["circuit_evaluations"] == 12 for row in results)
    routing_ok = all(
        min(row["routing_min"], 1.0) >= 0.0
        and max(abs(value - 1.0) for value in row["routing_row_sums"]) < 1.0e-6
        for row in results
    )
    missing_anchor_rejected = False
    try:
        QuestionRoutedRelationAdapter(config, core_kind="quantum").cuda()(
            torch.randn(24, 64, device="cuda"), grid_shapes
        )
    except RuntimeError:
        missing_anchor_rejected = True
    invalid_grid_rejected = False
    try:
        invalid = QuestionRoutedRelationAdapter(config, core_kind="quantum").cuda()
        invalid.set_text_anchor(torch.randn(2, 64, device="cuda"))
        invalid(torch.randn(24, 64, device="cuda"), [(1, 4, 4), (1, 2, 3)])
    except ValueError:
        invalid_grid_rejected = True
    cpu_rejected = False
    try:
        cpu_module = QuestionRoutedRelationAdapter(config, core_kind="quantum")
        cpu_module.set_text_anchor(torch.randn(2, 64))
        cpu_module(torch.randn(24, 64), grid_shapes)
    except RuntimeError:
        cpu_rejected = True
    artifact = {
        "status": "pass"
        if all(
            (
                equal_parameters,
                gradients_ok,
                finite_difference_ok,
                calls_ok,
                routing_ok,
                missing_anchor_rejected,
                invalid_grid_rejected,
                cpu_rejected,
            )
        )
        else "fail",
        "device": torch.cuda.get_device_name(),
        "config": asdict(config),
        "equal_parameter_count": equal_parameters,
        "all_gradient_groups_nonzero": gradients_ok,
        "finite_difference_ok": finite_difference_ok,
        "six_relations_per_image": calls_ok,
        "routing_is_probability_simplex": routing_ok,
        "missing_anchor_rejected": missing_anchor_rejected,
        "invalid_grid_rejected": invalid_grid_rejected,
        "cpu_path_rejected": cpu_rejected,
        "results": results,
    }
    output = Path("artifacts/qh016-gpu-validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
