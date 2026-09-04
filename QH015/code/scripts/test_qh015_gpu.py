from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from quantum_qwen38.question_conditioned_film import (
    QH015Config,
    QuestionConditionedFiLMAdapter,
)


def run(kind: str, x: torch.Tensor, split_sizes: list[int], config: QH015Config) -> dict:
    module = QuestionConditionedFiLMAdapter(config, core_kind=kind).cuda()
    for name in ("token_down", "visual_down", "text_down", "gate_up", "token_up"):
        getattr(module, name).to(dtype=torch.float32)
    module.set_text_anchor(torch.randn(1, config.hidden_size, device="cuda"))
    y = module(x, split_sizes)
    probe = torch.linspace(-1.0, 1.0, y.numel(), device=y.device).reshape_as(y)

    def objective() -> torch.Tensor:
        return (module.residual(x, split_sizes).float() * probe).sum()

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
    return {
        "kind": kind,
        "parameter_count": sum(p.numel() for p in module.parameters()),
        "finite": bool(torch.isfinite(y).all()),
        "circuit_calls": module.last_circuit_calls,
        "nonzero_gradient_fraction": sum(v > 0 for v in gradients.values()) / len(gradients),
        "gradient_norms": gradients,
        "finite_difference": finite_difference,
        "analytic_gradient": analytic,
        "finite_difference_absolute_error": abs(finite_difference - analytic),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH-015 test requires CUDA")
    torch.manual_seed(20260829)
    config = QH015Config(hidden_size=64, rank=12, n_qubits=4, depth=2)
    split_sizes = [10, 6]
    results = [
        run(kind, torch.randn(16, 64, device="cuda", requires_grad=True), split_sizes, config)
        for kind in ("quantum", "classical", "no_entanglement")
    ]
    equal_parameters = len({row["parameter_count"] for row in results}) == 1
    gradients_ok = all(row["nonzero_gradient_fraction"] == 1.0 for row in results)
    finite_difference_ok = all(
        row["finite_difference_absolute_error"] < 1.0e-3 for row in results
    )
    calls_ok = all(row["circuit_calls"] == 2 for row in results)
    missing_anchor_rejected = False
    try:
        QuestionConditionedFiLMAdapter(config, core_kind="quantum").cuda()(
            torch.randn(16, 64, device="cuda"), split_sizes
        )
    except RuntimeError:
        missing_anchor_rejected = True
    cpu_rejected = False
    try:
        cpu_module = QuestionConditionedFiLMAdapter(config, core_kind="quantum")
        cpu_module.set_text_anchor(torch.randn(1, 64))
        cpu_module(torch.randn(16, 64), split_sizes)
    except RuntimeError:
        cpu_rejected = True
    artifact = {
        "status": "pass" if all((
            equal_parameters,
            gradients_ok,
            finite_difference_ok,
            calls_ok,
            missing_anchor_rejected,
            cpu_rejected,
        )) else "fail",
        "device": torch.cuda.get_device_name(),
        "config": asdict(config),
        "equal_parameter_count": equal_parameters,
        "all_gradient_groups_nonzero": gradients_ok,
        "finite_difference_ok": finite_difference_ok,
        "one_call_per_image": calls_ok,
        "missing_anchor_rejected": missing_anchor_rejected,
        "cpu_path_rejected": cpu_rejected,
        "results": results,
    }
    output = Path("artifacts/qh015-gpu-validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
