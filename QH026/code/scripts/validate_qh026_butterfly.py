#!/usr/bin/env python3
"""Static CUDA, DQ, gradient, and matched-budget gates for QH-026."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code/src"))

from quantum_qwen38.qh026_butterfly_replacement import (
    QH026ClassicalCore,
    QH026StatevectorCore,
    QH026TensorCore,
)


def gradient_snapshot(core: torch.nn.Module, encoded: torch.Tensor, weights: torch.Tensor):
    local = encoded.detach().clone().requires_grad_(True)
    output = core(local)
    loss = (output * weights).sum()
    loss.backward()
    return output.detach(), local.grad.detach(), core.theta.grad.detach().clone(), float(loss.detach())


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH026 validation is CUDA-only")
    torch.manual_seed(20260840)
    device = torch.device("cuda:0")
    encoded = torch.randn(4, 20, device=device, dtype=torch.float32)
    weights = torch.randn(4, 10, device=device, dtype=torch.float32)
    qh = QH026StatevectorCore(10, 3, seed=20260841).to(device)
    dq = QH026TensorCore(10, 3, seed=20260841).to(device)
    cc = QH026ClassicalCore(10, 3, seed=20260841).to(device)
    no_ent = QH026StatevectorCore(10, 3, seed=20260841, entangle=False).to(device)
    dq.load_state_dict(qh.state_dict(), strict=True)
    no_ent.load_state_dict(qh.state_dict(), strict=True)

    qh_out, qh_xg, qh_tg, qh_loss = gradient_snapshot(qh, encoded, weights)
    dq_out, dq_xg, dq_tg, dq_loss = gradient_snapshot(dq, encoded, weights)
    cc_out, cc_xg, cc_tg, cc_loss = gradient_snapshot(cc, encoded, weights)
    no_out, no_xg, no_tg, no_loss = gradient_snapshot(no_ent, encoded, weights)

    epsilon = 1.0e-3
    index = (1, 3, 2)
    with torch.no_grad():
        original = float(qh.theta[index])
        qh.theta[index] = original + epsilon
        plus = float((qh(encoded) * weights).sum())
        qh.theta[index] = original - epsilon
        minus = float((qh(encoded) * weights).sum())
        qh.theta[index] = original
    finite_difference = (plus - minus) / (2 * epsilon)
    autograd_value = float(qh_tg[index])

    cpu_rejected = False
    try:
        QH026StatevectorCore(10, 3, seed=1)(torch.zeros(1, 20))
    except RuntimeError:
        cpu_rejected = True

    parameter_counts = {
        "qh": sum(parameter.numel() for parameter in qh.parameters()),
        "dq": sum(parameter.numel() for parameter in dq.parameters()),
        "cc": sum(parameter.numel() for parameter in cc.parameters()),
        "no_ent": sum(parameter.numel() for parameter in no_ent.parameters()),
    }
    payload = {
        "status": "ok",
        "candidate": "QH-026 depth3 reupload butterfly static validation",
        "parameter_counts": parameter_counts,
        "qh_dq_output_max_abs": float((qh_out - dq_out).abs().max()),
        "qh_dq_input_gradient_max_abs": float((qh_xg - dq_xg).abs().max()),
        "qh_dq_theta_gradient_max_abs": float((qh_tg - dq_tg).abs().max()),
        "qh_dq_input_gradient_relative_l2": float((qh_xg - dq_xg).norm() / qh_xg.norm()),
        "qh_dq_theta_gradient_relative_l2": float((qh_tg - dq_tg).norm() / qh_tg.norm()),
        "qh_no_ent_output_max_abs": float((qh_out - no_out).abs().max()),
        "gradient_norms": {
            "qh_input": float(qh_xg.norm()),
            "qh_theta": float(qh_tg.norm()),
            "dq_input": float(dq_xg.norm()),
            "dq_theta": float(dq_tg.norm()),
            "cc_input": float(cc_xg.norm()),
            "cc_theta": float(cc_tg.norm()),
            "no_ent_input": float(no_xg.norm()),
            "no_ent_theta": float(no_tg.norm()),
        },
        "losses": {"qh": qh_loss, "dq": dq_loss, "cc": cc_loss, "no_ent": no_loss},
        "finite_difference": {
            "index": list(index),
            "autograd": autograd_value,
            "central_difference": finite_difference,
            "absolute_error": abs(autograd_value - finite_difference),
        },
        "circuit_calls": {"qh": qh.circuit_calls, "dq": dq.circuit_calls, "no_ent": no_ent.circuit_calls},
        "cpu_quantum_path_rejected": cpu_rejected,
        "all_outputs_finite": all(torch.isfinite(value).all() for value in (qh_out, dq_out, cc_out, no_out)),
        "all_theta_gradient_groups_nonzero": all(float(value.norm()) > 0 for value in (qh_tg, dq_tg, cc_tg, no_tg)),
        "test_rows_inspected": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(payload, indent=2), flush=True)
    if len(set(parameter_counts.values())) != 1:
        raise AssertionError("core parameter budgets differ")
    if payload["qh_dq_output_max_abs"] > 2.0e-5:
        raise AssertionError("DQ output mismatch")
    if payload["qh_dq_input_gradient_max_abs"] > 5.0e-5:
        raise AssertionError("DQ input-gradient mismatch")
    if payload["qh_dq_theta_gradient_max_abs"] > 5.0e-5:
        raise AssertionError("DQ theta-gradient mismatch")
    if payload["finite_difference"]["absolute_error"] > 2.0e-3:
        raise AssertionError("finite-difference gate failed")
    if not cpu_rejected or not payload["all_outputs_finite"] or not payload["all_theta_gradient_groups_nonzero"]:
        raise AssertionError("static safety gate failed")
    output = PROJECT / "artifacts/qh026-butterfly-gpu-validation.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
