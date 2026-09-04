#!/usr/bin/env python3
"""Full Qwen3.8-27B smoke test for the QH-024 true replacement."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code/src"))

from quantum_qwen38.hyqut_replacement import QH024Config, freeze_and_replace_qh024


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH-024 full-model smoke is GPU-only")
    torch.manual_seed(20260838)
    model_dir = Path((PROJECT / "records/active_model_path.txt").read_text().strip())
    tokens = torch.load(
        PROJECT / "datasets/processed/wikitext2_qwen38/train-tokens.pt",
        map_location="cpu",
        weights_only=True,
    )[:64]
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    config = QH024Config()
    injection = freeze_and_replace_qh024(model, config, core_kind="quantum")
    replacement = model.model.language_model.layers[config.layer_index].self_attn.v_proj
    expected_trainable = (
        config.hidden_size * (2 * config.n_qubits)
        + config.n_qubits * config.output_size
        + 3 * config.n_qubits * config.depth
        + 1
    )
    expected_replacement = (
        config.hidden_size * config.svd_rank
        + config.svd_rank * config.output_size
        + expected_trainable
    )
    expected_removed = config.hidden_size * config.output_size - expected_replacement
    if injection["trainable_parameter_count"] != expected_trainable:
        raise AssertionError("full-model trainable parameter count mismatch")
    if injection["net_model_parameter_reduction"] != expected_removed:
        raise AssertionError("full-model deployed parameter reduction mismatch")

    input_ids = tokens.to(model.device, dtype=torch.long).unsqueeze(0)
    model.eval()
    replacement.set_capture_teacher_only(True)
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    hidden_states, teacher_target = replacement.take_capture()
    replacement.set_capture_teacher_only(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in replacement.parameters() if parameter.requires_grad],
        lr=1.0e-3,
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    student_before = replacement.student_forward(hidden_states)
    denominator = teacher_target.float().square().mean().clamp_min(1.0e-8)
    relative_mse_before = (student_before.float() - teacher_target.float()).square().mean() / denominator
    relative_mse_before.backward()
    gradient_groups = {
        "down": float(replacement.down.weight.grad.abs().max()),
        "up": float(replacement.up.weight.grad.abs().max()),
        "theta": float(replacement.core.theta.grad.abs().max()),
        "gamma": float(replacement.gamma.grad.abs()),
    }
    if any(value <= 0 for value in gradient_groups.values()):
        raise AssertionError(f"missing full-model gradient: {gradient_groups}")
    optimizer.step()
    with torch.inference_mode():
        student_after = replacement.student_forward(hidden_states)
        relative_mse_after = (
            (student_after.float() - teacher_target.float()).square().mean() / denominator
        )
    replacement.strip_teacher()
    if replacement.teacher_weight is not None:
        raise AssertionError("teacher was not stripped before deployed forward")
    with torch.inference_mode():
        output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    if not torch.isfinite(output.loss):
        raise AssertionError("full-model deployed loss is nonfinite")
    frozen_grad_tensors = sum(
        parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad
    )
    payload = {
        "status": "ok",
        "model_dir": str(model_dir),
        "injection": injection,
        "expected_trainable_parameters": expected_trainable,
        "expected_replacement_parameters": expected_replacement,
        "expected_net_model_parameter_reduction": expected_removed,
        "teacher_capture_shape": list(hidden_states.shape),
        "teacher_target_shape": list(teacher_target.shape),
        "one_step_relative_projection_mse_before": float(relative_mse_before.detach()),
        "one_step_relative_projection_mse_after": float(relative_mse_after),
        "gradient_groups": gradient_groups,
        "deployed_forward_mean_token_nll": float(output.loss),
        "simulator_calls": replacement.core.circuit_calls,
        "teacher_buffer_stripped": replacement.teacher_weight is None,
        "frozen_parameter_grad_tensor_count": frozen_grad_tensors,
        "gpu_peak_allocated_bytes": [
            torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())
        ],
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "claim_limit": "one-sample engineering smoke; not a task-performance result",
    }
    path = PROJECT / "artifacts/qh024-full-model-smoke.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
