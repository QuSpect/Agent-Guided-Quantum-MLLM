#!/usr/bin/env python3
"""Train-only-data integration smoke for QH-023 in the full 27B model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code/src"))

from quantum_qwen38.pauli_stiefel_adapter import freeze_and_inject_qh023


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH-023 integration smoke is CUDA-only")
    torch.manual_seed(20260835)
    model_dir = Path((PROJECT / "records/active_model_path.txt").read_text().strip())
    tokens = torch.load(
        PROJECT / "datasets/processed/wikitext2_qwen38/train-tokens.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    audit = freeze_and_inject_qh023(model)
    adapter = model.model.language_model.layers[7].self_attn.v_proj.adapter
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(trainable, lr=1.0e-4, weight_decay=0.0)
    losses: list[float] = []
    gradient_norms: list[float] = []
    calls_before = adapter.circuit_calls
    model.train()
    for step in range(4):
        start = step * 64
        input_ids = tokens[start : start + 64].to(device=model.device, dtype=torch.long).unsqueeze(0)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        gradient_norms.append(float(torch.nn.utils.clip_grad_norm_(trainable, 1.0)))
        optimizer.step()
        losses.append(float(loss.detach()))
    training_calls = adapter.circuit_calls - calls_before
    model.eval()
    calls_before_eval = adapter.circuit_calls
    with torch.inference_mode():
        for step in range(8):
            start = (step + 8) * 64
            input_ids = tokens[start : start + 64].to(device=model.device, dtype=torch.long).unsqueeze(0)
            output = model(input_ids=input_ids, use_cache=False).logits
            if not torch.isfinite(output).all():
                raise RuntimeError("non-finite full-model output")
    evaluation_calls = adapter.circuit_calls - calls_before_eval
    result = {
        "status": "pass",
        "candidate": "QH-023-static-full-model-smoke",
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "audit": audit,
        "steps": 4,
        "train_sequences_only": True,
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "losses_for_health_check_only": losses,
        "gradient_norms": gradient_norms,
        "all_gradient_norms_finite_nonzero": all(
            value > 0 and torch.isfinite(torch.tensor(value)) for value in gradient_norms
        ),
        "frozen_parameter_grad_tensor_count": sum(
            parameter.grad is not None
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "training_simulator_calls": training_calls,
        "eight_forward_simulator_calls": evaluation_calls,
        "orthogonality_max_error": max(adapter.last_orthogonality_error.values()),
        "task_effect_claimed": False,
    }
    output_path = PROJECT / "artifacts/qh023-full-model-smoke.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if (
        audit["trainable_parameter_count"] != 209
        or result["frozen_parameter_grad_tensor_count"] != 0
        or not result["all_gradient_norms_finite_nonzero"]
        or result["eight_forward_simulator_calls"] != 32
        or result["orthogonality_max_error"] >= 1.0e-5
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
