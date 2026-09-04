#!/usr/bin/env python3
"""One-step, train-only QH031 integration smoke on the frozen Qwen3.8-27B."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM


PROJECT = Path(__file__).resolve().parents[2]
TOKENS = PROJECT / "datasets/processed/wikitext2_qwen38/train-tokens.pt"
SOURCE_LOCK = PROJECT / "artifacts/qh026-butterfly-lock.json"
OUTPUT = PROJECT / "artifacts/qh031-full-model-smoke.json"
sys.path.insert(0, str(PROJECT / "code/src"))

from quantum_qwen38.qh031_shared_basis_replacement import (  # noqa: E402
    QH031Config,
    install_qh031,
    unique_trainable_parameters,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_projections(model, modules) -> None:
    for layer_index, module in modules.items():
        model.model.language_model.layers[layer_index].self_attn.v_proj = module


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH031 full-model smoke is GPU-only")
    torch.manual_seed(20260842)
    torch.cuda.manual_seed_all(20260842)
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    tokens = torch.load(TOKENS, map_location="cpu", weights_only=True)
    length = int(lock["sequence_length"])
    block_index = int(lock["train_indices"][0])
    start = block_index * length
    model_dir = Path((PROJECT / "records/active_model_path.txt").read_text().strip())
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = False
    config = QH031Config()
    replacements, originals, audit = install_qh031(model, config, branch="quantum")
    trainable = list(unique_trainable_parameters(replacements))
    input_ids = tokens[start : start + length].to(model.device).unsqueeze(0)
    backbone = model.model.language_model

    set_projections(model, originals)
    with torch.inference_mode():
        teacher_hidden = backbone(
            input_ids=input_ids, use_cache=False, return_dict=True
        ).last_hidden_state.detach()
    set_projections(model, replacements)
    shared_core = replacements[config.layer_indices[0]].core
    calls_before = int(shared_core.circuit_calls.item())
    optimizer = torch.optim.AdamW(trainable, lr=3.0e-3, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    student_hidden = backbone(
        input_ids=input_ids, use_cache=False, return_dict=True
    ).last_hidden_state
    denominator = teacher_hidden.float().square().mean().clamp_min(1.0e-8)
    relative_mse = (
        (student_hidden.float() - teacher_hidden.float()).square().mean() / denominator
    )
    relative_mse.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    calls_after = int(shared_core.circuit_calls.item())
    if calls_after - calls_before != len(config.layer_indices):
        raise AssertionError(
            f"expected two exact-simulator calls, observed {calls_after - calls_before}"
        )
    frozen_grad_tensors = sum(
        int(parameter.grad is not None)
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    if frozen_grad_tensors:
        raise AssertionError("a frozen Qwen/shared-SVD parameter received gradients")
    theta_grad_norm = float(shared_core.theta.grad.float().norm())
    if not theta_grad_norm > 0.0:
        raise AssertionError("shared quantum theta received no gradient")

    payload = {
        "status": "pass",
        "mode": "one-step train-only full-model integration smoke",
        "candidate": "QH-031 shared-basis two-layer quantum replacement",
        "model_dir": str(model_dir),
        "source_lock": str(SOURCE_LOCK),
        "source_lock_sha256": sha256(SOURCE_LOCK),
        "train_tokens_sha256": sha256(TOKENS),
        "sequence_length": length,
        "train_block_index": block_index,
        "holdout_rows_inspected": 0,
        "test_rows_inspected": 0,
        "parameter_audit": audit,
        "teacher_hidden_shape": list(teacher_hidden.shape),
        "student_hidden_shape": list(student_hidden.shape),
        "relative_final_hidden_mse_before_step": float(relative_mse.detach()),
        "gradient_norm_before_clip": float(gradient_norm.detach()),
        "theta_gradient_norm": theta_grad_norm,
        "unique_trainable_tensor_count": len(trainable),
        "unique_trainable_parameter_count": sum(p.numel() for p in trainable),
        "frozen_gradient_tensor_count": frozen_grad_tensors,
        "simulator_calls": calls_after - calls_before,
        "step_seconds": elapsed,
        "gpu_peak_allocated_bytes": [
            torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

