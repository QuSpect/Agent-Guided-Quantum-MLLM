#!/usr/bin/env python3
"""Two-step train-only QH034 integration smoke on frozen Qwen3.8-27B."""

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
SOURCE_LOCK = PROJECT / "artifacts/qh031-shared-basis-lock.json"
OUTPUT = PROJECT / "artifacts/qh034-full-model-smoke.json"
sys.path.insert(0, str(PROJECT / "code/src"))

from quantum_qwen38.qh034_spectral_modulation_replacement import (  # noqa: E402
    QH034Config,
    install_qh034,
    unique_trainable_parameters,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_projections(model, modules) -> None:
    for index, module in modules.items():
        model.model.language_model.layers[index].self_attn.v_proj = module


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QH034 full-model smoke is GPU-only")
    torch.manual_seed(20260857)
    torch.cuda.manual_seed_all(20260857)
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    tokens = torch.load(TOKENS, map_location="cpu", weights_only=True)
    length = int(lock["sequence_length"])
    indices = [int(value) for value in lock["train_indices"][:2]]
    model_dir = Path((PROJECT / "records/active_model_path.txt").read_text().strip())
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16,
        device_map="balanced", low_cpu_mem_usage=True, attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = False
    config = QH034Config()
    replacements, originals, audit = install_qh034(model, config, branch="quantum")
    trainable = list(unique_trainable_parameters(replacements))
    optimizer = torch.optim.AdamW(trainable, lr=1.0e-3, weight_decay=0.0)
    backbone = model.model.language_model
    core = replacements[config.layer_indices[0]].core
    calls_before = int(core.circuit_calls.item())
    rows = []
    for step, index in enumerate(indices, start=1):
        start = index * length
        input_ids = tokens[start : start + length].to(
            model.device, dtype=torch.long
        ).unsqueeze(0)
        set_projections(model, originals)
        with torch.inference_mode():
            teacher = backbone(
                input_ids=input_ids, use_cache=False, return_dict=True
            ).last_hidden_state.detach()
        set_projections(model, replacements)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        student = backbone(
            input_ids=input_ids, use_cache=False, return_dict=True
        ).last_hidden_state
        denominator = teacher.float().square().mean().clamp_min(1.0e-8)
        loss = (student.float() - teacher.float()).square().mean() / denominator
        loss.backward()
        theta_grad = core.theta.grad.detach().float()
        gamma_grad = torch.stack(
            [module.gamma.grad.detach().float().abs() for module in replacements.values()]
        )
        total_grad = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        rows.append({
            "step": step,
            "train_block_index": index,
            "relative_final_hidden_mse": float(loss.detach()),
            "gradient_norm_before_clip": float(total_grad.detach()),
            "theta_gradient_norm": float(theta_grad.norm()),
            "theta_nonzero_gradient_elements": int((theta_grad != 0).sum()),
            "gamma_gradient_norm": float(gamma_grad.norm()),
            "step_seconds": time.perf_counter() - started,
        })
    calls = int(core.circuit_calls.item()) - calls_before
    expected = len(indices) * len(config.layer_indices)
    if calls != expected:
        raise AssertionError(f"simulator calls {calls} != {expected}")
    if any(row["theta_gradient_norm"] <= 0 for row in rows):
        raise AssertionError("QH034 theta did not receive a gradient")
    frozen_grad_tensors = sum(
        int(parameter.grad is not None)
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    if frozen_grad_tensors:
        raise AssertionError("a frozen Qwen/shared-basis parameter received gradients")
    payload = {
        "status": "pass",
        "mode": "two-step already-consumed-train-only full-model integration smoke",
        "candidate": "QH-034 quantum shared-SVD spectral modulation",
        "model_dir": str(model_dir),
        "source_lock": str(SOURCE_LOCK),
        "source_lock_sha256": sha256(SOURCE_LOCK),
        "train_tokens_sha256": sha256(TOKENS),
        "sequence_length": length,
        "train_block_indices": indices,
        "holdout_rows_inspected": 0,
        "test_rows_inspected": 0,
        "parameter_audit": audit,
        "steps": rows,
        "final_theta_l2_norm": float(core.theta.detach().norm()),
        "final_gammas": {
            str(index): float(module.gamma.detach())
            for index, module in replacements.items()
        },
        "unique_trainable_tensor_count": len(trainable),
        "unique_trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable
        ),
        "frozen_gradient_tensor_count": frozen_grad_tensors,
        "simulator_calls": calls,
        "gpu_peak_allocated_bytes": [
            torch.cuda.max_memory_allocated(i)
            for i in range(torch.cuda.device_count())
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
