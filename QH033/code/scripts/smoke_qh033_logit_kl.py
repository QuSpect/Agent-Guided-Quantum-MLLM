#!/usr/bin/env python3
"""One-step output-distribution KL smoke from the QH032 screen checkpoint."""

from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForMultimodalLM

PROJECT = Path(__file__).resolve().parents[2]
TOKENS = PROJECT / "datasets/processed/wikitext2_qwen38/train-tokens.pt"
LOCK = PROJECT / "artifacts/qh032-train-screen-lock.json"
CHECKPOINT = PROJECT / "artifacts/qh032_train_screen/qh032-screen-checkpoint.pt"
OUTPUT = PROJECT / "artifacts/qh033-logit-kl-smoke.json"
sys.path.insert(0, str(PROJECT / "code/src"))
from quantum_qwen38.qh032_pauli_observable_replacement import QH032Config, install_qh032, unique_trainable_parameters

def set_projections(model, modules):
    for index, module in modules.items(): model.model.language_model.layers[index].self_attn.v_proj = module

def load_checkpoint(replacements):
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    first = sorted(replacements)[0]
    with torch.no_grad():
        replacements[first].core.theta.copy_(state["core.theta"].to(replacements[first].core.theta.device))
        replacements[first].core.alpha.copy_(state["core.alpha"].to(replacements[first].core.alpha.device))
        for index, module in replacements.items(): module.gamma.copy_(state[f"gamma.{index}"].to(module.gamma.device))

def main():
    if not torch.cuda.is_available(): raise RuntimeError("QH033 KL smoke is GPU-only")
    torch.manual_seed(20260852); torch.cuda.manual_seed_all(20260852)
    lock = json.loads(LOCK.read_text())
    tokens = torch.load(TOKENS, map_location="cpu", weights_only=True)
    length = lock["sequence_length"]; index = lock["train_indices"][0]
    inputs = tokens[index * length : (index + 1) * length].to("cuda:0", dtype=torch.long).unsqueeze(0)
    model_dir = Path((PROJECT / "records/active_model_path.txt").read_text().strip())
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16, device_map="balanced", low_cpu_mem_usage=True, attn_implementation="sdpa")
    model.eval(); model.config.use_cache = False
    config = QH032Config(); replacements, originals, audit = install_qh032(model, config, branch="quantum")
    load_checkpoint(replacements)
    parameters = list(unique_trainable_parameters(replacements))
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-3, weight_decay=0.0)
    set_projections(model, originals)
    with torch.inference_mode(): teacher_logits = model(input_ids=inputs, use_cache=False).logits[:, :-1].detach()
    set_projections(model, replacements)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    student_logits = model(input_ids=inputs, use_cache=False).logits[:, :-1]
    token_count = student_logits.shape[0] * student_logits.shape[1]
    loss = F.kl_div(
        F.log_softmax(student_logits.float(), dim=-1),
        F.softmax(teacher_logits.float(), dim=-1),
        reduction="sum",
    ) / token_count
    loss.backward()
    core = replacements[config.layer_indices[0]].core
    grad = {
        "theta": float(core.theta.grad.norm()), "alpha": float(core.alpha.grad.norm()),
        "gamma": float(torch.stack([m.gamma.grad.abs() for m in replacements.values()]).norm()),
    }
    total_grad = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
    optimizer.step(); torch.cuda.synchronize()
    frozen_grads = sum(int(p.grad is not None) for p in model.parameters() if not p.requires_grad)
    if not all(torch.isfinite(torch.tensor(v)) and v > 0 for v in grad.values()) or frozen_grads:
        raise AssertionError(f"invalid KL gradient contract: {grad}, frozen={frozen_grads}")
    payload = {
        "status": "pass", "candidate": "QH-033 output-distribution KL stage",
        "source": "QH032 train-only checkpoint", "train_block_index": index,
        "sequence_length": length, "teacher_logits_shape": list(teacher_logits.shape),
        "student_logits_shape": list(student_logits.shape), "mean_token_teacher_to_student_kl": float(loss.detach()),
        "gradient_norms": grad, "total_gradient_norm": total_grad, "step_seconds": time.perf_counter() - started,
        "frozen_gradient_tensor_count": frozen_grads, "parameter_audit": audit,
        "validation_rows_inspected": 0, "test_rows_inspected": 0,
        "gpu_peak_allocated_bytes": [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))

if __name__ == "__main__": main()

