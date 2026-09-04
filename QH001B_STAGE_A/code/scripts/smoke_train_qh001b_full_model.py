#!/usr/bin/env python3
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForMultimodalLM, AutoProcessor


PROJECT_DIR = Path(__file__).resolve().parents[2]
RECORD_DIR = PROJECT_DIR / "records"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
sys.path.insert(0, str(PROJECT_DIR / "code/src"))

from quantum_qwen38.quantum_residual_bf16 import freeze_and_inject_qh001b


def make_image() -> Path:
    path = ARTIFACT_DIR / "smoke_train_shapes.png"
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((24, 64, 112, 152), fill="red")
    draw.rectangle((148, 64, 232, 148), fill="blue")
    image.save(path)
    return path


def main() -> None:
    torch.manual_seed(20260828)
    model_dir = Path((RECORD_DIR / "active_model_path.txt").read_text(encoding="utf-8").strip())
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    audit = freeze_and_inject_qh001b(model)
    adapter = model.model.visual.merger.adapter
    model.train()

    image_path = make_image()
    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "path": str(image_path)},
            {"type": "text", "text": "图中左侧的圆形是什么颜色？只回答颜色。"},
        ],
    }
    prompt_inputs = processor.apply_chat_template(
        [user_message],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    full_inputs = processor.apply_chat_template(
        [user_message, {"role": "assistant", "content": [{"type": "text", "text": "红色"}]}],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    labels = full_inputs["input_ids"].clone()
    prompt_length = int(prompt_inputs["input_ids"].shape[-1])
    labels[:, :prompt_length] = -100
    full_inputs["labels"] = labels

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=3.0e-4, weight_decay=0.0)
    before_theta = adapter.vqc.theta.detach().clone()
    before_up = adapter.up.weight.detach().clone()
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**full_inputs)
    loss = outputs.loss
    loss.backward()
    grad_result = {
        "down_grad_norm": float(adapter.down.weight.grad.float().norm()),
        "vqc_grad_norm": float(adapter.vqc.theta.grad.float().norm()),
        "up_grad_norm": float(adapter.up.weight.grad.float().norm()),
        "raw_scale_grad": float(adapter.raw_scale.grad.float()),
    }
    optimizer.step()
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started

    frozen_grad_tensors = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    result = {
        "status": "ok",
        "candidate": "QH-001b",
        "audit": audit,
        "loss": float(loss.detach()),
        "prompt_tokens": prompt_length,
        "total_tokens": int(full_inputs["input_ids"].shape[-1]),
        "supervised_tokens": int((labels != -100).sum()),
        "step_seconds": elapsed_seconds,
        "gpu_peak_allocated_bytes": [
            torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())
        ],
        "gradients": grad_result,
        "theta_update_norm": float((adapter.vqc.theta.detach() - before_theta).float().norm()),
        "up_update_norm": float((adapter.up.weight.detach() - before_up).float().norm()),
        "frozen_parameter_grad_tensor_count": len(frozen_grad_tensors),
        "frozen_parameter_grad_examples": frozen_grad_tensors[:10],
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = RECORD_DIR / "qh001b_full_model_smoke_train.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
