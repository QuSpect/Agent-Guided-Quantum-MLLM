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
from quantum_qwen38.segmented_anchor_stable import freeze_and_inject_qh003b


def gpu_memory() -> list[dict]:
    values = []
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        values.append(
            {
                "index": index,
                "allocated_bytes": torch.cuda.memory_allocated(index),
                "reserved_bytes": torch.cuda.memory_reserved(index),
                "free_bytes": free_bytes,
                "total_bytes": total_bytes,
            }
        )
    return values


def make_test_image() -> Path:
    path = ARTIFACT_DIR / "synthetic_shapes.png"
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((24, 64, 112, 152), fill="red")
    draw.rectangle((148, 64, 232, 148), fill="blue")
    draw.text((50, 174), "A", fill="black")
    draw.text((184, 174), "B", fill="black")
    image.save(path)
    return path


def concat_pooler(output) -> torch.Tensor:
    pooler = output.pooler_output
    return torch.cat(pooler, dim=0) if isinstance(pooler, (list, tuple)) else pooler


def grad_norm(parameter: torch.Tensor | None) -> float | None:
    return None if parameter is None or parameter.grad is None else float(parameter.grad.float().norm())


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = Path((RECORD_DIR / "active_model_path.txt").read_text(encoding="utf-8").strip())
    revision = (RECORD_DIR / "qwen38_model_revision.txt").read_text(encoding="utf-8").strip()
    memory_before = gpu_memory()

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started

    text_messages = [{"role": "user", "content": [{"type": "text", "text": "只输出两个字：成功"}]}]
    text_inputs = processor.apply_chat_template(
        text_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    generate_started = time.perf_counter()
    with torch.inference_mode():
        base_output = model.generate(**text_inputs, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    base_generate_seconds = time.perf_counter() - generate_started
    base_new_tokens = base_output[0][text_inputs["input_ids"].shape[-1] :]
    base_text = processor.decode(base_new_tokens, skip_special_tokens=True)

    image_path = make_test_image()
    image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path)},
                {"type": "text", "text": "图中左侧是什么颜色的圆形？只回答颜色。"},
            ],
        }
    ]
    image_inputs = processor.apply_chat_template(
        image_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    with torch.inference_mode():
        base_vision_output = model.model.get_image_features(
            image_inputs["pixel_values"], image_inputs["image_grid_thw"]
        )
        base_features = concat_pooler(base_vision_output).detach().float()
    # Capture this before replacing any module so the record never includes
    # adapter parameters in the frozen checkpoint total.
    base_parameter_count = sum(parameter.numel() for parameter in model.parameters())

    qh001_audit = freeze_and_inject_qh001b(model)
    qh001_wrapper = model.model.visual.merger
    qh001_adapter = qh001_wrapper.adapter
    qh001_started = time.perf_counter()
    qh001_output = model.model.get_image_features(
        image_inputs["pixel_values"], image_inputs["image_grid_thw"]
    )
    qh001_features = concat_pooler(qh001_output)
    qh001_loss = qh001_features.float().square().mean()
    qh001_loss.backward()
    torch.cuda.synchronize()
    qh001_seconds = time.perf_counter() - qh001_started
    qh001_delta = qh001_features.detach().float() - base_features
    with torch.inference_mode():
        qh001_text_output = model.generate(**text_inputs, max_new_tokens=8, do_sample=False)
    qh001_text_equal = torch.equal(qh001_text_output, base_output)
    qh001_result = {
        "audit": qh001_audit,
        "vision_tokens": int(qh001_features.shape[0]),
        "feature_shape": list(qh001_features.shape),
        "feature_forward_backward_seconds": qh001_seconds,
        "relative_delta_rms": float(
            qh001_delta.square().mean().sqrt() / base_features.square().mean().sqrt()
        ),
        "changed_fraction": float((qh001_features.detach().float() != base_features).float().mean()),
        "down_grad_norm": grad_norm(qh001_adapter.down.weight),
        "vqc_grad_norm": grad_norm(qh001_adapter.vqc.theta),
        "up_grad_norm": grad_norm(qh001_adapter.up.weight),
        "text_only_tokens_exactly_equal": qh001_text_equal,
    }
    model.model.visual.merger = qh001_wrapper.frozen_merger
    model.zero_grad(set_to_none=True)

    qh003_audit = freeze_and_inject_qh003b(model)
    qh003_wrapper = model.model.visual
    qh003_adapter = qh003_wrapper.adapter
    qh003_started = time.perf_counter()
    qh003_output = model.model.get_image_features(
        image_inputs["pixel_values"], image_inputs["image_grid_thw"]
    )
    qh003_features = concat_pooler(qh003_output)
    qh003_loss = qh003_features.float().square().mean()
    qh003_loss.backward()
    torch.cuda.synchronize()
    qh003_seconds = time.perf_counter() - qh003_started
    qh003_delta = qh003_features.detach().float() - base_features
    with torch.inference_mode():
        qh003_text_output = model.generate(**text_inputs, max_new_tokens=8, do_sample=False)
    qh003_result = {
        "audit": qh003_audit,
        "vision_tokens": int(qh003_features.shape[0]),
        "feature_shape": list(qh003_features.shape),
        "anchors": qh003_adapter.last_num_anchors,
        "feature_forward_backward_seconds": qh003_seconds,
        "relative_delta_rms": float(
            qh003_delta.square().mean().sqrt() / base_features.square().mean().sqrt()
        ),
        "changed_fraction": float((qh003_features.detach().float() != base_features).float().mean()),
        "down_grad_norm": grad_norm(qh003_adapter.down.weight),
        "vqc_grad_norm": grad_norm(qh003_adapter.vqc.theta),
        "up_grad_norm": grad_norm(qh003_adapter.up.weight),
        "text_only_tokens_exactly_equal": torch.equal(qh003_text_output, base_output),
    }

    device_map = {
        str(key): str(value) for key, value in (getattr(model, "hf_device_map", {}) or {}).items()
    }
    result = {
        "status": "ok",
        "hf_revision": revision,
        "model_dir": str(model_dir),
        "model_class": model.__class__.__name__,
        "processor_class": processor.__class__.__name__,
        "load_seconds": load_seconds,
        "base_text_generate_seconds": base_generate_seconds,
        "base_text_output": base_text,
        "base_parameter_count": base_parameter_count,
        "device_map": device_map,
        "gpu_memory_before": memory_before,
        "gpu_memory_after": gpu_memory(),
        "qh001b": qh001_result,
        "qh003b": qh003_result,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = RECORD_DIR / "full_model_injection_validation.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
