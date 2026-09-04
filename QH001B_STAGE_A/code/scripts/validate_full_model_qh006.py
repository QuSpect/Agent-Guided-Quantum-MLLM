#!/usr/bin/env python3
"""Validate QH-006e/QH-006d against real Qwen3.8-27B visual features."""

from __future__ import annotations

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

from quantum_qwen38.amplitude_unitary import freeze_and_inject_qh006


def make_image() -> Path:
    path = ARTIFACT_DIR / "qh006_shapes.png"
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


def trainable_gradients(model: torch.nn.Module) -> dict[str, float]:
    return {
        name: float(parameter.grad.float().norm())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }


def delta_metrics(output: torch.Tensor, base: torch.Tensor) -> dict:
    difference = output.detach().float() - base.float()
    return {
        "relative_delta_rms": float(
            difference.square().mean().sqrt()
            / base.float().square().mean().sqrt().clamp_min(1.0e-12)
        ),
        "changed_fraction": float((output.detach().float() != base.float()).float().mean()),
        "max_abs_delta": float(difference.abs().max()),
    }


def main() -> None:
    torch.manual_seed(20260828)
    model_dir = Path((RECORD_DIR / "active_model_path.txt").read_text().strip())
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
    base_parameter_count = sum(parameter.numel() for parameter in model.parameters())

    text_messages = [
        {"role": "user", "content": [{"type": "text", "text": "只输出两个字：成功"}]}
    ]
    text_inputs = processor.apply_chat_template(
        text_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    with torch.inference_mode():
        base_text_tokens = model.generate(
            **text_inputs,
            max_new_tokens=8,
            do_sample=False,
        )

    image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(make_image())},
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
        base_features = concat_pooler(
            model.model.get_image_features(
                image_inputs["pixel_values"],
                image_inputs["image_grid_thw"],
            )
        ).detach()
    probe = torch.randn_like(base_features)

    post_audit = freeze_and_inject_qh006(model, mode="post")
    post_wrapper = model.model.visual.merger
    post_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-2,
        weight_decay=0.0,
    )
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    started = time.perf_counter()
    post_optimizer.zero_grad(set_to_none=True)
    post_features_before = concat_pooler(
        model.model.get_image_features(
            image_inputs["pixel_values"],
            image_inputs["image_grid_thw"],
        )
    )
    post_loss = (post_features_before.float() * probe.float()).mean()
    post_loss.backward()
    post_gradients = trainable_gradients(model)
    post_optimizer.step()
    torch.cuda.synchronize()
    post_step_seconds = time.perf_counter() - started
    with torch.inference_mode():
        post_features_after = concat_pooler(
            model.model.get_image_features(
                image_inputs["pixel_values"],
                image_inputs["image_grid_thw"],
            )
        )
        post_text_tokens = model.generate(
            **text_inputs,
            max_new_tokens=8,
            do_sample=False,
        )
    post_result = {
        "audit": post_audit,
        "identity_initialization": delta_metrics(post_features_before, base_features),
        "after_one_optimizer_step": delta_metrics(post_features_after, base_features),
        "loss": float(post_loss.detach()),
        "gradient_norms": post_gradients,
        "all_gradients_nonzero": all(value > 0 for value in post_gradients.values()),
        "step_seconds": post_step_seconds,
        "peak_allocated_bytes": [
            torch.cuda.max_memory_allocated(index)
            for index in range(torch.cuda.device_count())
        ],
        "text_only_tokens_exactly_equal": torch.equal(post_text_tokens, base_text_tokens),
    }

    model.model.visual.merger = post_wrapper.frozen_merger
    model.zero_grad(set_to_none=True)
    sandwich_audit = freeze_and_inject_qh006(model, mode="sandwich")
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    started = time.perf_counter()
    sandwich_features = concat_pooler(
        model.model.get_image_features(
            image_inputs["pixel_values"],
            image_inputs["image_grid_thw"],
        )
    )
    sandwich_loss = (sandwich_features.float() * probe.float()).mean()
    sandwich_loss.backward()
    torch.cuda.synchronize()
    sandwich_seconds = time.perf_counter() - started
    sandwich_gradients = trainable_gradients(model)
    with torch.inference_mode():
        sandwich_text_tokens = model.generate(
            **text_inputs,
            max_new_tokens=8,
            do_sample=False,
        )
    sandwich_result = {
        "audit": sandwich_audit,
        "identity_initialization": delta_metrics(sandwich_features, base_features),
        "loss": float(sandwich_loss.detach()),
        "gradient_norms": sandwich_gradients,
        "all_gradients_nonzero": all(value > 0 for value in sandwich_gradients.values()),
        "forward_backward_seconds": sandwich_seconds,
        "peak_allocated_bytes": [
            torch.cuda.max_memory_allocated(index)
            for index in range(torch.cuda.device_count())
        ],
        "text_only_tokens_exactly_equal": torch.equal(
            sandwich_text_tokens,
            base_text_tokens,
        ),
    }

    result = {
        "status": "ok",
        "base_parameter_count": base_parameter_count,
        "vision_feature_shape": list(base_features.shape),
        "post_qh006e": post_result,
        "sandwich_qh006d": sandwich_result,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = RECORD_DIR / "full_model_qh006_validation.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
