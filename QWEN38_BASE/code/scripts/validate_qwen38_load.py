#!/usr/bin/env python3
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "models" / "Qwen3.8-27B"
RECORD_DIR = PROJECT_DIR / "records"


def gpu_memory() -> list[dict]:
    result = []
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        result.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "allocated_bytes": torch.cuda.memory_allocated(index),
                "reserved_bytes": torch.cuda.memory_reserved(index),
                "free_bytes": free_bytes,
                "total_bytes": total_bytes,
            }
        )
    return result


def main() -> None:
    revision = (RECORD_DIR / "qwen38_model_revision.txt").read_text(encoding="utf-8").strip()
    before = gpu_memory()
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    load_seconds = time.perf_counter() - load_started
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "只输出两个字：成功"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    text = processor.decode(generated, skip_special_tokens=True)

    device_map = getattr(model, "hf_device_map", None)
    result = {
        "status": "ok",
        "revision": revision,
        "model_class": model.__class__.__name__,
        "processor_class": processor.__class__.__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "output_tokens": int(generated.shape[-1]),
        "decoded_output": text,
        "device_map": device_map,
        "gpu_memory_before": before,
        "gpu_memory_after": gpu_memory(),
        "gpu_peak_allocated_bytes": [
            torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())
        ],
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    (RECORD_DIR / "qwen38_load_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
