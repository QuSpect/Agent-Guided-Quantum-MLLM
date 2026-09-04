from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

parser=argparse.ArgumentParser(description="Base-model sanity inference; scheme evaluation uses run_reached_stage.py")
parser.add_argument("--model", type=Path, default=Path("./models/Qwen3.8-27B"))
parser.add_argument("--prompt", default="用一句话解释量子—经典混合模型。")
parser.add_argument("--max-new-tokens", type=int, default=96)
args=parser.parse_args()
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required")
processor=AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
model=AutoModelForMultimodalLM.from_pretrained(
    args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True
).eval()
messages=[{"role":"user","content":[{"type":"text","text":args.prompt}]}]
inputs=processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to(model.device)
with torch.inference_mode():
    output=model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
new_tokens=output[:, inputs["input_ids"].shape[1]:]
print(processor.batch_decode(new_tokens, skip_special_tokens=True)[0])
