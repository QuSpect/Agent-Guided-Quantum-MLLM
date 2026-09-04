#!/usr/bin/env python3
"""Two-step Qwen3.8-27B smoke for QH037 quantum or classical arm."""
from __future__ import annotations
import argparse,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForMultimodalLM
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh037_grouped_qvaf_ffn_replacement import QH037Config,install_qh037,unique_trainable_parameters
def main():
 p=argparse.ArgumentParser();p.add_argument("--branch",choices=["quantum","classical"],required=True);a=p.parse_args();torch.manual_seed(20260864);torch.cuda.manual_seed_all(20260864)
 model_dir=Path((PROJECT/"records/active_model_path.txt").read_text().strip());tokens=torch.load(PROJECT/"datasets/processed/wikitext2_qwen38/train-tokens.pt",map_location="cpu",weights_only=True)
 model=AutoModelForMultimodalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,device_map="balanced",low_cpu_mem_usage=True,attn_implementation="sdpa");model.eval();model.config.use_cache=False;cfg=QH037Config();replacement,original,audit=install_qh037(model,cfg,branch=a.branch);params=list(unique_trainable_parameters(replacement));opt=torch.optim.AdamW(params,lr=1e-3);rows=[];before=int(replacement.core.circuit_calls.item())
 for step in range(2):
  x=tokens[(7000+step)*64:(7001+step)*64].to(model.device).unsqueeze(0);model.model.language_model.layers[cfg.target_layer].mlp=original
  with torch.inference_mode():teacher=model(input_ids=x,use_cache=False).logits[:,:-1].detach()
  model.model.language_model.layers[cfg.target_layer].mlp=replacement;opt.zero_grad(set_to_none=True);started=time.perf_counter();student=model(input_ids=x,use_cache=False).logits[:,:-1];loss=F.kl_div(F.log_softmax(student.float(),-1),F.softmax(teacher.float(),-1),reduction="sum")/(student.shape[0]*student.shape[1]);loss.backward();theta_nonzero=int((replacement.core.theta.grad.abs()>0).sum());frozen=sum(int(p.grad is not None) for p in model.parameters() if not p.requires_grad);torch.nn.utils.clip_grad_norm_(params,1.0);opt.step();torch.cuda.synchronize();rows.append({"step":step,"kl":float(loss),"theta_nonzero_grad":theta_nonzero,"frozen_gradient_tensor_count":frozen,"seconds":time.perf_counter()-started})
 calls=int(replacement.core.circuit_calls.item())-before;expected=2 if a.branch=="quantum" else 0
 if calls!=expected or any(r["frozen_gradient_tensor_count"] for r in rows) or rows[-1]["theta_nonzero_grad"]!=replacement.core.theta.numel():raise AssertionError({"calls":calls,"rows":rows})
 payload={"status":"pass","branch":a.branch,"audit":audit,"rows":rows,"simulator_calls":calls,"peak_vram_bytes":[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())],"created_at":datetime.now(timezone.utc).isoformat()};out=PROJECT/f"artifacts/qh037-full-model-smoke-{a.branch}.json";out.write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
