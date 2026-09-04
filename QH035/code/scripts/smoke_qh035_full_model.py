#!/usr/bin/env python3
"""Two-step already-consumed-data 27B integration smoke for QH035."""
from __future__ import annotations
import hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import torch
from transformers import AutoModelForMultimodalLM
PROJECT=Path(__file__).resolve().parents[2]; TOKENS=PROJECT/"datasets/processed/wikitext2_qwen38/train-tokens.pt"
LOCK=PROJECT/"artifacts/qh031-shared-basis-lock.json"; OUTPUT=PROJECT/"artifacts/qh035-full-model-smoke.json"
sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh035_shared_rmsnorm_replacement import QH035Config,install_qh035,unique_trainable_parameters
def sha(path):
 d=hashlib.sha256()
 with path.open("rb") as h:
  for c in iter(lambda:h.read(8<<20),b""):d.update(c)
 return d.hexdigest()
def set_norms(model,modules,cfg):
 for i,m in modules.items(): setattr(model.model.language_model.layers[i],cfg.norm_name,m)
def main():
 if not torch.cuda.is_available(): raise RuntimeError("QH035 full-model smoke is GPU-only")
 torch.manual_seed(20260861); torch.cuda.manual_seed_all(20260861); lock=json.loads(LOCK.read_text()); tokens=torch.load(TOKENS,map_location="cpu",weights_only=True)
 n=int(lock["sequence_length"]); indices=[int(i) for i in lock["train_indices"][:2]]; model_dir=Path((PROJECT/"records/active_model_path.txt").read_text().strip())
 model=AutoModelForMultimodalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,device_map="balanced",low_cpu_mem_usage=True,attn_implementation="sdpa")
 model.eval(); model.config.use_cache=False; cfg=QH035Config(); replacements,originals,audit=install_qh035(model,cfg,branch="quantum")
 params=list(unique_trainable_parameters(replacements)); opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=0.0); backbone=model.model.language_model
 core=replacements[cfg.layer_indices[0]].core; before=int(core.circuit_calls.item()); rows=[]
 for step,index in enumerate(indices,1):
  x=tokens[index*n:(index+1)*n].to(model.device,dtype=torch.long).unsqueeze(0); set_norms(model,originals,cfg)
  with torch.inference_mode(): teacher=backbone(input_ids=x,use_cache=False,return_dict=True).last_hidden_state.detach()
  set_norms(model,replacements,cfg); opt.zero_grad(set_to_none=True); started=time.perf_counter(); student=backbone(input_ids=x,use_cache=False,return_dict=True).last_hidden_state
  loss=(student.float()-teacher.float()).square().mean()/teacher.float().square().mean().clamp_min(1e-8); loss.backward(); theta=core.theta.grad.detach().float()
  gamma=torch.stack([m.gamma.grad.detach().float().abs() for m in replacements.values()]); total=torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); torch.cuda.synchronize()
  rows.append({"step":step,"train_block_index":index,"relative_final_hidden_mse":float(loss.detach()),"gradient_norm_before_clip":float(total),
    "theta_gradient_norm":float(theta.norm()),"theta_nonzero_gradient_elements":int((theta!=0).sum()),"gamma_gradient_norm":float(gamma.norm()),
    "step_seconds":time.perf_counter()-started})
 calls=int(core.circuit_calls.item())-before; expected=len(indices)*len(cfg.layer_indices)
 if calls!=expected or any(r["theta_gradient_norm"]<=0 for r in rows): raise AssertionError(f"QH035 call/gradient contract failed {calls}/{expected}")
 frozen=sum(int(p.grad is not None) for p in model.parameters() if not p.requires_grad)
 if frozen: raise AssertionError("frozen gradient")
 payload={"status":"pass","mode":"two-step already-consumed-train-only full-model integration smoke","candidate":"QH-035 shared RMSNorm quantum residual",
  "model_dir":str(model_dir),"source_lock":str(LOCK),"source_lock_sha256":sha(LOCK),"train_tokens_sha256":sha(TOKENS),"sequence_length":n,
  "train_block_indices":indices,"holdout_rows_inspected":0,"test_rows_inspected":0,"parameter_audit":audit,"steps":rows,
  "final_theta_l2_norm":float(core.theta.detach().norm()),"final_gammas":{str(i):float(m.gamma.detach()) for i,m in replacements.items()},
  "unique_trainable_parameter_count":sum(p.numel() for p in params),"frozen_gradient_tensor_count":frozen,"simulator_calls":calls,
  "gpu_peak_allocated_bytes":[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())],"created_at":datetime.now(timezone.utc).isoformat()}
 OUTPUT.write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
