#!/usr/bin/env python3
"""One-step output-distribution KL smoke for zero-initialized QH034."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForMultimodalLM
PROJECT=Path(__file__).resolve().parents[2]
TOKENS=PROJECT/"datasets/processed/wikitext2_qwen38/train-tokens.pt"
LOCK=PROJECT/"artifacts/qh031-shared-basis-lock.json"
OUTPUT=PROJECT/"artifacts/qh034-logit-kl-smoke.json"
sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh034_spectral_modulation_replacement import QH034Config,install_qh034,unique_trainable_parameters
def set_proj(model,modules):
    for i,m in modules.items(): model.model.language_model.layers[i].self_attn.v_proj=m
def main():
    if not torch.cuda.is_available(): raise RuntimeError("QH034 KL smoke is GPU-only")
    torch.manual_seed(20260858); torch.cuda.manual_seed_all(20260858)
    lock=json.loads(LOCK.read_text()); tokens=torch.load(TOKENS,map_location="cpu",weights_only=True)
    n=int(lock["sequence_length"]); index=int(lock["train_indices"][0])
    x=tokens[index*n:(index+1)*n].to("cuda:0",dtype=torch.long).unsqueeze(0)
    model_dir=Path((PROJECT/"records/active_model_path.txt").read_text().strip())
    model=AutoModelForMultimodalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,device_map="balanced",low_cpu_mem_usage=True,attn_implementation="sdpa")
    model.eval(); model.config.use_cache=False
    config=QH034Config(); replacements,originals,audit=install_qh034(model,config,branch="quantum")
    params=list(unique_trainable_parameters(replacements)); optimizer=torch.optim.AdamW(params,lr=1.0e-3,weight_decay=0.0)
    set_proj(model,originals)
    with torch.inference_mode(): teacher=model(input_ids=x,use_cache=False).logits[:,:-1].detach()
    set_proj(model,replacements); optimizer.zero_grad(set_to_none=True); started=time.perf_counter()
    student=model(input_ids=x,use_cache=False).logits[:,:-1]; nt=student.shape[0]*student.shape[1]
    loss=F.kl_div(F.log_softmax(student.float(),dim=-1),F.softmax(teacher.float(),dim=-1),reduction="sum")/nt
    loss.backward(); core=replacements[config.layer_indices[0]].core
    theta_grad=core.theta.grad.detach().float(); gamma_grad=torch.stack([m.gamma.grad.detach().float().abs() for m in replacements.values()])
    total=float(torch.nn.utils.clip_grad_norm_(params,1.0)); optimizer.step(); torch.cuda.synchronize()
    frozen=sum(int(p.grad is not None) for p in model.parameters() if not p.requires_grad)
    calls=int(core.circuit_calls.item())
    if theta_grad.norm()<=0 or not torch.isfinite(theta_grad).all() or frozen or calls!=2: raise AssertionError("QH034 KL contract failed")
    payload={"status":"pass","candidate":"QH-034 zero-init output KL","train_block_index":index,"sequence_length":n,
      "teacher_logits_shape":list(teacher.shape),"mean_token_teacher_to_student_kl":float(loss.detach()),
      "theta_gradient_norm":float(theta_grad.norm()),"theta_nonzero_gradient_elements":int((theta_grad!=0).sum()),
      "gamma_gradient_norm":float(gamma_grad.norm()),"total_gradient_norm":total,"post_step_theta_norm":float(core.theta.detach().norm()),
      "step_seconds":time.perf_counter()-started,"simulator_calls":calls,"frozen_gradient_tensor_count":frozen,
      "parameter_audit":audit,"validation_rows_inspected":0,"test_rows_inspected":0,
      "gpu_peak_allocated_bytes":[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]}
    OUTPUT.write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
