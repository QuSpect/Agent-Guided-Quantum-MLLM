#!/usr/bin/env python3
"""Run locked base/shared/QH035/CC035 train-only arms."""
from __future__ import annotations
import argparse,hashlib,json,math,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from transformers import AutoModelForMultimodalLM
PROJECT=Path(__file__).resolve().parents[2];LOCK=PROJECT/"artifacts/qh035-norm-screen-lock.json";TOKENS=PROJECT/"datasets/processed/wikitext2_qwen38/train-tokens.pt";OUT=PROJECT/"artifacts/qh035_norm_screen"
sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh035_shared_rmsnorm_replacement import ExactQuantumNormCore,QH035Config,install_qh035,unique_trainable_parameters
def sha(p):
 d=hashlib.sha256()
 with p.open("rb") as h:
  for c in iter(lambda:h.read(8<<20),b""):d.update(c)
 return d.hexdigest()
def set_norms(model,modules,cfg):
 for i,m in modules.items():setattr(model.model.language_model.layers[i],cfg.norm_name,m)
def block(tokens,i,n,device):return tokens[i*n:(i+1)*n].to(device=device,dtype=torch.long).unsqueeze(0)
def parse():
 p=argparse.ArgumentParser();p.add_argument("--candidate",choices=["base","sharednorm","qh035","cc035"],required=True);return p.parse_args()
@torch.inference_mode()
def evaluate(model,tokens,indices,n):
 rows=[]
 for i in indices:
  x=block(tokens,i,n,model.device);t=time.perf_counter();loss=model(input_ids=x,labels=x,use_cache=False).loss;rows.append({"block_index":i,"mean_token_nll":float(loss),"seconds":time.perf_counter()-t})
 mean=float(np.mean([r["mean_token_nll"] for r in rows]));return {"metrics":{"mean_token_nll":mean,"perplexity":math.exp(mean),"mean_seconds":float(np.mean([r["seconds"] for r in rows]))},"rows":rows}
def train(model,replacements,originals,tokens,lock,cfg):
 params=list(unique_trainable_parameters(replacements));p=lock["training_protocol"];opt=torch.optim.AdamW(params,lr=p["learning_rate"],weight_decay=p["weight_decay"])
 core=replacements[cfg.layer_indices[0]].core;before=int(core.circuit_calls.item());losses=[];seconds=[];theta=[];gamma=[]
 for pos,i in enumerate(lock["train_indices"],1):
  x=block(tokens,int(i),lock["sequence_length"],model.device);set_norms(model,originals,cfg)
  with torch.inference_mode():teacher=model(input_ids=x,use_cache=False).logits[:,:-1].detach()
  set_norms(model,replacements,cfg);opt.zero_grad(set_to_none=True);started=time.perf_counter();student=model(input_ids=x,use_cache=False).logits[:,:-1];nt=student.shape[0]*student.shape[1]
  loss=F.kl_div(F.log_softmax(student.float(),dim=-1),F.softmax(teacher.float(),dim=-1),reduction="sum")/nt;loss.backward();theta.append(float(core.theta.grad.norm()));gamma.append(float(torch.stack([m.gamma.grad.abs() for m in replacements.values()]).norm()))
  torch.nn.utils.clip_grad_norm_(params,p["gradient_clip_norm"]);opt.step();torch.cuda.synchronize();losses.append(float(loss.detach()));seconds.append(time.perf_counter()-started)
  if pos%64==0:print(json.dumps({"trained":pos,"recent_kl":float(np.mean(losses[-64:])),"recent_seconds":float(np.mean(seconds[-64:]))}),flush=True)
  del teacher,student,loss,x
 calls=int(core.circuit_calls.item())-before;expected=len(lock["train_indices"])*len(cfg.layer_indices) if isinstance(core,ExactQuantumNormCore) else 0
 if calls!=expected:raise AssertionError(f"calls {calls}!={expected}")
 frozen=sum(int(p.grad is not None) for p in model.parameters() if not p.requires_grad)
 if frozen:raise AssertionError("frozen gradient")
 return {"steps":len(losses),"kl_first32":float(np.mean(losses[:32])),"kl_last32":float(np.mean(losses[-32:])),"kl_min":float(np.min(losses)),
  "theta_grad_mean":float(np.mean(theta)),"gamma_grad_mean":float(np.mean(gamma)),"mean_student_step_seconds":float(np.mean(seconds)),"simulator_calls":calls,
  "frozen_gradient_tensor_count":frozen,"final_theta_norm":float(core.theta.detach().norm()),"final_gammas":{str(i):float(m.gamma.detach()) for i,m in replacements.items()},"losses":losses}
def save_ckpt(replacements,path):
 first=sorted(replacements)[0];torch.save({"core.theta":replacements[first].core.theta.detach().cpu(),**{f"gamma.{i}":m.gamma.detach().cpu() for i,m in replacements.items()}},path)
def main():
 a=parse();lock=json.loads(LOCK.read_text());
 if lock["status"]!="locked_before_qh035_train_only_screen_results":raise RuntimeError("invalid lock")
 random.seed(lock["seed"]);np.random.seed(lock["seed"]);torch.manual_seed(lock["seed"]);torch.cuda.manual_seed_all(lock["seed"]);tokens=torch.load(TOKENS,map_location="cpu",weights_only=True)
 model_dir=Path((PROJECT/"records/active_model_path.txt").read_text().strip());model=AutoModelForMultimodalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,device_map="balanced",low_cpu_mem_usage=True,attn_implementation="sdpa")
 model.eval();model.config.use_cache=False;cfg=QH035Config();replacements=originals=None
 if a.candidate=="base":model.requires_grad_(False);audit={"candidate":"base","net_model_parameter_reduction":0};training={"steps":0}
 else:
  branch="classical" if a.candidate=="cc035" else "quantum";replacements,originals,audit=install_qh035(model,cfg,branch=branch)
  if a.candidate=="sharednorm":
   for m in replacements.values():m.branch_enabled=False;m.requires_grad_(False)
   training={"steps":0}
  else:training=train(model,replacements,originals,tokens,lock,cfg)
 evaluation=evaluate(model,tokens,[int(i) for i in lock["eval_indices"]],lock["sequence_length"]);OUT.mkdir(parents=True,exist_ok=True)
 if replacements is not None and a.candidate in {"qh035","cc035"}:save_ckpt(replacements,OUT/f"{a.candidate}-screen-checkpoint.pt")
 payload={"status":"ok","candidate":a.candidate,"lock_sha256":sha(LOCK),"dataset":{"train_indices":lock["train_indices"] if training["steps"] else [],"eval_indices":lock["eval_indices"],
  "validation_rows_inspected":0,"test_rows_inspected":0},"injection":audit,"training":training,"evaluation":evaluation,"created_at":datetime.now(timezone.utc).isoformat()}
 path=OUT/f"screen-{a.candidate}-s{lock['seed']}-n{lock['train_count']}-e{lock['eval_count']}.json";path.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":"ok","candidate":a.candidate,"nll":evaluation["metrics"]["mean_token_nll"],"sha256":sha(path)},indent=2))
if __name__=="__main__":main()
