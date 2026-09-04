#!/usr/bin/env python3
"""Run locked base/zero/QH037/CC037 train-only arms."""
from __future__ import annotations
import argparse,hashlib,json,math,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from transformers import AutoModelForMultimodalLM
PROJECT=Path(__file__).resolve().parents[2];LOCK=PROJECT/"artifacts/qh037-grouped-qvaf-screen-lock.json";TOKENS=PROJECT/"datasets/processed/wikitext2_qwen38/train-tokens.pt";OUT=PROJECT/"artifacts/qh037_grouped_qvaf_screen";sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh037_grouped_qvaf_ffn_replacement import ExactGroupedQVAFCore,QH037Config,install_qh037,unique_trainable_parameters
def sha(p):
 d=hashlib.sha256()
 with p.open("rb") as h:
  for c in iter(lambda:h.read(8<<20),b""):d.update(c)
 return d.hexdigest()
def block(tokens,i,n,device):return tokens[i*n:(i+1)*n].to(device=device,dtype=torch.long).unsqueeze(0)
def set_mlp(model,cfg,module):model.model.language_model.layers[cfg.target_layer].mlp=module
@torch.inference_mode()
def evaluate(model,tokens,indices,n):
 rows=[]
 for i in indices:
  x=block(tokens,int(i),n,model.device);started=time.perf_counter();loss=model(input_ids=x,labels=x,use_cache=False).loss;torch.cuda.synchronize();rows.append({"block_index":int(i),"mean_token_nll":float(loss),"seconds":time.perf_counter()-started})
 mean=float(np.mean([r["mean_token_nll"] for r in rows]));return {"metrics":{"mean_token_nll":mean,"perplexity":math.exp(mean),"mean_seconds":float(np.mean([r["seconds"] for r in rows]))},"rows":rows}
def train(model,replacement,original,tokens,lock,cfg):
 params=list(unique_trainable_parameters(replacement));p=lock["training_protocol"];opt=torch.optim.AdamW(params,lr=p["learning_rate"],weight_decay=p["weight_decay"]);before=int(replacement.core.circuit_calls.item());losses=[];seconds=[];theta=[];down=[];up=[]
 for pos,i in enumerate(lock["train_indices"],1):
  x=block(tokens,int(i),lock["sequence_length"],model.device);set_mlp(model,cfg,original)
  with torch.inference_mode():teacher=model(input_ids=x,use_cache=False).logits[:,:-1].detach()
  set_mlp(model,cfg,replacement);opt.zero_grad(set_to_none=True);started=time.perf_counter();student=model(input_ids=x,use_cache=False).logits[:,:-1];loss=F.kl_div(F.log_softmax(student.float(),-1),F.softmax(teacher.float(),-1),reduction="sum")/(student.shape[0]*student.shape[1]);loss.backward();theta.append(float(replacement.core.theta.grad.float().norm()));down.append(float(replacement.down.weight.grad.float().norm()));up.append(float(replacement.up.weight.grad.float().norm()));torch.nn.utils.clip_grad_norm_(params,p["gradient_clip_norm"]);opt.step();torch.cuda.synchronize();losses.append(float(loss.detach()));seconds.append(time.perf_counter()-started)
  if pos%64==0:print(json.dumps({"trained":pos,"recent_kl":float(np.mean(losses[-64:])),"recent_seconds":float(np.mean(seconds[-64:]))}),flush=True)
  del teacher,student,loss,x
 calls=int(replacement.core.circuit_calls.item())-before;expected=len(lock["train_indices"]) if isinstance(replacement.core,ExactGroupedQVAFCore) else 0;frozen=sum(int(p.grad is not None) for p in model.parameters() if not p.requires_grad)
 if calls!=expected or frozen:raise AssertionError({"calls":calls,"expected":expected,"frozen":frozen})
 return {"steps":len(losses),"kl_first32":float(np.mean(losses[:32])),"kl_last32":float(np.mean(losses[-32:])),"kl_min":float(np.min(losses)),"theta_grad_mean":float(np.mean(theta)),"down_grad_mean":float(np.mean(down)),"up_grad_mean":float(np.mean(up)),"mean_student_step_seconds":float(np.mean(seconds)),"simulator_calls":calls,"frozen_gradient_tensor_count":frozen,"losses":losses},opt
def save_checkpoint(replacement,opt,path,candidate,lock,cfg,training):
 torch.save({"candidate":candidate,"config":cfg.__dict__,"lock_sha256":sha(LOCK),"steps":training["steps"],"replacement_state_dict":{k:v.detach().cpu() for k,v in replacement.state_dict().items()},"optimizer_state_dict":opt.state_dict(),"training_summary":{k:v for k,v in training.items() if k!="losses"}},path)
def main():
 p=argparse.ArgumentParser();p.add_argument("--candidate",choices=["base","zero","qh037","cc037"],required=True);a=p.parse_args();lock=json.loads(LOCK.read_text())
 if lock["status"]!="locked_before_qh037_train_only_screen_results":raise RuntimeError("invalid lock")
 random.seed(lock["seed"]);np.random.seed(lock["seed"]);torch.manual_seed(lock["seed"]);torch.cuda.manual_seed_all(lock["seed"]);tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);model_dir=Path((PROJECT/"records/active_model_path.txt").read_text().strip());model=AutoModelForMultimodalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,device_map="balanced",low_cpu_mem_usage=True,attn_implementation="sdpa");model.eval();model.config.use_cache=False;cfg=QH037Config();replacement=original=opt=None
 if a.candidate=="base":model.requires_grad_(False);audit={"candidate":"base","net_model_parameter_reduction":0};training={"steps":0}
 else:
  branch="classical" if a.candidate=="cc037" else "quantum";replacement,original,audit=install_qh037(model,cfg,branch=branch)
  if a.candidate=="zero":replacement.branch_enabled=False;replacement.requires_grad_(False);training={"steps":0}
  else:training,opt=train(model,replacement,original,tokens,lock,cfg)
 evaluation=evaluate(model,tokens,lock["eval_indices"],lock["sequence_length"]);OUT.mkdir(parents=True,exist_ok=True)
 if a.candidate in {"qh037","cc037"}:save_checkpoint(replacement,opt,OUT/f"{a.candidate}-screen-checkpoint.pt",a.candidate,lock,cfg,training)
 payload={"status":"ok","candidate":a.candidate,"lock_sha256":sha(LOCK),"dataset":{"train_indices":lock["train_indices"] if training["steps"] else [],"eval_indices":lock["eval_indices"],"validation_rows_inspected":0,"test_rows_inspected":0},"injection":audit,"training":training,"evaluation":evaluation,"created_at":datetime.now(timezone.utc).isoformat()};path=OUT/f"screen-{a.candidate}-s{lock['seed']}-n{lock['train_count']}-e{lock['eval_count']}.json";path.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":"ok","candidate":a.candidate,"nll":evaluation["metrics"]["mean_token_nll"],"sha256":sha(path)},indent=2))
if __name__=="__main__":main()
