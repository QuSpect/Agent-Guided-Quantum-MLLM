#!/usr/bin/env python3
"""Frozen-checkpoint QH033 paired C4 evaluation."""
from __future__ import annotations
import argparse,hashlib,json,math,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForMultimodalLM
PROJECT=Path(__file__).resolve().parents[2];LOCK=PROJECT/"artifacts/qh033-c4-lock.json"
TOKENS=PROJECT/"datasets/processed/c4_validation_qwen38/validation-tokens.pt";OUT=PROJECT/"artifacts/c4_qh033"
sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh032_pauli_observable_replacement import QH032Config,install_qh032
def sha(path):
 d=hashlib.sha256()
 with path.open("rb") as h:
  for c in iter(lambda:h.read(8<<20),b""):d.update(c)
 return d.hexdigest()
def args():
 p=argparse.ArgumentParser();p.add_argument("--candidate",choices=["base","sharedsvd1024","qh033","cc033"],required=True);return p.parse_args()
def load(replacements,path):
 state=torch.load(path,map_location="cpu",weights_only=True);first=sorted(replacements)[0]
 with torch.no_grad():
  replacements[first].core.theta.copy_(state["core.theta"].to(replacements[first].core.theta.device))
  replacements[first].core.alpha.copy_(state["core.alpha"].to(replacements[first].core.alpha.device))
  for i,m in replacements.items():m.gamma.copy_(state[f"gamma.{i}"].to(m.gamma.device))
@torch.inference_mode()
def evaluate(model,tokens,indices,n):
 rows=[]
 for pos,i in enumerate(indices,1):
  x=tokens[i*n:(i+1)*n].to(device=model.device,dtype=torch.long).unsqueeze(0);started=time.perf_counter();loss=model(input_ids=x,labels=x,use_cache=False).loss
  rows.append({"block_index":i,"mean_token_nll":float(loss),"seconds":time.perf_counter()-started})
  if pos%64==0:print(json.dumps({"evaluated":pos,"total":len(indices),"running_nll":float(np.mean([r["mean_token_nll"] for r in rows]))}),flush=True)
 nll=float(np.mean([r["mean_token_nll"] for r in rows]));return {"metrics":{"mean_token_nll":nll,"perplexity":math.exp(nll),
  "mean_seconds":float(np.mean([r["seconds"] for r in rows]))},"rows":rows}
def main():
 a=args();lock=json.loads(LOCK.read_text())
 if lock["status"]!="locked_before_qh033_c4_results":raise RuntimeError("invalid lock")
 random.seed(lock["seed"]);np.random.seed(lock["seed"]);torch.manual_seed(lock["seed"]);torch.cuda.manual_seed_all(lock["seed"])
 tokens=torch.load(TOKENS,map_location="cpu",weights_only=True)
 if sha(TOKENS)!=lock["evaluation_dataset"]["tokens_sha256"]:raise RuntimeError("token hash changed")
 model_dir=Path((PROJECT/"records/active_model_path.txt").read_text().strip());model=AutoModelForMultimodalLM.from_pretrained(
  model_dir,local_files_only=True,dtype=torch.bfloat16,device_map="balanced",low_cpu_mem_usage=True,attn_implementation="sdpa")
 model.eval();model.config.use_cache=False;model.requires_grad_(False);replacements=None
 if a.candidate=="base":audit={"candidate":"base","net_model_parameter_reduction":0}
 else:
  branch="classical" if a.candidate=="cc033" else "quantum";replacements,_,audit=install_qh032(model,QH032Config(),branch=branch)
  for m in replacements.values():m.requires_grad_(False)
  if a.candidate=="sharedsvd1024":
   for m in replacements.values():m.branch_enabled=False
  else:
   info=lock["frozen_checkpoints"][a.candidate];path=Path(info["path"])
   if sha(path)!=info["sha256"]:raise RuntimeError("checkpoint hash mismatch")
   load(replacements,path)
 indices=[int(i) for i in lock["eval_indices"]];n=lock["sequence_length"];active=evaluate(model,tokens,indices,n);no_ent=None;calls=0
 if a.candidate=="qh033":
  core=replacements[QH032Config().layer_indices[0]].core;expected=len(indices)*2;calls=int(core.circuit_calls.item())
  if calls!=expected:raise AssertionError("active call mismatch")
  core.use_entanglers=False;before=calls;no_ent=evaluate(model,tokens,indices,n);after=int(core.circuit_calls.item())
  if after-before!=expected:raise AssertionError("no-ent call mismatch")
  calls=after
 elif a.candidate=="cc033" and int(replacements[QH032Config().layer_indices[0]].core.circuit_calls.item())!=0:raise AssertionError("CC simulator call")
 OUT.mkdir(parents=True,exist_ok=True);payload={"status":"ok","candidate":a.candidate,"lock_sha256":sha(LOCK),
  "dataset":{"eval_indices":indices,"test_rows_inspected_or_used_for_fitness":0},"injection":audit,"training":{"steps":0},
  "evaluation_active":active,"evaluation_no_entanglement":no_ent,"simulator_calls":calls,"created_at":datetime.now(timezone.utc).isoformat()}
 path=OUT/f"c4-{a.candidate}-s{lock['seed']}-e{lock['eval_count']}.json";path.write_text(json.dumps(payload,indent=2))
 print(json.dumps({"status":"ok","candidate":a.candidate,"nll":active["metrics"]["mean_token_nll"],"sha256":sha(path)},indent=2))
if __name__=="__main__":main()

