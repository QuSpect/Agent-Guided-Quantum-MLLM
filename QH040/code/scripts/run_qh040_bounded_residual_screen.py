#!/usr/bin/env python3
"""Run locked base/scaffold/QH040/CC040 train-only arms."""
from __future__ import annotations
import argparse,json,math,random,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForMultimodalLM
import run_qh039_frozen_scaffold_screen as common
PROJECT=Path(__file__).resolve().parents[2];LOCK=PROJECT/"artifacts/qh040-bounded-residual-screen-lock.json";TOKENS=PROJECT/"datasets/processed/wikitext2_qwen38/train-tokens.pt";BACKBONE=PROJECT/"artifacts/qh037_grouped_qvaf_screen/cc037-screen-checkpoint.pt";SOURCE=PROJECT/"code/src/quantum_qwen38/qh040_bounded_quantum_residual.py";ENTRY=PROJECT/"code/scripts/run_qh040_bounded_residual_screen.py";OUT=PROJECT/"artifacts/qh040_bounded_residual_screen";sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh040_bounded_quantum_residual import QH040Config,install_qh040
def sha(p):return common.sha(p)
def save_checkpoint(rep,opt,path,candidate,cfg,training,model_dir):
 torch.save({"candidate":candidate,"config":cfg.__dict__,"data_lock_sha256":sha(LOCK),"source_sha256":sha(SOURCE),"entrypoint_sha256":sha(ENTRY),"backbone_checkpoint_sha256":sha(BACKBONE),"base_model_path":str(model_dir),"base_config_sha256":sha(model_dir/"config.json"),"steps":training["steps"],"precision":{"base":"bfloat16","simulator":"float32 exact statevector","device":"CUDA GPU only"},"replacement_state_dict":{k:v.detach().cpu() for k,v in rep.state_dict().items()},"optimizer":"AdamW","optimizer_state_dict":opt.state_dict(),"scheduler_state_dict":None,"training_summary":{k:v for k,v in training.items() if k!="losses"}},path)
def main():
 p=argparse.ArgumentParser();p.add_argument("--candidate",choices=["base","scaffold","qh040","cc040"],required=True);a=p.parse_args();lock=json.loads(LOCK.read_text())
 if lock["status"]!="locked_before_qh040_train_only_screen_results":raise RuntimeError("invalid lock")
 random.seed(lock["seed"]);np.random.seed(lock["seed"]);torch.manual_seed(lock["seed"]);torch.cuda.manual_seed_all(lock["seed"]);tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);model_dir=Path((PROJECT/"records/active_model_path.txt").read_text().strip());model=AutoModelForMultimodalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,device_map="balanced",low_cpu_mem_usage=True,attn_implementation="sdpa");model.eval();model.config.use_cache=False;cfg=QH040Config();rep=original=opt=None
 if a.candidate=="base":model.requires_grad_(False);audit={"candidate":"base","net_model_parameter_reduction":0};training={"steps":0}
 else:
  branch="classical" if a.candidate=="cc040" else "quantum";rep,original,audit=install_qh040(model,BACKBONE,cfg,branch)
  if a.candidate=="scaffold":rep.correction_enabled=False;rep.requires_grad_(False);training={"steps":0}
  else:
   training,opt=common.train(model,rep,original,tokens,lock,cfg);eff=rep.effective_alpha().detach().float();training["final_effective_alpha_norm"]=float(eff.norm());training["final_max_abs_effective_alpha"]=float(eff.abs().max())
 evaluation=common.evaluate(model,tokens,lock["eval_indices"],lock["sequence_length"]);OUT.mkdir(parents=True,exist_ok=True)
 if a.candidate in {"qh040","cc040"}:save_checkpoint(rep,opt,OUT/f"{a.candidate}-screen-checkpoint.pt",a.candidate,cfg,training,model_dir)
 payload={"status":"ok","candidate":a.candidate,"lock_sha256":sha(LOCK),"source_sha256":sha(SOURCE),"dataset":{"train_indices":lock["train_indices"] if training["steps"] else [],"eval_indices":lock["eval_indices"],"validation_rows_inspected":0,"test_rows_inspected":0},"injection":audit,"training":training,"evaluation":evaluation,"created_at":datetime.now(timezone.utc).isoformat()};path=OUT/f"screen-{a.candidate}-s{lock['seed']}-n{lock['train_count']}-e{lock['eval_count']}.json";path.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":"ok","candidate":a.candidate,"nll":evaluation["metrics"]["mean_token_nll"],"sha256":sha(path)},indent=2))
if __name__=="__main__":main()
