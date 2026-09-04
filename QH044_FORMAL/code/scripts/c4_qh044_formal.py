#!/usr/bin/env python3
"""Fresh C4 formal evaluation of frozen QH044/CC044 plus causal controls."""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForMultimodalLM
PROJECT=Path(__file__).resolve().parents[2];ART=PROJECT/"artifacts";TOKENS=PROJECT/"datasets/processed/c4_validation_qwen38/validation-tokens.pt";QCKPT=ART/"qh044_extended_distillation/qh044-checkpoint.pt";CCKPT=ART/"qh044_extended_distillation/cc044-checkpoint.pt";SOURCE=PROJECT/"code/src/quantum_qwen38/qh037_grouped_qvaf_ffn_replacement.py";ENTRY=PROJECT/"code/scripts/c4_qh044_formal.py";LOCK=ART/"qh044-c4-formal-lock.json";OUT=ART/"c4_qh044_formal";sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh037_grouped_qvaf_ffn_replacement import QH037Config,install_qh037
def sha(p):
 d=hashlib.sha256()
 with Path(p).open("rb") as h:
  for c in iter(lambda:h.read(8<<20),b""):d.update(c)
 return d.hexdigest()
def lock():
 for p in (TOKENS,QCKPT,CCKPT,SOURCE,ENTRY):
  if not p.exists():raise FileNotFoundError(p)
 used=set()
 for path in ART.rglob("*.json"):
  if path==LOCK:continue
  try:text=path.read_text();v=json.loads(text)
  except Exception:continue
  if "c4" not in str(path).lower() and "c4" not in text.lower():continue
  def walk(x):
   if isinstance(x,dict):
    for k,c in x.items():
     if k in {"eval_indices","evaluation_indices","block_indices"} and isinstance(c,list):used.update(int(i) for i in c)
     elif k=="block_index" and isinstance(c,int):used.add(c)
     walk(c)
   elif isinstance(x,list):
    for c in x:walk(c)
  walk(v)
 tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);available=np.asarray(sorted(set(range(len(tokens)//256))-used),dtype=np.int64);selected=np.random.default_rng(20260883).permutation(available)[:128].tolist()
 if len(selected)!=128 or set(selected)&used:raise RuntimeError("C4 overlap")
 payload={"status":"locked_before_qh044_c4_results","candidate":"QH044 extended grouped QVAF full-FFN replacement","promotion_evidence":"train-only QH-CC CI entirely below zero","checkpoints":{"qh044":{"path":str(QCKPT),"sha256":sha(QCKPT),"steps":1256},"cc044":{"path":str(CCKPT),"sha256":sha(CCKPT),"steps":1256}},"architecture":{"target_layer":31,"original_parameters_removed":267386880,"replacement_parameters":655616,"net_model_parameter_reduction":266731264,"quantum":"16 independent 4q depth2 exact statevectors","simulator":"CUDA float32 exact statevector"},"arms":{"base":"original","zero":"delete layer31 FFN","qh044":"frozen quantum checkpoint","cc044":"frozen equal-parameter classical checkpoint","qh044_noent":"QH checkpoint with controlled-RY calls disabled","qh044_dq":"QH checkpoint on independent tensor-axis simulator"},"dataset":"allenai/c4 en validation processed fixed shard","tokens_sha256":sha(TOKENS),"sequence_length":256,"seed":20260883,"evaluation_indices":selected,"evaluation_count":128,"excluded_prior_c4_indices_count":len(used),"available_before_selection":len(available),"overlap":0,"training_steps":0,"pre_registered_gates":{"compression_noninferiority":"upper paired bootstrap 95% CI QH-base < +0.005","active_causality":"upper CI QH-zero < 0","equal_parameter_quantum_specific":"upper CI QH-CC < 0","entanglement_specific":"upper CI QH-noent < 0","dq_equivalence":"max absolute per-block NLL difference QH-DQ < 2e-5"},"source_hashes":{"replacement":sha(SOURCE),"entrypoint":sha(ENTRY)},"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()};LOCK.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":payload["status"],"excluded":len(used),"available":len(available),"sha256":sha(LOCK)},indent=2))
def load_model():
 path=Path((PROJECT/"records/active_model_path.txt").read_text().strip());m=AutoModelForMultimodalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,device_map="balanced",low_cpu_mem_usage=True,attn_implementation="sdpa");m.eval();m.config.use_cache=False;m.requires_grad_(False);return m
@torch.inference_mode()
def run(arm):
 lockv=json.loads(LOCK.read_text());tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);m=load_model();audit={"arm":"base","net_model_parameter_reduction":0};rep=None
 if arm!="base":
  branch={"zero":"classical","qh044":"quantum","cc044":"classical","qh044_noent":"no_ent","qh044_dq":"dq"}[arm];rep,_,audit=install_qh037(m,QH037Config(),branch);ckpt=torch.load(CCKPT if arm in {"zero","cc044"} else QCKPT,map_location="cpu",weights_only=False);rep.load_state_dict(ckpt["replacement_state_dict"],strict=True);rep.requires_grad_(False)
  if arm=="zero":rep.branch_enabled=False
 rows=[]
 for i in lockv["evaluation_indices"]:
  x=tokens[i*256:(i+1)*256].to(device=m.device,dtype=torch.long).unsqueeze(0);started=time.perf_counter();loss=m(input_ids=x,labels=x,use_cache=False).loss;torch.cuda.synchronize();rows.append({"block_index":i,"mean_token_nll":float(loss),"seconds":time.perf_counter()-started})
 nll=float(np.mean([r["mean_token_nll"] for r in rows]));calls=int(rep.core.circuit_calls.item()) if rep is not None else 0;expected=128 if arm in {"qh044","qh044_noent","qh044_dq"} else 0
 if calls!=expected:raise AssertionError({"calls":calls,"expected":expected})
 payload={"status":"ok","arm":arm,"lock_sha256":sha(LOCK),"checkpoint_sha256":None if arm=="base" else sha(CCKPT if arm in {"zero","cc044"} else QCKPT),"training_steps":0,"audit":audit,"simulator_calls":calls,"evaluation":{"metrics":{"mean_token_nll":nll,"perplexity":math.exp(nll),"mean_seconds":float(np.mean([r["seconds"] for r in rows]))},"rows":rows},"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()};OUT.mkdir(parents=True,exist_ok=True);path=OUT/f"formal-{arm}-s{lockv['seed']}-e128.json";path.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":"ok","arm":arm,"nll":nll,"simulator_calls":calls,"sha256":sha(path)},indent=2))
def analyze():
 def load(n):return json.loads(next(OUT.glob(f"formal-{n}-*.json")).read_text())
 def ci(a,b,s):
  d=np.asarray([x["mean_token_nll"] for x in a["evaluation"]["rows"]])-np.asarray([x["mean_token_nll"] for x in b["evaluation"]["rows"]]);r=np.random.default_rng(s);m=d[r.integers(0,len(d),size=(50000,len(d)))].mean(1);return {"mean_delta_nll":float(d.mean()),"paired_bootstrap_95_ci":[float(np.quantile(m,.025)),float(np.quantile(m,.975))]}
 names=("base","zero","qh044","cc044","qh044_noent","qh044_dq");d={n:load(n) for n in names};nll={n:v["evaluation"]["metrics"]["mean_token_nll"] for n,v in d.items()};cmp={"qh_minus_base":ci(d["qh044"],d["base"],1),"qh_minus_zero":ci(d["qh044"],d["zero"],2),"qh_minus_cc":ci(d["qh044"],d["cc044"],3),"qh_minus_noent":ci(d["qh044"],d["qh044_noent"],4),"qh_minus_dq":ci(d["qh044"],d["qh044_dq"],5)};dqmax=max(abs(x["mean_token_nll"]-y["mean_token_nll"]) for x,y in zip(d["qh044"]["evaluation"]["rows"],d["qh044_dq"]["evaluation"]["rows"]));g={"compression_noninferiority":cmp["qh_minus_base"]["paired_bootstrap_95_ci"][1]<.005,"active_causality":cmp["qh_minus_zero"]["paired_bootstrap_95_ci"][1]<0,"equal_parameter_quantum_specific":cmp["qh_minus_cc"]["paired_bootstrap_95_ci"][1]<0,"entanglement_specific":cmp["qh_minus_noent"]["paired_bootstrap_95_ci"][1]<0,"dq_equivalence":dqmax<2e-5};p={"status":"ok","analysis":"QH044 frozen C4-128 formal evaluation","point_nll":nll,"comparisons":cmp,"dq_max_abs_per_block_nll":dqmax,"gates":g,"verdict":"quantum_specific_confirmed" if all(g.values()) else "not_quantum_specific_confirmed","training_steps":0,"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()};path=OUT/"qh044-c4-formal-analysis.json";path.write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument("mode",choices=["lock","run","analyze"]);p.add_argument("--arm",choices=["base","zero","qh044","cc044","qh044_noent","qh044_dq"]);a=p.parse_args();{"lock":lambda:lock(),"run":lambda:run(a.arm),"analyze":lambda:analyze()}[a.mode]()
if __name__=="__main__":main()
