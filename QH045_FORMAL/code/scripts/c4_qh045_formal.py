#!/usr/bin/env python3
"""Fresh frozen C4 formal evaluation of QH045 selective entanglement."""
from __future__ import annotations
import argparse, gc, hashlib, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForMultimodalLM

PROJECT=Path(__file__).resolve().parents[2];ART=PROJECT/"artifacts"
TOKENS=PROJECT/"datasets/processed/c4_validation_qwen38/validation-tokens.pt"
START=ART/"qh044_extended_distillation/qh044-checkpoint.pt"
QCKPT=ART/"qh045_selective_entanglement_screen/qh045-checkpoint.pt"
CCKPT=ART/"qh045_selective_entanglement_screen/cc045-checkpoint.pt"
SOURCE=PROJECT/"code/src/quantum_qwen38/qh045_selective_entanglement_ffn_replacement.py"
ENTRY=PROJECT/"code/scripts/c4_qh045_formal.py";LOCK=ART/"qh045-c4-formal-lock.json";OUT=ART/"c4_qh045_formal"
sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh045_selective_entanglement_ffn_replacement import install_qh045

def sha(path):
 d=hashlib.sha256()
 with Path(path).open("rb") as h:
  for chunk in iter(lambda:h.read(8<<20),b""):d.update(chunk)
 return d.hexdigest()

def make_lock():
 used=set()
 for path in ART.rglob("*.json"):
  if path==LOCK:continue
  try:text=path.read_text();value=json.loads(text)
  except Exception:continue
  if "c4" not in str(path).lower() and "c4" not in text.lower():continue
  def walk(x):
   if isinstance(x,dict):
    for key,child in x.items():
     if key in {"eval_indices","evaluation_indices","block_indices"} and isinstance(child,list):used.update(int(i) for i in child)
     elif key=="block_index" and isinstance(child,int):used.add(child)
     walk(child)
   elif isinstance(x,list):
    for child in x:walk(child)
  walk(value)
 tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);available=np.asarray(sorted(set(range(len(tokens)//256))-used),dtype=np.int64);selected=np.random.default_rng(20260887).permutation(available)[:64].tolist()
 if len(selected)!=64 or set(selected)&used:raise RuntimeError("C4 overlap")
 payload={"status":"locked_before_qh045_c4_results","candidate":"QH045 shared-start selective entanglement","promotion_evidence":"train-only QH-noent and QH-CC paired 95% intervals entirely below zero","checkpoints":{"start":{"path":str(START),"sha256":sha(START)},"qh045":{"path":str(QCKPT),"sha256":sha(QCKPT)},"cc045":{"path":str(CCKPT),"sha256":sha(CCKPT)}},"architecture":{"target_layer":31,"removed":267386880,"deployed":655680,"net_reduction":266731200,"trainable_during_screen":64,"formal_training_steps":0},"arms":{"base":"original","noent":"shared frozen QH044 with selectors zero","qh045":"frozen exact selective controlled-RY","cc045":"frozen equal-parameter classical neighbour interaction after same no-ent simulator scaffold","dq":"independent tensor-axis reproduction of QH045"},"dataset":"allenai/c4 en validation processed fixed shard","tokens_sha256":sha(TOKENS),"seed":20260887,"sequence_length":256,"evaluation_indices":selected,"evaluation_count":64,"excluded_prior_c4_indices_count":len(used),"overlap":0,"pre_registered_gates":{"compression_noninferiority":"upper paired 95% CI QH-base < +0.005","active_selective_entanglement":"upper CI QH-noent < 0","equal_parameter_quantum_specific":"upper CI QH-CC < 0","dq_equivalence":"max per-block |QH-DQ| < 2e-5"},"source_hashes":{"source":sha(SOURCE),"entrypoint":sha(ENTRY)},"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()};LOCK.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":payload["status"],"excluded":len(used),"available":len(available),"sha256":sha(LOCK)},indent=2))

def load_model():
 path=Path((PROJECT/"records/active_model_path.txt").read_text().strip());model=AutoModelForMultimodalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,device_map="balanced",low_cpu_mem_usage=True,attn_implementation="sdpa");model.eval();model.config.use_cache=False;model.requires_grad_(False);return model

@torch.inference_mode()
def run(arm):
 lock=json.loads(LOCK.read_text());tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);model=load_model();replacement=None;audit={"arm":"base","net_model_parameter_reduction":0};checkpoint=None
 if arm!="base":
  if arm=="noent":checkpoint=START;state=torch.load(checkpoint,map_location="cpu",weights_only=False)["replacement_state_dict"];branch="quantum"
  elif arm=="qh045":checkpoint=QCKPT;state=torch.load(checkpoint,map_location="cpu",weights_only=False)["replacement_state_dict"];branch="quantum"
  elif arm=="cc045":checkpoint=CCKPT;state=torch.load(checkpoint,map_location="cpu",weights_only=False)["replacement_state_dict"];branch="classical"
  else:checkpoint=QCKPT;state=torch.load(checkpoint,map_location="cpu",weights_only=False)["replacement_state_dict"];branch="dq"
  replacement,_,audit=install_qh045(model,state,branch);replacement.requires_grad_(False)
 rows=[]
 for index in lock["evaluation_indices"]:
  inputs=tokens[index*256:(index+1)*256].to(device=model.device,dtype=torch.long).unsqueeze(0);started=time.perf_counter();loss=model(input_ids=inputs,labels=inputs,use_cache=False).loss;torch.cuda.synchronize();rows.append({"block_index":index,"mean_token_nll":float(loss),"seconds":time.perf_counter()-started})
 nll=float(np.mean([row["mean_token_nll"] for row in rows]));calls=0 if replacement is None else int(replacement.core.circuit_calls.item());expected=0 if arm=="base" else 64
 if calls!=expected:raise AssertionError({"arm":arm,"calls":calls,"expected":expected})
 payload={"status":"ok","arm":arm,"lock_sha256":sha(LOCK),"checkpoint_sha256":None if checkpoint is None else sha(checkpoint),"training_steps":0,"audit":audit,"simulator_calls":calls,"evaluation":{"metrics":{"mean_token_nll":nll,"perplexity":math.exp(nll)},"rows":rows},"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()};OUT.mkdir(parents=True,exist_ok=True);path=OUT/f"formal-{arm}-s{lock['seed']}-e64.json";path.write_text(json.dumps(payload,indent=2));print(json.dumps({"arm":arm,"nll":nll,"calls":calls,"sha256":sha(path)},indent=2),flush=True);del model,replacement;gc.collect();torch.cuda.empty_cache()

def analyze():
 def load(name):return json.loads(next(OUT.glob(f"formal-{name}-*.json")).read_text())
 def compare(a,b,seed):
  delta=np.asarray([x["mean_token_nll"] for x in a["evaluation"]["rows"]])-np.asarray([x["mean_token_nll"] for x in b["evaluation"]["rows"]]);rng=np.random.default_rng(seed);means=delta[rng.integers(0,len(delta),size=(50000,len(delta)))].mean(1);return {"mean_delta_nll":float(delta.mean()),"paired_bootstrap_95_ci":[float(np.quantile(means,.025)),float(np.quantile(means,.975))]}
 data={name:load(name) for name in ("base","noent","qh045","cc045","dq")};nll={name:value["evaluation"]["metrics"]["mean_token_nll"] for name,value in data.items()};comparisons={"qh_minus_base":compare(data["qh045"],data["base"],1),"qh_minus_noent":compare(data["qh045"],data["noent"],2),"qh_minus_cc":compare(data["qh045"],data["cc045"],3),"qh_minus_dq":compare(data["qh045"],data["dq"],4)};dqmax=max(abs(a["mean_token_nll"]-b["mean_token_nll"]) for a,b in zip(data["qh045"]["evaluation"]["rows"],data["dq"]["evaluation"]["rows"]));gates={"compression_noninferiority":comparisons["qh_minus_base"]["paired_bootstrap_95_ci"][1]<.005,"active_selective_entanglement":comparisons["qh_minus_noent"]["paired_bootstrap_95_ci"][1]<0,"equal_parameter_quantum_specific":comparisons["qh_minus_cc"]["paired_bootstrap_95_ci"][1]<0,"dq_equivalence":dqmax<2e-5};payload={"status":"ok","analysis":"QH045 frozen C4-64 formal evaluation","point_nll":nll,"comparisons":comparisons,"dq_max_abs_per_block_nll":dqmax,"gates":gates,"verdict":"quantum_specific_confirmed" if all(gates.values()) else "not_quantum_specific_confirmed","training_steps":0,"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()};(OUT/"qh045-c4-formal-analysis.json").write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))

def run_all():
 for arm in ("base","noent","qh045","cc045","dq"):run(arm)
 analyze()

def main():
 parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["lock","run","run-all","analyze"]);parser.add_argument("--arm",choices=["base","noent","qh045","cc045","dq"]);args=parser.parse_args();{"lock":make_lock,"run":lambda:run(args.arm),"run-all":run_all,"analyze":analyze}[args.mode]()
if __name__=="__main__":main()
