#!/usr/bin/env python3
"""Pre-register disjoint C4 validation for frozen QH033/CC033 checkpoints."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
PROJECT=Path(__file__).resolve().parents[2];ART=PROJECT/"artifacts"
TOKENS=PROJECT/"datasets/processed/c4_validation_qwen38/validation-tokens.pt"
PARENT=ART/"qh033_logit_kl_screen/qh033-logit-kl-screen-analysis.json"
QH=ART/"qh033_logit_kl_screen/qh033-screen-checkpoint.pt";CC=ART/"qh033_logit_kl_screen/cc033-screen-checkpoint.pt"
OUTPUT=ART/"qh033-c4-lock.json";SEED=20260855;LENGTH=256;EVAL_COUNT=128
def sha(path):
 d=hashlib.sha256()
 with path.open("rb") as h:
  for c in iter(lambda:h.read(8<<20),b""):d.update(c)
 return d.hexdigest()
def used_c4():
 used=set();parsed=0
 for path in ART.rglob("*.json"):
  if path==OUTPUT:continue
  try:v=json.loads(path.read_text())
  except Exception:continue
  parsed+=1
  def walk(x,c4=False):
   if isinstance(x,dict):
    text=json.dumps(x,ensure_ascii=False)[:4000].lower();local=c4 or "allenai/c4" in text or "c4_validation" in text
    for key,child in x.items():
     if local and key=="eval_indices" and isinstance(child,list):used.update(int(i) for i in child)
     elif local and key=="block_index" and isinstance(child,int):used.add(child)
     walk(child,local)
   elif isinstance(x,list):
    for child in x:walk(child,c4)
  walk(v)
 return used,parsed
def main():
 for p in (TOKENS,PARENT,QH,CC):
  if not p.exists():raise FileNotFoundError(p)
 parent=json.loads(PARENT.read_text())
 if parent["verdict"]!="continue_to_new_c4_lock":raise RuntimeError("QH033 did not pass train-only screen")
 tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);complete=len(tokens)//LENGTH;used,parsed=used_c4()
 available=np.asarray(sorted(set(range(complete))-used),dtype=np.int64);indices=np.random.default_rng(SEED).permutation(available)[:EVAL_COUNT].tolist()
 if len(indices)!=EVAL_COUNT or set(indices)&used:raise RuntimeError("insufficient or overlapping C4 indices")
 payload={"status":"locked_before_qh033_c4_results","candidate":"QH-033 Pauli-observable replacement after hidden+logit distillation",
  "parent":{"analysis":str(PARENT),"sha256":sha(PARENT),"verdict":parent["verdict"]},
  "frozen_checkpoints":{"qh033":{"path":str(QH),"sha256":sha(QH)},"cc033":{"path":str(CC),"sha256":sha(CC)}},
  "architecture":{"layers":[55,59],"rank":1024,"qubits":10,"observables":60,"trainable_during_prior_training":82,
   "trainable_during_c4_evaluation":0,"net_model_parameter_reduction":3145646},
  "training_history":{"stage1":"256-step final-hidden relative MSE from QH032 lock","stage2":"256-step teacher-logit KL from QH033 lock",
   "additional_training_after_this_lock":0},
  "evaluation_dataset":{"name":"allenai/c4 en validation","revision":"1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
   "selection":"unused token blocks from pinned first-2048-row validation shard","tokens_sha256":sha(TOKENS)},
  "sequence_length":LENGTH,"seed":SEED,"eval_indices":indices,"eval_count":EVAL_COUNT,
  "excluded_prior_c4_indices_count":len(used),"available_before_selection":len(available),"prior_json_parsed":parsed,"overlap":0,
  "evaluation_protocol":{"primary_metric":"mean token negative log likelihood","compression_noninferiority_margin":0.005,"paired_bootstrap_samples":20000,
   "arms":["frozen_dense_base","shared_svd1024_and_QH_own_zero","frozen_QH033","frozen_CC033_equal_parameter","same_QH033_no_entanglement"],
   "quantum_specific_requires":["QH active upper95CI below own-zero","QH upper95CI below equal-parameter CC","QH upper95CI below no-ent intervention"]},
  "test_rows_inspected_or_used_for_fitness":0,"created_at":datetime.now(timezone.utc).isoformat()}
 OUTPUT.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":payload["status"],"eval_count":EVAL_COUNT,"excluded":len(used),
  "available":len(available),"sha256":sha(OUTPUT)},indent=2))
if __name__=="__main__":main()

