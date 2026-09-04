#!/usr/bin/env python3
"""Lock fresh train-only blocks for QH038 shifted-ring FC-VQC screen."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
PROJECT=Path(__file__).resolve().parents[2];ART=PROJECT/"artifacts";TOKENS=PROJECT/"datasets/processed/wikitext2_qwen38/train-tokens.pt"
STATIC=ART/"qh038-static-validation.json";QSMOKE=ART/"qh038-full-model-smoke-quantum.json";CSMOKE=ART/"qh038-full-model-smoke-classical.json";SOURCE=PROJECT/"code/src/quantum_qwen38/qh038_shifted_ring_fcvqc_ffn_replacement.py";OUTPUT=ART/"qh038-shifted-ring-screen-lock.json"
SEED=20260867;LENGTH=256;TRAIN_COUNT=256;EVAL_COUNT=64
def sha(path):
 d=hashlib.sha256()
 with path.open("rb") as h:
  for c in iter(lambda:h.read(8<<20),b""):d.update(c)
 return d.hexdigest()
def used_indices():
 used=set();parsed=0
 for path in ART.rglob("*.json"):
  if path==OUTPUT:continue
  try:v=json.loads(path.read_text())
  except Exception:continue
  parsed+=1
  def walk(x):
   if isinstance(x,dict):
    for k,c in x.items():
     if k in {"train_indices","train_block_indices","eval_indices","evaluation_indices","block_indices"} and isinstance(c,list):used.update(int(i) for i in c)
     elif k in {"train_block_index","block_index"} and isinstance(c,int):used.add(c)
     walk(c)
   elif isinstance(x,list):
    for c in x:walk(c)
  walk(v)
 return used,parsed
def main():
 for p in (TOKENS,STATIC,QSMOKE,CSMOKE,SOURCE):
  if not p.exists():raise FileNotFoundError(p)
 if json.loads(STATIC.read_text())["status"]!="pass" or json.loads(QSMOKE.read_text())["status"]!="pass" or json.loads(CSMOKE.read_text())["status"]!="pass":raise RuntimeError("QH038 prerequisites invalid")
 tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);complete=len(tokens)//LENGTH;used,parsed=used_indices();available=np.asarray(sorted(set(range(complete))-used),dtype=np.int64);selected=np.random.default_rng(SEED).permutation(available)[:TRAIN_COUNT+EVAL_COUNT].tolist();train=selected[:TRAIN_COUNT];evaluation=selected[TRAIN_COUNT:]
 if len(selected)!=TRAIN_COUNT+EVAL_COUNT or set(train)&set(evaluation) or set(selected)&used:raise AssertionError("QH038 split invalid")
 payload={"status":"locked_before_qh038_train_only_screen_results","candidate":"QH-038 shifted-ring inter-block FC-VQC complete FFN replacement","parent":"QH-037 plus arXiv:2604.23931v2 / arXiv:2602.16623 inter-block connectivity",
  "single_mutation":"same target/rank/parameters and two local stages as QH037; between stages cyclically shift lane q by -q blocks and regroup","architecture":{"hidden_size":5120,"intermediate_size":17408,"target_layer":31,"latent_size":64,"groups":16,"qubits_per_group":4,"stages":2,"circuit_parameters":256,"low_rank_parameters":655360,"trainable":655616,"net_model_parameter_reduction":266731264,"branch_off":"exact zero FFN"},
  "controls":{"base":"original layer31 SwiGLU","own_zero":"same replacement branch disabled","equal_parameter_classical":"same projections, shifted-ring regrouping, and 256-parameter grouped sine/product stages","future_no_ent":"remove controlled-RY only","dequantized":"independent tensor-axis implementation"},
  "evidence":{n:{"path":str(p),"sha256":sha(p)} for n,p in (("static",STATIC),("quantum_smoke",QSMOKE),("classical_smoke",CSMOKE))},"source":{"path":str(SOURCE),"sha256":sha(SOURCE)},"dataset":"Salesforce/wikitext wikitext-2-raw-v1 train only","tokens_sha256":sha(TOKENS),"sequence_length":LENGTH,"seed":SEED,"train_indices":train,"eval_indices":evaluation,"train_count":TRAIN_COUNT,"eval_count":EVAL_COUNT,"excluded_prior_indices_count":len(used),"available_before_selection":len(available),"prior_json_parsed":parsed,"overlap":0,
  "training_protocol":{"objective":"mean-token KL(teacher original FFN || replacement student), temperature 1","optimizer":"AdamW","learning_rate":0.001,"weight_decay":0.0,"gradient_clip_norm":1.0,"steps":TRAIN_COUNT,"all_other_parameters_frozen":True,"save_checkpoint_optimizer_source_hash":True},
  "screen_rule":{"continue_to_new_C4_only_if":["QH038 point NLL lower than own-zero","QH038 point NLL lower than equal-parameter CC038","QH038 no worse than base by 0.005 NLL"],"otherwise":"reject or mutate before C4","no_claim_from_train_screen":True},"validation_rows_inspected":0,"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()}
 OUTPUT.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":payload["status"],"train_count":TRAIN_COUNT,"eval_count":EVAL_COUNT,"excluded":len(used),"available":len(available),"sha256":sha(OUTPUT)},indent=2))
if __name__=="__main__":main()
