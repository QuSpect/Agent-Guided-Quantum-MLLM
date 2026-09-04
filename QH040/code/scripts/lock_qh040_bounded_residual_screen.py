#!/usr/bin/env python3
"""Lock fresh train-only blocks for QH040."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
PROJECT=Path(__file__).resolve().parents[2];ART=PROJECT/"artifacts";TOKENS=PROJECT/"datasets/processed/wikitext2_qwen38/train-tokens.pt";BACKBONE=ART/"qh037_grouped_qvaf_screen/cc037-screen-checkpoint.pt";STATIC=ART/"qh040-static-validation.json";QSMOKE=ART/"qh040-full-model-smoke-quantum.json";CSMOKE=ART/"qh040-full-model-smoke-classical.json";SOURCE=PROJECT/"code/src/quantum_qwen38/qh040_bounded_quantum_residual.py";OUTPUT=ART/"qh040-bounded-residual-screen-lock.json";SEED=20260871;LENGTH=256;TRAIN_COUNT=256;EVAL_COUNT=64;STEPS=512
def sha(path):
 d=hashlib.sha256()
 with path.open("rb") as h:
  for c in iter(lambda:h.read(8<<20),b""):d.update(c)
 return d.hexdigest()
def main():
 for p in (TOKENS,BACKBONE,STATIC,QSMOKE,CSMOKE,SOURCE):
  if not p.exists():raise FileNotFoundError(p)
 if any(json.loads(p.read_text())["status"]!="pass" for p in (STATIC,QSMOKE,CSMOKE)):raise RuntimeError("QH040 prerequisites invalid")
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
 tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);complete=len(tokens)//LENGTH;available=np.asarray(sorted(set(range(complete))-used),dtype=np.int64);selected=np.random.default_rng(SEED).permutation(available)[:TRAIN_COUNT+EVAL_COUNT].tolist();train=selected[:TRAIN_COUNT];evaluation=selected[TRAIN_COUNT:]
 if len(selected)!=TRAIN_COUNT+EVAL_COUNT or set(selected)&used:raise AssertionError("QH040 split invalid")
 payload={"status":"locked_before_qh040_train_only_screen_results","candidate":"QH-040 bounded quantum residual on frozen CC037 scaffold","parent":"QH039 revealed alpha-norm overfit","single_mutation":"replace unconstrained alpha by effective alpha=0.05*tanh(raw), preserving parameter count, circuit, objective, steps, and controls","backbone":{"path":str(BACKBONE),"sha256":sha(BACKBONE),"frozen":True},"architecture":{"target_layer":31,"deployed_parameters":655936,"net_model_parameter_reduction":266730944,"trainable":320,"effective_alpha_cap_per_coordinate":0.05,"branch_off":"exact CC037 scaffold"},"controls":{"base":"original SwiGLU","scaffold":"same module correction disabled","equal_parameter_classical":"same bounded alpha and frozen scaffold with 256-parameter classical residual","future_no_ent":"remove controlled-RY only","dequantized":"independent tensor-axis"},"evidence":{n:{"path":str(p),"sha256":sha(p)} for n,p in (("static",STATIC),("quantum_smoke",QSMOKE),("classical_smoke",CSMOKE))},"source":{"path":str(SOURCE),"sha256":sha(SOURCE)},"dataset":"Salesforce/wikitext wikitext-2-raw-v1 train only","tokens_sha256":sha(TOKENS),"sequence_length":LENGTH,"seed":SEED,"train_indices":train,"eval_indices":evaluation,"train_count":TRAIN_COUNT,"eval_count":EVAL_COUNT,"excluded_prior_indices_count":len(used),"available_before_selection":len(available),"prior_json_parsed":parsed,"overlap":0,"training_protocol":{"objective":"mean-token teacher-logit KL","optimizer":"AdamW","learning_rate":0.003,"weight_decay":0.0,"gradient_clip_norm":1.0,"steps":STEPS,"cycles_over_train_indices":2,"all_scaffold_and_base_parameters_frozen":True,"save_checkpoint_optimizer_source_hash":True},"screen_rule":{"continue_to_new_C4_only_if":["QH040 point NLL lower than own scaffold","QH040 point NLL lower than CC040","QH040 no worse than base by 0.005"],"otherwise":"reject or mutate before C4"},"validation_rows_inspected":0,"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()};OUTPUT.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":payload["status"],"excluded":len(used),"available":len(available),"sha256":sha(OUTPUT)},indent=2))
if __name__=="__main__":main()
