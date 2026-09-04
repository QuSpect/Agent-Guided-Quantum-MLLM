#!/usr/bin/env python3
"""Lock fresh train-only blocks for QH035 shared-RMSNorm screening."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
PROJECT=Path(__file__).resolve().parents[2]; ART=PROJECT/"artifacts"; TOKENS=PROJECT/"datasets/processed/wikitext2_qwen38/train-tokens.pt"
AUDIT=ART/"qh035-rmsnorm-sharing-audit.json"; STATIC=ART/"qh035-static-validation.json"; SMOKE=ART/"qh035-full-model-smoke.json"
SOURCE=PROJECT/"code/src/quantum_qwen38/qh035_shared_rmsnorm_replacement.py"; OUTPUT=ART/"qh035-norm-screen-lock.json"
SEED=20260862; LENGTH=256; TRAIN_COUNT=256; EVAL_COUNT=64
def sha(path):
 d=hashlib.sha256()
 with path.open("rb") as h:
  for c in iter(lambda:h.read(8<<20),b""):d.update(c)
 return d.hexdigest()
def used_indices():
 used=set(); parsed=0
 for path in ART.rglob("*.json"):
  if path==OUTPUT:continue
  try:v=json.loads(path.read_text())
  except Exception:continue
  parsed+=1
  def walk(x):
   if isinstance(x,dict):
    for k,c in x.items():
     if k in {"train_indices","train_block_indices"} and isinstance(c,list):used.update(int(i) for i in c)
     elif k=="train_block_index" and isinstance(c,int):used.add(c)
     walk(c)
   elif isinstance(x,list):
    for c in x:walk(c)
  walk(v)
 return used,parsed
def main():
 for p in (TOKENS,AUDIT,STATIC,SMOKE,SOURCE):
  if not p.exists():raise FileNotFoundError(p)
 audit=json.loads(AUDIT.read_text()); static=json.loads(STATIC.read_text()); smoke=json.loads(SMOKE.read_text())
 if audit["status"]!="complete_weight_only_no_dataset" or static["status"]!="pass" or smoke["status"]!="pass":raise RuntimeError("QH035 gates invalid")
 chosen=audit["best_same_family_clusters"]["post_attention_layernorm"]["4"]
 if chosen["layers"]!=[11,12,15,13]:raise RuntimeError("weight-only selected cluster changed")
 tokens=torch.load(TOKENS,map_location="cpu",weights_only=True); complete=len(tokens)//LENGTH; used,parsed=used_indices(); available=np.asarray(sorted(set(range(complete))-used),dtype=np.int64)
 selected=np.random.default_rng(SEED).permutation(available)[:TRAIN_COUNT+EVAL_COUNT].tolist(); train=selected[:TRAIN_COUNT]; evaluation=selected[TRAIN_COUNT:]
 if len(selected)!=TRAIN_COUNT+EVAL_COUNT or set(train)&set(evaluation) or set(selected)&used:raise AssertionError("QH035 split invalid")
 payload={"status":"locked_before_qh035_train_only_screen_results","candidate":"QH-035 shared RMSNorm plus runtime 13q token residual",
  "selection":{"method":"weight-only effective-scale cluster","family":"post_attention_layernorm","layers":[11,12,13,15],
    "rms_relative_shared_scale_error":chosen["rms_relative_weight_error"],"max_relative_shared_scale_error":chosen["max_relative_weight_error"]},
  "architecture":{"hidden_size":5120,"padded_state_dimension":8192,"qubits":13,"depth":2,"offsets":[1,2,4],
    "circuit_parameters":104,"layer_gammas":4,"trainable":108,"net_model_parameter_reduction":15252,"zero_angles":"exact shared classical RMSNorm"},
  "controls":{"base":"original four independent RMSNorms","own_zero":"one frozen mean effective scale shared by four layers",
    "equal_parameter_classical":"104-parameter sign/XOR token transform plus four gammas","no_ent":"controlled rotations removed","dequantized":"independent tensor-axis gates"},
  "evidence":{n:{"path":str(p),"sha256":sha(p)} for n,p in (("weight_audit",AUDIT),("static",STATIC),("full_model",SMOKE))},
  "source":{"path":str(SOURCE),"sha256":sha(SOURCE)},"dataset":"Salesforce/wikitext wikitext-2-raw-v1 train only","tokens_sha256":sha(TOKENS),
  "sequence_length":LENGTH,"seed":SEED,"train_indices":train,"eval_indices":evaluation,"train_count":TRAIN_COUNT,"eval_count":EVAL_COUNT,
  "excluded_prior_train_indices_count":len(used),"available_before_selection":len(available),"prior_json_parsed":parsed,"overlap":0,
  "training_protocol":{"objective":"mean-token KL(teacher original norms || compressed student) at temperature 1","optimizer":"AdamW",
    "learning_rate":0.001,"weight_decay":0.0,"gradient_clip_norm":1.0,"steps":TRAIN_COUNT,"all_other_parameters_frozen":True},
  "screen_rule":{"continue_to_new_C4_only_if":["QH035 point NLL lower than shared Norm own-zero","QH035 point NLL lower than equal-parameter CC035"],
    "otherwise":"reject or structurally mutate before C4","no_claim_from_train_screen":True},"validation_rows_inspected":0,"test_rows_inspected":0,
  "created_at":datetime.now(timezone.utc).isoformat()}
 OUTPUT.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":payload["status"],"train_count":TRAIN_COUNT,"eval_count":EVAL_COUNT,
  "excluded":len(used),"available":len(available),"sha256":sha(OUTPUT)},indent=2))
if __name__=="__main__":main()
