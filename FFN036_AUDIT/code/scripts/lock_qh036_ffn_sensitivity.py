#!/usr/bin/env python3
"""Lock fresh WikiText-2 train-only blocks for QH036 FFN deletion sensitivity."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
PROJECT=Path(__file__).resolve().parents[2];ART=PROJECT/"artifacts";TOKENS=PROJECT/"datasets/processed/wikitext2_qwen38/train-tokens.pt"
SOURCE=PROJECT/"code/scripts/audit_qh036_ffn_sensitivity.py";OUTPUT=ART/"qh036-ffn-sensitivity-lock.json"
SEED=20260863;LENGTH=256;COUNT=12
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
     if k in {"train_indices","train_block_indices","eval_indices","evaluation_indices"} and isinstance(c,list):used.update(int(i) for i in c)
     elif k in {"train_block_index","block_index"} and isinstance(c,int):used.add(c)
     walk(c)
   elif isinstance(x,list):
    for c in x:walk(c)
  walk(v)
 return used,parsed
def main():
 for p in (TOKENS,SOURCE):
  if not p.exists():raise FileNotFoundError(p)
 tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);complete=len(tokens)//LENGTH;used,parsed=used_indices();available=np.asarray(sorted(set(range(complete))-used),dtype=np.int64)
 selected=np.random.default_rng(SEED).permutation(available)[:COUNT].tolist()
 if len(selected)!=COUNT or set(selected)&used:raise AssertionError("QH036 sensitivity split invalid")
 payload={"status":"locked_before_qh036_ffn_sensitivity_results","candidate":"QH-036 activation-aware FFN replacement feasibility",
  "stage":"train-only destructive ablation; no optimization; no validation/test","dataset":"Salesforce/wikitext wikitext-2-raw-v1 train only",
  "tokens_sha256":sha(TOKENS),"sequence_length":LENGTH,"seed":SEED,"block_indices":selected,"count":COUNT,
  "excluded_prior_indices_count":len(used),"available_before_selection":len(available),"prior_json_parsed":parsed,"overlap":0,
  "source":{"path":str(SOURCE),"sha256":sha(SOURCE)},
  "screen_protocol":{"arms":["base","zero one complete FFN at each of 64 layers"],"metric":"paired token NLL delta",
   "selection":"rank layers by smallest mean deletion delta; this only chooses candidates for a later independent activation-recoverability audit",
   "claim_limit":"No quantum or quality claim from 12 train blocks"},
  "next_gate":"Only top deletion-tolerant layers may enter closed-form affine/reduced-rank activation distillation on a new train-only split",
  "validation_rows_inspected":0,"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()}
 OUTPUT.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":payload["status"],"count":COUNT,"excluded":len(used),"available":len(available),"sha256":sha(OUTPUT)},indent=2))
if __name__=="__main__":main()
