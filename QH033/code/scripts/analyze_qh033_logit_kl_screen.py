#!/usr/bin/env python3
"""Analyze QH033 train-only output-KL screen."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/"artifacts/qh033_logit_kl_screen"; LOCK=ROOT/"artifacts/qh033-logit-kl-screen-lock.json"
FILES={n:RUN/f"screen-{n}-s20260853-n256-e64.json" for n in ("base","sharedsvd1024","qh033","cc033")}; OUT=RUN/"qh033-logit-kl-screen-analysis.json"
def sha(path):
 d=hashlib.sha256()
 with path.open("rb") as h:
  for c in iter(lambda:h.read(8<<20),b""):d.update(c)
 return d.hexdigest()
def main():
 lock=json.loads(LOCK.read_text()); expected=lock["eval_indices"]; payloads={}; arrays={}
 for n,p in FILES.items():
  v=json.loads(p.read_text());
  if v["lock_sha256"]!=sha(LOCK):raise AssertionError("lock mismatch")
  rows=v["evaluation"]["rows"]
  if [r["block_index"] for r in rows]!=expected:raise AssertionError("order mismatch")
  payloads[n]=v; arrays[n]=np.asarray([r["mean_token_nll"] for r in rows])
 rng=np.random.default_rng(lock["seed"]); ids=rng.integers(0,len(expected),size=(20000,len(expected))); comparisons={}
 for label,l,r in (("shared_minus_base","sharedsvd1024","base"),("qh_minus_shared","qh033","sharedsvd1024"),("qh_minus_cc","qh033","cc033")):
  d=arrays[l]-arrays[r]; b=d[ids].mean(1); comparisons[label]={"mean_delta_nll":float(d.mean()),"descriptive_95_ci":[float(np.quantile(b,.025)),float(np.quantile(b,.975))]}
 gates={"qh_point_better_than_own_zero":comparisons["qh_minus_shared"]["mean_delta_nll"]<0,"qh_point_better_than_cc":comparisons["qh_minus_cc"]["mean_delta_nll"]<0}; gates["continue_to_c4"]=all(gates.values())
 result={"status":"ok","analysis":"QH033 train-only teacher-logit KL screen","verdict":"continue_to_new_c4_lock" if gates["continue_to_c4"] else "reject_pauli_observable_family",
   "point_nll":{n:float(a.mean()) for n,a in arrays.items()},"comparisons":comparisons,"screen_gates":gates,
   "training":{n:{k:v for k,v in payloads[n]["training"].items() if k!="losses"} for n in ("qh033","cc033")},
   "claim_limit":"Train-only screen; no quality or quantum claim.","validation_rows_inspected":0,"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()}
 OUT.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":main()

