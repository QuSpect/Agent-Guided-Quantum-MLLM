#!/usr/bin/env python3
"""Pre-registered paired statistics for QH033 C4 evaluation."""
from __future__ import annotations
import hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/"artifacts/c4_qh033";LOCK=ROOT/"artifacts/qh033-c4-lock.json"
FILES={n:RUN/f"c4-{n}-s20260855-e128.json" for n in ("base","sharedsvd1024","qh033","cc033")};OUT=RUN/"c4-qh033-paired-analysis.json"
def sha(path):
 d=hashlib.sha256()
 with path.open("rb") as h:
  for c in iter(lambda:h.read(8<<20),b""):d.update(c)
 return d.hexdigest()
def rows(v,key="evaluation_active"):return {int(r["block_index"]):float(r["mean_token_nll"]) for r in v[key]["rows"]}
def main():
 lock=json.loads(LOCK.read_text());expected=lock["eval_indices"];payloads={};arrays={}
 for n,p in FILES.items():
  v=json.loads(p.read_text());
  if v["lock_sha256"]!=sha(LOCK) or v["training"]["steps"]!=0:raise AssertionError("lock/training violation")
  rm=rows(v)
  if list(rm)!=expected:raise AssertionError("order mismatch")
  payloads[n]=v;arrays[n]=np.asarray([rm[i] for i in expected])
 ne=rows(payloads["qh033"],"evaluation_no_entanglement");arrays["qh033_no_ent"]=np.asarray([ne[i] for i in expected])
 samples=lock["evaluation_protocol"]["paired_bootstrap_samples"];rng=np.random.default_rng(lock["seed"]);ids=rng.integers(0,len(expected),size=(samples,len(expected)))
 srng=np.random.default_rng(lock["seed"]+1);signs=srng.choice(np.asarray([-1.,1.]),size=(samples,len(expected)));comparisons={}
 defs={"shared_minus_base":("sharedsvd1024","base"),"qh_minus_base":("qh033","base"),"cc_minus_base":("cc033","base"),
  "qh_active_minus_own_zero":("qh033","sharedsvd1024"),"qh_minus_cc":("qh033","cc033"),"qh_active_minus_no_ent":("qh033","qh033_no_ent")}
 for label,(l,r) in defs.items():
  d=arrays[l]-arrays[r];boot=d[ids].mean(1);observed=float(d.mean());perm=(d[None,:]*signs).mean(1)
  comparisons[label]={"mean_delta_nll":observed,"paired_bootstrap_95_ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
   "sign_flip_two_sided_p":float((np.count_nonzero(np.abs(perm)>=abs(observed))+1)/(samples+1))}
 def upper(k):return comparisons[k]["paired_bootstrap_95_ci"][1]
 margin=lock["evaluation_protocol"]["compression_noninferiority_margin"];reduction=payloads["qh033"]["injection"]["net_model_parameter_reduction"]
 gates={"compression_noninferior":upper("qh_minus_base")<=margin and reduction>0,"strict_better_base":upper("qh_minus_base")<0,
  "active_better_own_zero":upper("qh_active_minus_own_zero")<0,"quantum_better_cc":upper("qh_minus_cc")<0,
  "entanglement_causal":upper("qh_active_minus_no_ent")<0,"net_reduction_positive":reduction>0}
 gates["quantum_specific_promotion"]=all(gates[k] for k in ("compression_noninferior","active_better_own_zero","quantum_better_cc","entanglement_causal"))
 verdict="promote_qh033_to_external_test" if gates["quantum_specific_promotion"] else "retain_compression_reject_qh033_quantum_specific_claim"
 result={"status":"ok","analysis":"QH033 locked 128-block C4 validation","verdict":verdict,"lock_sha256":sha(LOCK),
  "point_metrics":{n:{"nll":float(a.mean()),"perplexity":math.exp(float(a.mean()))} for n,a in arrays.items()},"comparisons":comparisons,"gates":gates,
  "parameter_audit":{"net_model_parameter_reduction":reduction,"prior_trainable_parameters":82,"evaluation_trainable_parameters":0},
  "claim_limits":["C4 development validation, not final external test","Exact GPU statevector is classically reproducible","All three attribution gates required"],
  "test_rows_inspected_or_used_for_fitness":0,"created_at":datetime.now(timezone.utc).isoformat()}
 OUT.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":main()

