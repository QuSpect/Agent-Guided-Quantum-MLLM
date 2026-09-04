#!/usr/bin/env python3
"""Analyze the locked QH039 train-only screen."""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
PROJECT=Path(__file__).resolve().parents[2];OUT=PROJECT/"artifacts/qh039_frozen_scaffold_screen";LOCK=PROJECT/"artifacts/qh039-frozen-scaffold-screen-lock.json"
def load(name):return json.loads(next(OUT.glob(f"screen-{name}-*.json")).read_text())
def ci(a,b,seed):
 d=np.asarray([x["mean_token_nll"] for x in a["evaluation"]["rows"]])-np.asarray([x["mean_token_nll"] for x in b["evaluation"]["rows"]]);rng=np.random.default_rng(seed);means=d[rng.integers(0,len(d),size=(20000,len(d)))].mean(1);return {"mean_delta_nll":float(d.mean()),"descriptive_95_ci":[float(np.quantile(means,.025)),float(np.quantile(means,.975))]}
def main():
 lock=json.loads(LOCK.read_text());d={n:load(n) for n in ("base","scaffold","qh039","cc039")};nll={n:v["evaluation"]["metrics"]["mean_token_nll"] for n,v in d.items()};comparisons={"scaffold_minus_base":ci(d["scaffold"],d["base"],1),"qh_minus_scaffold":ci(d["qh039"],d["scaffold"],2),"qh_minus_cc":ci(d["qh039"],d["cc039"],3),"qh_minus_base":ci(d["qh039"],d["base"],4)};gates={"qh_point_better_than_own_scaffold":nll["qh039"]<nll["scaffold"],"qh_point_better_than_cc":nll["qh039"]<nll["cc039"],"qh_within_base_plus_0p005":nll["qh039"]-nll["base"]<=.005};gates["continue_to_c4"]=all(gates.values());payload={"status":"ok","analysis":"QH039 frozen trained scaffold plus switchable residual train-only screen","verdict":"continue_to_new_c4" if gates["continue_to_c4"] else "reject_or_mutate_before_c4","point_nll":nll,"comparisons":comparisons,"screen_gates":gates,"training":{n:{k:v for k,v in d[n]["training"].items() if k!="losses"} for n in ("qh039","cc039")},"claim_limit":"Train-only screen; no quality or quantum claim.","validation_rows_inspected":0,"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()};path=OUT/"qh039-frozen-scaffold-screen-analysis.json";path.write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
