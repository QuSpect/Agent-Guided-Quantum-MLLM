#!/usr/bin/env python3
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
PROJECT=Path(__file__).resolve().parents[2];OUT=PROJECT/"artifacts/qh042_xz_screen"
def load(n):return json.loads(next(OUT.glob(f"screen-{n}-*.json")).read_text())
def ci(a,b,s):
 d=np.asarray([x["mean_token_nll"] for x in a["evaluation"]["rows"]])-np.asarray([x["mean_token_nll"] for x in b["evaluation"]["rows"]]);r=np.random.default_rng(s);m=d[r.integers(0,len(d),size=(20000,len(d)))].mean(1);return {"mean_delta_nll":float(d.mean()),"descriptive_95_ci":[float(np.quantile(m,.025)),float(np.quantile(m,.975))]}
def main():
 d={n:load(n) for n in ("base","scaffold","qh042","cc042")};nll={n:v["evaluation"]["metrics"]["mean_token_nll"] for n,v in d.items()};cmp={"scaffold_minus_base":ci(d["scaffold"],d["base"],1),"qh_minus_scaffold":ci(d["qh042"],d["scaffold"],2),"qh_minus_cc":ci(d["qh042"],d["cc042"],3),"qh_minus_base":ci(d["qh042"],d["base"],4)};g={"qh_point_better_than_own_scaffold":nll["qh042"]<nll["scaffold"],"qh_point_better_than_cc":nll["qh042"]<nll["cc042"],"qh_within_base_plus_0p005":nll["qh042"]-nll["base"]<=.005};g["continue_to_c4"]=all(g.values());p={"status":"ok","analysis":"QH042 signed-XZ bounded residual train-only screen","verdict":"continue_to_new_c4" if g["continue_to_c4"] else "reject_or_mutate_before_c4","point_nll":nll,"comparisons":cmp,"screen_gates":g,"training":{n:{k:v for k,v in d[n]["training"].items() if k!="losses"} for n in ("qh042","cc042")},"claim_limit":"Train-only screen","validation_rows_inspected":0,"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()};path=OUT/"qh042-xz-screen-analysis.json";path.write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))
if __name__=="__main__":main()
