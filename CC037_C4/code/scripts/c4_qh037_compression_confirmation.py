#!/usr/bin/env python3
"""Fresh C4 confirmation of the fixed CC037 full-FFN compression scaffold."""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForMultimodalLM
PROJECT=Path(__file__).resolve().parents[2];ART=PROJECT/"artifacts";TOKENS=PROJECT/"datasets/processed/c4_validation_qwen38/validation-tokens.pt";CHECKPOINT=ART/"qh037_grouped_qvaf_screen/cc037-screen-checkpoint.pt";SOURCE=PROJECT/"code/src/quantum_qwen38/qh037_grouped_qvaf_ffn_replacement.py";ENTRY=PROJECT/"code/scripts/c4_qh037_compression_confirmation.py";LOCK=ART/"qh037-compression-c4-confirmation-lock.json";OUT=ART/"c4_qh037_compression_confirmation";sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh037_grouped_qvaf_ffn_replacement import QH037Config,install_qh037
def sha(p):
 d=hashlib.sha256()
 with Path(p).open("rb") as h:
  for c in iter(lambda:h.read(8<<20),b""):d.update(c)
 return d.hexdigest()
def lock():
 for p in (TOKENS,CHECKPOINT,SOURCE,ENTRY):
  if not p.exists():raise FileNotFoundError(p)
 used=set();sources=[]
 for path in ART.rglob("*.json"):
  if path==LOCK:continue
  try:text=path.read_text();v=json.loads(text)
  except Exception:continue
  if "c4" not in str(path).lower() and "c4" not in text.lower():continue
  sources.append(str(path))
  def walk(x):
   if isinstance(x,dict):
    for k,c in x.items():
     if k in {"eval_indices","evaluation_indices","block_indices"} and isinstance(c,list):used.update(int(i) for i in c)
     elif k=="block_index" and isinstance(c,int):used.add(c)
     walk(c)
   elif isinstance(x,list):
    for c in x:walk(c)
  walk(v)
 tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);complete=len(tokens)//256;available=np.asarray(sorted(set(range(complete))-used),dtype=np.int64);selected=np.random.default_rng(20260879).permutation(available)[:256].tolist()
 if len(selected)!=256 or set(selected)&used:raise AssertionError("C4 confirmation overlap")
 payload={"status":"locked_before_qh037_compression_confirmation_results","claim":"compression-only; no quantum advantage claim","candidate":"fixed CC037 grouped rank-64 full-FFN replacement","checkpoint":{"path":str(CHECKPOINT),"sha256":sha(CHECKPOINT),"training_steps":256,"updated_during_confirmation":False},"architecture":{"target_layer":31,"original_ffn_parameters_removed":267386880,"replacement_parameters":655616,"net_model_parameter_reduction":266731264,"fraction_of_27b_base":266731264/27356728560},"arms":{"base":"original Qwen3.8-27B","delete_zero":"layer31 FFN output zero","cc037_scaffold":"fixed registered checkpoint; no training"},"dataset":"allenai/c4 en validation processed fixed shard","tokens_sha256":sha(TOKENS),"sequence_length":256,"seed":20260879,"evaluation_indices":selected,"evaluation_count":256,"excluded_prior_c4_indices_count":len(used),"available_before_selection":len(available),"overlap":0,"source_hashes":{"replacement":sha(SOURCE),"entrypoint":sha(ENTRY)},"pre_result_failure":{"event":"first base attempt reached first block but CUDA loss rejected Int32 labels before returning any loss","fix":"cast input_ids and labels to torch.long only","prior_lock_sha256":"6aba101d56cbc9a0c5a5fe4be18f2808c54ad8711b36af10ed5d68b14ee55cbe","scores_produced":0,"selection_preserved_by_same_seed_and_excluding_current_lock_from_used-set_scan":True},"pre_registered_gates":{"compression_noninferiority":"upper paired bootstrap 95% CI of scaffold-base < +0.005 NLL","restoration":"upper paired bootstrap 95% CI of scaffold-delete_zero < 0","strict_quality_improvement_exploratory":"upper paired bootstrap 95% CI of scaffold-base < 0"},"training_steps_on_confirmation":0,"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()};LOCK.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":payload["status"],"excluded":len(used),"available":len(available),"sha256":sha(LOCK)},indent=2))
def load_model():
 model_dir=Path((PROJECT/"records/active_model_path.txt").read_text().strip());model=AutoModelForMultimodalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,device_map="balanced",low_cpu_mem_usage=True,attn_implementation="sdpa");model.eval();model.config.use_cache=False;model.requires_grad_(False);return model,model_dir
@torch.inference_mode()
def run(arm):
 lockv=json.loads(LOCK.read_text());tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);model,model_dir=load_model();audit={"arm":arm,"net_model_parameter_reduction":0};cfg=QH037Config()
 if arm!="base":
  rep,_,audit=install_qh037(model,cfg,"classical");state=torch.load(CHECKPOINT,map_location="cpu",weights_only=False);rep.load_state_dict(state["replacement_state_dict"],strict=True);rep.requires_grad_(False);rep.branch_enabled=arm=="cc037_scaffold"
 rows=[]
 for i in lockv["evaluation_indices"]:
  x=tokens[i*256:(i+1)*256].to(device=model.device,dtype=torch.long).unsqueeze(0);started=time.perf_counter();loss=model(input_ids=x,labels=x,use_cache=False).loss;torch.cuda.synchronize();rows.append({"block_index":i,"mean_token_nll":float(loss),"seconds":time.perf_counter()-started})
 nll=float(np.mean([r["mean_token_nll"] for r in rows]));payload={"status":"ok","arm":arm,"lock_sha256":sha(LOCK),"checkpoint_sha256":sha(CHECKPOINT) if arm!="base" else None,"training_steps":0,"audit":audit,"evaluation":{"metrics":{"mean_token_nll":nll,"perplexity":math.exp(nll),"mean_seconds":float(np.mean([r["seconds"] for r in rows]))},"rows":rows},"base_model_path":str(model_dir),"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()};OUT.mkdir(parents=True,exist_ok=True);path=OUT/f"confirmation-{arm}-s{lockv['seed']}-e256.json";path.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":"ok","arm":arm,"nll":nll,"sha256":sha(path)},indent=2))
def analyze():
 def load(n):return json.loads(next(OUT.glob(f"confirmation-{n}-*.json")).read_text())
 def ci(a,b,s):
  d=np.asarray([x["mean_token_nll"] for x in a["evaluation"]["rows"]])-np.asarray([x["mean_token_nll"] for x in b["evaluation"]["rows"]]);r=np.random.default_rng(s);m=d[r.integers(0,len(d),size=(50000,len(d)))].mean(1);return {"mean_delta_nll":float(d.mean()),"paired_bootstrap_95_ci":[float(np.quantile(m,.025)),float(np.quantile(m,.975))]}
 d={n:load(n) for n in ("base","delete_zero","cc037_scaffold")};nll={n:v["evaluation"]["metrics"]["mean_token_nll"] for n,v in d.items()};sb=ci(d["cc037_scaffold"],d["base"],1);sz=ci(d["cc037_scaffold"],d["delete_zero"],2);g={"compression_noninferiority":sb["paired_bootstrap_95_ci"][1]<.005,"restoration_over_delete_zero":sz["paired_bootstrap_95_ci"][1]<0,"strict_quality_improvement_exploratory":sb["paired_bootstrap_95_ci"][1]<0};p={"status":"ok","analysis":"QH037 fixed scaffold C4-256 compression confirmation","claim":"compression only, not quantum","point_nll":nll,"comparisons":{"scaffold_minus_base":sb,"scaffold_minus_delete_zero":sz},"gates":g,"verdict":"confirm_compression" if g["compression_noninferiority"] and g["restoration_over_delete_zero"] else "compression_not_confirmed","training_steps":0,"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()};path=OUT/"qh037-compression-confirmation-analysis.json";path.write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument("mode",choices=["lock","run","analyze"]);p.add_argument("--arm",choices=["base","delete_zero","cc037_scaffold"]);a=p.parse_args();{"lock":lambda:lock(),"run":lambda:run(a.arm),"analyze":lambda:analyze()}[a.mode]()
if __name__=="__main__":main()
