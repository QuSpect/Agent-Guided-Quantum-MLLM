#!/usr/bin/env python3
"""Continue QH037/CC037 from saved optimizer state on fresh WikiText train-only blocks."""
from __future__ import annotations
import argparse,hashlib,json,math,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from transformers import AutoModelForMultimodalLM
PROJECT=Path(__file__).resolve().parents[2];ART=PROJECT/"artifacts";TOKENS=PROJECT/"datasets/processed/wikitext2_qwen38/train-tokens.pt";QSTART=ART/"qh037_grouped_qvaf_screen/qh037-screen-checkpoint.pt";CSTART=ART/"qh037_grouped_qvaf_screen/cc037-screen-checkpoint.pt";SOURCE=PROJECT/"code/src/quantum_qwen38/qh037_grouped_qvaf_ffn_replacement.py";ENTRY=PROJECT/"code/scripts/qh044_extended_distillation.py";LOCK=ART/"qh044-extended-distillation-lock.json";OUT=ART/"qh044_extended_distillation";sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh037_grouped_qvaf_ffn_replacement import ExactGroupedQVAFCore,QH037Config,install_qh037,unique_trainable_parameters
def sha(p):
 d=hashlib.sha256()
 with Path(p).open("rb") as h:
  for c in iter(lambda:h.read(8<<20),b""):d.update(c)
 return d.hexdigest()
def lock():
 for p in (TOKENS,QSTART,CSTART,SOURCE,ENTRY):
  if not p.exists():raise FileNotFoundError(p)
 used=set()
 for path in ART.rglob("*.json"):
  if path==LOCK:continue
  try:text=path.read_text();v=json.loads(text)
  except Exception:continue
  if "wikitext" not in text.lower():continue
  def walk(x):
   if isinstance(x,dict):
    for k,c in x.items():
     if k in {"train_indices","train_block_indices","eval_indices","evaluation_indices","block_indices"} and isinstance(c,list):used.update(int(i) for i in c)
     elif k in {"train_block_index","block_index"} and isinstance(c,int):used.add(c)
     walk(c)
   elif isinstance(x,list):
    for c in x:walk(c)
  walk(v)
 tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);available=np.asarray(sorted(set(range(len(tokens)//256))-used),dtype=np.int64);sel=np.random.default_rng(20260881).permutation(available)[:564].tolist()
 if len(sel)!=564 or set(sel)&used:raise RuntimeError({"available":len(available)})
 payload={"status":"locked_before_qh044_extended_distillation_results","single_mutation":"continue QH037 and CC037 from their registered step-256 parameters and AdamW state for 1000 additional steps; architecture, lr, objective, data length and equal-parameter control unchanged","starting_checkpoints":{"qh037":{"path":str(QSTART),"sha256":sha(QSTART),"steps":256},"cc037":{"path":str(CSTART),"sha256":sha(CSTART),"steps":256}},"architecture":{"target_layer":31,"replacement_parameters":655616,"net_model_parameter_reduction":266731264,"trainable_parameters":655616},"dataset":"Salesforce/wikitext wikitext-2-raw-v1 train only","tokens_sha256":sha(TOKENS),"sequence_length":256,"seed":20260881,"train_indices":sel[:500],"eval_indices":sel[500:],"train_count":500,"eval_count":64,"excluded_prior_wikitext_indices_count":len(used),"available_before_selection":len(available),"overlap":0,"training_protocol":{"objective":"mean-token teacher-logit KL","optimizer":"resume AdamW state from QH037/CC037","learning_rate":.001,"weight_decay":0.,"gradient_clip_norm":1.,"additional_steps":1000,"cycles_over_train_indices":2,"cumulative_steps":1256,"base_parameters_frozen":True},"arms":{"base":"original","zero":"delete layer31 FFN","qh037_start":"fixed step256 QH","cc037_start":"fixed step256 CC","qh044":"QH +1000 equal-budget steps","cc044":"CC +1000 equal-budget steps"},"screen_rule":{"continue_to_new_C4_only_if":["QH044 point NLL lower than delete-zero","QH044 point NLL lower than CC044","QH044 no worse than base by 0.005"]},"forbidden":"do not reuse EXP044 C4-256 to tune this candidate","validation_rows_inspected":0,"test_rows_inspected":0,"source_hashes":{"replacement":sha(SOURCE),"entrypoint":sha(ENTRY)},"created_at":datetime.now(timezone.utc).isoformat()};LOCK.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":payload["status"],"excluded":len(used),"available":len(available),"sha256":sha(LOCK)},indent=2))
def model():
 path=Path((PROJECT/"records/active_model_path.txt").read_text().strip());m=AutoModelForMultimodalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,device_map="balanced",low_cpu_mem_usage=True,attn_implementation="sdpa");m.eval();m.config.use_cache=False;return m,path
def block(tokens,i,device):return tokens[i*256:(i+1)*256].to(device=device,dtype=torch.long).unsqueeze(0)
@torch.inference_mode()
def evaluate(m,tokens,indices):
 rows=[]
 for i in indices:
  x=block(tokens,i,m.device);started=time.perf_counter();loss=m(input_ids=x,labels=x,use_cache=False).loss;torch.cuda.synchronize();rows.append({"block_index":i,"mean_token_nll":float(loss),"seconds":time.perf_counter()-started})
 nll=float(np.mean([r["mean_token_nll"] for r in rows]));return {"metrics":{"mean_token_nll":nll,"perplexity":math.exp(nll),"mean_seconds":float(np.mean([r["seconds"] for r in rows]))},"rows":rows}
def install(m,branch,checkpoint):
 rep,original,audit=install_qh037(m,QH037Config(),branch);state=torch.load(checkpoint,map_location="cpu",weights_only=False);rep.load_state_dict(state["replacement_state_dict"],strict=True);return rep,original,audit,state
def train(m,rep,original,tokens,lockv,opt):
 params=list(unique_trainable_parameters(rep));losses=[];seconds=[];before=int(rep.core.circuit_calls.item());cfg=QH037Config()
 for pos in range(1,1001):
  i=lockv["train_indices"][(pos-1)%500];x=block(tokens,i,m.device);m.model.language_model.layers[cfg.target_layer].mlp=original
  with torch.inference_mode():teacher=m(input_ids=x,use_cache=False).logits[:,:-1].detach()
  m.model.language_model.layers[cfg.target_layer].mlp=rep;opt.zero_grad(set_to_none=True);started=time.perf_counter();student=m(input_ids=x,use_cache=False).logits[:,:-1];loss=F.kl_div(F.log_softmax(student.float(),-1),F.softmax(teacher.float(),-1),reduction="sum")/(student.shape[0]*student.shape[1]);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step();torch.cuda.synchronize();losses.append(float(loss));seconds.append(time.perf_counter()-started)
  if pos%100==0:print(json.dumps({"trained_additional":pos,"recent_kl":float(np.mean(losses[-100:])),"recent_seconds":float(np.mean(seconds[-100:]))}),flush=True)
 calls=int(rep.core.circuit_calls.item())-before;expected=1000 if isinstance(rep.core,ExactGroupedQVAFCore) else 0;frozen=sum(int(p.grad is not None) for p in m.parameters() if not p.requires_grad)
 if calls!=expected or frozen:raise AssertionError({"calls":calls,"expected":expected,"frozen":frozen})
 return {"additional_steps":1000,"cumulative_steps":1256,"kl_first50":float(np.mean(losses[:50])),"kl_last50":float(np.mean(losses[-50:])),"kl_min":float(np.min(losses)),"mean_student_step_seconds":float(np.mean(seconds)),"simulator_calls":calls,"frozen_gradient_tensor_count":frozen},opt
def run(candidate):
 lockv=json.loads(LOCK.read_text());random.seed(lockv["seed"]);np.random.seed(lockv["seed"]);torch.manual_seed(lockv["seed"]);torch.cuda.manual_seed_all(lockv["seed"]);tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);m,model_dir=model();rep=original=opt=None;tr={"additional_steps":0}
 if candidate=="base":m.requires_grad_(False);audit={"candidate":"base","net_model_parameter_reduction":0}
 elif candidate=="zero":rep,original,audit,_=install(m,"classical",CSTART);rep.branch_enabled=False;rep.requires_grad_(False)
 else:
  quantum=candidate in {"qh037_start","qh044"};start=QSTART if quantum else CSTART;rep,original,audit,state=install(m,"quantum" if quantum else "classical",start)
  if candidate.endswith("_start"):rep.requires_grad_(False)
  else:
   opt=torch.optim.AdamW(list(unique_trainable_parameters(rep)),lr=.001,weight_decay=0.);opt.load_state_dict(state["optimizer_state_dict"]);tr,opt=train(m,rep,original,tokens,lockv,opt)
 ev=evaluate(m,tokens,lockv["eval_indices"]);OUT.mkdir(parents=True,exist_ok=True)
 if candidate in {"qh044","cc044"}:torch.save({"candidate":candidate,"config":QH037Config().__dict__,"data_lock_sha256":sha(LOCK),"source_sha256":sha(SOURCE),"entrypoint_sha256":sha(ENTRY),"starting_checkpoint_sha256":sha(QSTART if candidate=="qh044" else CSTART),"base_model_path":str(model_dir),"base_config_sha256":sha(model_dir/"config.json"),"steps":1256,"additional_steps":1000,"precision":{"base":"bfloat16","simulator":"float32 exact statevector","device":"CUDA GPU only"},"replacement_state_dict":{k:v.detach().cpu() for k,v in rep.state_dict().items()},"optimizer":"AdamW resumed","optimizer_state_dict":opt.state_dict(),"scheduler_state_dict":None,"training_summary":tr},OUT/f"{candidate}-checkpoint.pt")
 payload={"status":"ok","candidate":candidate,"lock_sha256":sha(LOCK),"training":tr,"audit":audit,"evaluation":ev,"validation_rows_inspected":0,"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()};path=OUT/f"screen-{candidate}-s{lockv['seed']}-n500-e64.json";path.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":"ok","candidate":candidate,"nll":ev["metrics"]["mean_token_nll"],"sha256":sha(path)},indent=2))
def analyze():
 def load(n):return json.loads(next(OUT.glob(f"screen-{n}-*.json")).read_text())
 def ci(a,b,s):
  d=np.asarray([x["mean_token_nll"] for x in a["evaluation"]["rows"]])-np.asarray([x["mean_token_nll"] for x in b["evaluation"]["rows"]]);r=np.random.default_rng(s);m=d[r.integers(0,len(d),size=(20000,len(d)))].mean(1);return {"mean_delta_nll":float(d.mean()),"descriptive_95_ci":[float(np.quantile(m,.025)),float(np.quantile(m,.975))]}
 names=("base","zero","qh037_start","cc037_start","qh044","cc044");d={n:load(n) for n in names};nll={n:v["evaluation"]["metrics"]["mean_token_nll"] for n,v in d.items()};cmp={"qh044_minus_zero":ci(d["qh044"],d["zero"],1),"qh044_minus_cc044":ci(d["qh044"],d["cc044"],2),"qh044_minus_base":ci(d["qh044"],d["base"],3),"qh044_minus_qh037_start":ci(d["qh044"],d["qh037_start"],4),"cc044_minus_cc037_start":ci(d["cc044"],d["cc037_start"],5)};g={"qh_point_better_than_zero":nll["qh044"]<nll["zero"],"qh_point_better_than_cc":nll["qh044"]<nll["cc044"],"qh_within_base_plus_0p005":nll["qh044"]-nll["base"]<=.005};g["continue_to_c4"]=all(g.values());p={"status":"ok","analysis":"QH044 extended equal-budget distillation train-only screen","point_nll":nll,"comparisons":cmp,"gates":g,"verdict":"continue_to_new_c4" if g["continue_to_c4"] else "reject_or_mutate_before_c4","claim_limit":"train-only","created_at":datetime.now(timezone.utc).isoformat()};(OUT/"qh044-extended-analysis.json").write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument("mode",choices=["lock","run","analyze"]);p.add_argument("--candidate",choices=["base","zero","qh037_start","cc037_start","qh044","cc044"]);a=p.parse_args();{"lock":lambda:lock(),"run":lambda:run(a.candidate),"analyze":lambda:analyze()}[a.mode]()
if __name__=="__main__":main()
