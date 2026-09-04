#!/usr/bin/env python3
"""Static, smoke, lock, screen and analysis workflow for QH043."""
from __future__ import annotations
import argparse,hashlib,json,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from transformers import AutoModelForMultimodalLM
import run_qh039_frozen_scaffold_screen as common
PROJECT=Path(__file__).resolve().parents[2];ART=PROJECT/"artifacts";TOKENS=PROJECT/"datasets/processed/wikitext2_qwen38/train-tokens.pt";BACKBONE=ART/"qh037_grouped_qvaf_screen/cc037-screen-checkpoint.pt";SOURCE=PROJECT/"code/src/quantum_qwen38/qh043_harmonic_x_residual.py";ENTRY=PROJECT/"code/scripts/qh043_harmonic_workflow.py";LOCK=ART/"qh043-harmonic-screen-lock.json";OUT=ART/"qh043_harmonic_screen";sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh043_harmonic_x_residual import *
def sha(p):
 d=hashlib.sha256()
 with Path(p).open("rb") as h:
  for c in iter(lambda:h.read(8<<20),b""):d.update(c)
 return d.hexdigest()
def static():
 d=torch.device("cuda:0");cfg=QH043Config(hidden_size=16,intermediate_size=24,target_layer=0,latent_size=8,group_size=4,num_qubits=4,depth=2);q=ExactHarmonicXCore(cfg).to(d);dq=DequantizedHarmonicXCore(cfg).to(d);dq.load_state_dict(q.state_dict());torch.manual_seed(43)
 with torch.no_grad():q.theta.normal_(0,.2);dq.theta.copy_(q.theta)
 x=torch.randn(3,5,8,device=d,requires_grad=True);x2=x.detach().clone().requires_grad_(True);y=q(x);y2=dq(x2);p=torch.randn_like(y);(y*p).sum().backward();(y2*p).sum().backward();diff={"forward":float((y-y2).abs().max()),"input_grad":float((x.grad-x2.grad).abs().max()),"theta_grad":float((q.theta.grad-dq.theta.grad).abs().max())};cpu=False
 try:q(torch.randn(2,8))
 except RuntimeError:cpu=True
 payload={"status":"pass","q_dq_max_abs":diff,"theta_nonzero":int((q.theta.grad.abs()>0).sum()),"cpu_rejected":cpu,"parameter_audit":qh043_parameter_audit(cfg),"created_at":datetime.now(timezone.utc).isoformat()}
 if max(diff.values())>2e-5 or payload["theta_nonzero"]!=q.theta.numel() or not cpu:raise AssertionError(payload)
 (ART/"qh043-static-validation.json").write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))
def load_model():
 model_dir=Path((PROJECT/"records/active_model_path.txt").read_text().strip());model=AutoModelForMultimodalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,device_map="balanced",low_cpu_mem_usage=True,attn_implementation="sdpa");model.eval();model.config.use_cache=False;return model,model_dir
def smoke(branch):
 torch.manual_seed(20260876);torch.cuda.manual_seed_all(20260876);tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);model,_=load_model();cfg=QH043Config();rep,original,audit=install_qh043(model,BACKBONE,cfg,branch);params=list(unique_trainable_parameters(rep));opt=torch.optim.AdamW(params,lr=3e-3);rows=[];before=int(rep.core.circuit_calls.item())
 for step in range(2):
  x=tokens[(7012+step)*64:(7013+step)*64].to(model.device).unsqueeze(0);model.model.language_model.layers[cfg.target_layer].mlp=original
  with torch.inference_mode():teacher=model(input_ids=x,use_cache=False).logits[:,:-1].detach()
  model.model.language_model.layers[cfg.target_layer].mlp=rep;opt.zero_grad(set_to_none=True);started=time.perf_counter();student=model(input_ids=x,use_cache=False).logits[:,:-1];loss=F.kl_div(F.log_softmax(student.float(),-1),F.softmax(teacher.float(),-1),reduction="sum")/(student.shape[0]*student.shape[1]);loss.backward();row={"step":step,"kl":float(loss),"theta_nonzero":int((rep.core.theta.grad.abs()>0).sum()),"alpha_nonzero":int((rep.alpha.grad.abs()>0).sum()),"frozen_gradient_tensor_count":sum(int(v.grad is not None) for v in model.parameters() if not v.requires_grad)};torch.nn.utils.clip_grad_norm_(params,1);opt.step();torch.cuda.synchronize();row["seconds"]=time.perf_counter()-started;rows.append(row)
 calls=int(rep.core.circuit_calls.item())-before;expected=2 if branch=="quantum" else 0
 if calls!=expected or rows[-1]["theta_nonzero"]!=rep.core.theta.numel() or rows[-1]["alpha_nonzero"]!=rep.alpha.numel() or any(r["frozen_gradient_tensor_count"] for r in rows):raise AssertionError(rows)
 payload={"status":"pass","branch":branch,"audit":audit,"rows":rows,"simulator_calls":calls,"peak_vram_bytes":[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())],"created_at":datetime.now(timezone.utc).isoformat()};(ART/f"qh043-full-model-smoke-{branch}.json").write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))
def lock():
 prereq=[TOKENS,BACKBONE,SOURCE,ART/"qh043-static-validation.json",ART/"qh043-full-model-smoke-quantum.json",ART/"qh043-full-model-smoke-classical.json"]
 for p in prereq:
  if not p.exists():raise FileNotFoundError(p)
 used=set()
 for path in ART.rglob("*.json"):
  if path==LOCK:continue
  try:v=json.loads(path.read_text())
  except Exception:continue
  def walk(x):
   if isinstance(x,dict):
    for k,c in x.items():
     if k in {"train_indices","train_block_indices","eval_indices","evaluation_indices","block_indices"} and isinstance(c,list):used.update(int(i) for i in c)
     elif k in {"train_block_index","block_index"} and isinstance(c,int):used.add(c)
     walk(c)
   elif isinstance(x,list):
    for c in x:walk(c)
  walk(v)
 tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);available=np.asarray(sorted(set(range(len(tokens)//256))-used),dtype=np.int64);sel=np.random.default_rng(20260877).permutation(available)[:320].tolist()
 if len(sel)!=320:raise RuntimeError("insufficient fresh blocks")
 payload={"status":"locked_before_qh043_train_only_screen_results","candidate":"QH-043 harmonic signed-X residual","parent":"QH041 local-X residual","single_mutation":"multiply the second data re-upload angle by 2 to expose a fixed second harmonic; retain scaffold, readout, 320 parameters, cap, objective, optimizer, steps, and CC","backbone":{"path":str(BACKBONE),"sha256":sha(BACKBONE),"frozen":True},"architecture":{"target_layer":31,"deployed_parameters":655936,"net_model_parameter_reduction":266730944,"trainable":320,"observable":"local X","encoding_harmonics":[1,2],"alpha_cap":.05},"source":{"path":str(SOURCE),"sha256":sha(SOURCE)},"evidence":{p.stem:{"path":str(p),"sha256":sha(p)} for p in prereq[3:]},"dataset":"Salesforce/wikitext wikitext-2-raw-v1 train only","tokens_sha256":sha(TOKENS),"sequence_length":256,"seed":20260877,"train_indices":sel[:256],"eval_indices":sel[256:],"train_count":256,"eval_count":64,"excluded_prior_indices_count":len(used),"available_before_selection":len(available),"overlap":0,"training_protocol":{"objective":"mean-token teacher-logit KL","optimizer":"AdamW","learning_rate":.003,"weight_decay":0.,"gradient_clip_norm":1.,"steps":512,"cycles_over_train_indices":2,"all_scaffold_and_base_parameters_frozen":True,"save_checkpoint_optimizer_source_hash":True},"screen_rule":{"continue_to_new_C4_only_if":["QH043 point NLL lower than scaffold","QH043 point NLL lower than CC043","QH043 no worse than base by 0.005"]},"literature_basis":["arXiv:2008.08605","Communications Physics s42005-026-02680-x","arXiv:2606.11673"],"validation_rows_inspected":0,"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()};LOCK.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":payload["status"],"excluded":len(used),"available":len(available),"sha256":sha(LOCK)},indent=2))
def screen(candidate):
 lock=json.loads(LOCK.read_text());random.seed(lock["seed"]);np.random.seed(lock["seed"]);torch.manual_seed(lock["seed"]);torch.cuda.manual_seed_all(lock["seed"]);tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);model,model_dir=load_model();cfg=QH043Config();rep=original=opt=None
 if candidate=="base":model.requires_grad_(False);audit={"candidate":"base","net_model_parameter_reduction":0};tr={"steps":0}
 else:
  branch="classical" if candidate=="cc043" else "quantum";rep,original,audit=install_qh043(model,BACKBONE,cfg,branch)
  if candidate=="scaffold":rep.correction_enabled=False;rep.requires_grad_(False);tr={"steps":0}
  else:tr,opt=common.train(model,rep,original,tokens,lock,cfg);eff=rep.effective_alpha().detach().float();tr["final_effective_alpha_norm"]=float(eff.norm());tr["final_max_abs_effective_alpha"]=float(eff.abs().max())
 ev=common.evaluate(model,tokens,lock["eval_indices"],256);OUT.mkdir(parents=True,exist_ok=True)
 if candidate in {"qh043","cc043"}:torch.save({"candidate":candidate,"config":cfg.__dict__,"data_lock_sha256":sha(LOCK),"source_sha256":sha(SOURCE),"entrypoint_sha256":sha(ENTRY),"backbone_checkpoint_sha256":sha(BACKBONE),"base_model_path":str(model_dir),"base_config_sha256":sha(model_dir/"config.json"),"steps":tr["steps"],"precision":{"base":"bfloat16","simulator":"float32 exact statevector","device":"CUDA GPU only"},"replacement_state_dict":{k:v.detach().cpu() for k,v in rep.state_dict().items()},"optimizer":"AdamW","optimizer_state_dict":opt.state_dict(),"scheduler_state_dict":None,"training_summary":{k:v for k,v in tr.items() if k!="losses"}},OUT/f"{candidate}-screen-checkpoint.pt")
 payload={"status":"ok","candidate":candidate,"lock_sha256":sha(LOCK),"source_sha256":sha(SOURCE),"dataset":{"train_indices":lock["train_indices"] if tr["steps"] else [],"eval_indices":lock["eval_indices"],"validation_rows_inspected":0,"test_rows_inspected":0},"injection":audit,"training":tr,"evaluation":ev,"created_at":datetime.now(timezone.utc).isoformat()};path=OUT/f"screen-{candidate}-s{lock['seed']}-n256-e64.json";path.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":"ok","candidate":candidate,"nll":ev["metrics"]["mean_token_nll"],"sha256":sha(path)},indent=2))
def analyze():
 def load(n):return json.loads(next(OUT.glob(f"screen-{n}-*.json")).read_text())
 def ci(a,b,s):
  d=np.asarray([x["mean_token_nll"] for x in a["evaluation"]["rows"]])-np.asarray([x["mean_token_nll"] for x in b["evaluation"]["rows"]]);r=np.random.default_rng(s);m=d[r.integers(0,len(d),size=(20000,len(d)))].mean(1);return {"mean_delta_nll":float(d.mean()),"descriptive_95_ci":[float(np.quantile(m,.025)),float(np.quantile(m,.975))]}
 d={n:load(n) for n in ("base","scaffold","qh043","cc043")};nll={n:v["evaluation"]["metrics"]["mean_token_nll"] for n,v in d.items()};cmp={"scaffold_minus_base":ci(d["scaffold"],d["base"],1),"qh_minus_scaffold":ci(d["qh043"],d["scaffold"],2),"qh_minus_cc":ci(d["qh043"],d["cc043"],3),"qh_minus_base":ci(d["qh043"],d["base"],4)};g={"qh_point_better_than_own_scaffold":nll["qh043"]<nll["scaffold"],"qh_point_better_than_cc":nll["qh043"]<nll["cc043"],"qh_within_base_plus_0p005":nll["qh043"]-nll["base"]<=.005};g["continue_to_c4"]=all(g.values());p={"status":"ok","analysis":"QH043 harmonic signed-X train-only screen","verdict":"continue_to_new_c4" if g["continue_to_c4"] else "reject_or_mutate_before_c4","point_nll":nll,"comparisons":cmp,"screen_gates":g,"training":{n:{k:v for k,v in d[n]["training"].items() if k!="losses"} for n in ("qh043","cc043")},"claim_limit":"Train-only screen","created_at":datetime.now(timezone.utc).isoformat()};(OUT/"qh043-harmonic-screen-analysis.json").write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument("mode",choices=["static","smoke","lock","screen","analyze"]);p.add_argument("--branch",choices=["quantum","classical"]);p.add_argument("--candidate",choices=["base","scaffold","qh043","cc043"]);a=p.parse_args();{"static":lambda:static(),"smoke":lambda:smoke(a.branch),"lock":lambda:lock(),"screen":lambda:screen(a.candidate),"analyze":lambda:analyze()}[a.mode]()
if __name__=="__main__":main()
