#!/usr/bin/env python3
"""Train-only causal screen for selectively reopening QH044 entanglers."""
from __future__ import annotations
import argparse, hashlib, json, math, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForMultimodalLM

PROJECT=Path(__file__).resolve().parents[2]; ART=PROJECT/"artifacts"
TOKENS=PROJECT/"datasets/processed/wikitext2_qwen38/train-tokens.pt"
START=ART/"qh044_extended_distillation/qh044-checkpoint.pt"
SOURCE=PROJECT/"code/src/quantum_qwen38/qh045_selective_entanglement_ffn_replacement.py"
ENTRY=PROJECT/"code/scripts/qh045_selective_entanglement_screen.py"
LOCK=ART/"qh045-selective-entanglement-lock.json"; OUT=ART/"qh045_selective_entanglement_screen"
sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh045_selective_entanglement_ffn_replacement import install_qh045

def sha(path):
 d=hashlib.sha256()
 with Path(path).open("rb") as h:
  for chunk in iter(lambda:h.read(8<<20),b""):d.update(chunk)
 return d.hexdigest()

def make_lock():
 used=set()
 for path in ART.rglob("*.json"):
  if path==LOCK:continue
  try:text=path.read_text(); value=json.loads(text)
  except Exception:continue
  if "wikitext" not in text.lower():continue
  def walk(x):
   if isinstance(x,dict):
    for key,child in x.items():
     if key in {"train_indices","train_block_indices","eval_indices","evaluation_indices","block_indices"} and isinstance(child,list):used.update(int(i) for i in child)
     elif key in {"train_block_index","block_index"} and isinstance(child,int):used.add(child)
     walk(child)
   elif isinstance(x,list):
    for child in x:walk(child)
  walk(value)
 tokens=torch.load(TOKENS,map_location="cpu",weights_only=True)
 available=np.asarray(sorted(set(range(len(tokens)//256))-used),dtype=np.int64)
 selected=np.random.default_rng(20260885).permutation(available)[:448].tolist()
 if len(selected)!=448 or set(selected)&used:raise RuntimeError({"available":len(available)})
 payload={"status":"locked_before_qh045_results","parent":"QH044 formal C4 found fixed ring entanglement significantly harmful","literature_basis":["arXiv:2010.10217 quantum circuit architecture search","arXiv:2107.10845 QuantumNAS gate pruning"],"single_mutation":"load the identical frozen QH044 checkpoint in both arms, set 64 shared-across-depth entanglement selectors exactly to zero, and train only those selectors; QH scales saved controlled-RY angles, CC adds a matched neighbour-product interaction after the same frozen no-ent simulator scaffold","dataset":"Salesforce/wikitext wikitext-2-raw-v1 train only","tokens_sha256":sha(TOKENS),"start_checkpoint_sha256":sha(START),"seed":20260885,"sequence_length":256,"train_indices":selected[:384],"eval_indices":selected[384:],"train_count":384,"eval_count":64,"excluded_prior_wikitext_indices_count":len(used),"overlap":0,"training":{"steps":512,"optimizer":"AdamW","learning_rate":.01,"weight_decay":0.0,"gradient_clip_norm":1.0,"objective":"mean-token teacher-logit KL","base_and_all_QH044_parameters_frozen":True,"trainable_parameters_per_arm":64},"parameter_audit":{"removed":267386880,"deployed":655680,"net_reduction":266731200},"gates":{"promote_only_if":["QH point NLL lower than shared no-ent start","QH point NLL lower than equal-parameter CC","QH within base +0.005"],"formal_holdout":"new C4 only after all screen point gates"},"validation_rows_inspected":0,"test_rows_inspected":0,"source_hashes":{"source":sha(SOURCE),"entrypoint":sha(ENTRY)},"created_at":datetime.now(timezone.utc).isoformat()}
 LOCK.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":payload["status"],"available":len(available),"excluded":len(used),"sha256":sha(LOCK)},indent=2))

def load_model():
 model_dir=Path((PROJECT/"records/active_model_path.txt").read_text().strip())
 model=AutoModelForMultimodalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,device_map="balanced",low_cpu_mem_usage=True,attn_implementation="sdpa")
 model.eval();model.config.use_cache=False;return model,model_dir
def block(tokens,index,device):return tokens[index*256:(index+1)*256].to(device=device,dtype=torch.long).unsqueeze(0)
@torch.inference_mode()
def evaluate(model,tokens,indices):
 rows=[]
 for index in indices:
  inputs=block(tokens,index,model.device);started=time.perf_counter();loss=model(input_ids=inputs,labels=inputs,use_cache=False).loss;torch.cuda.synchronize();rows.append({"block_index":index,"mean_token_nll":float(loss),"seconds":time.perf_counter()-started})
 nll=float(np.mean([row["mean_token_nll"] for row in rows]));return {"metrics":{"mean_token_nll":nll,"perplexity":math.exp(nll)},"rows":rows}

def run(candidate):
 lock=json.loads(LOCK.read_text());random.seed(lock["seed"]);np.random.seed(lock["seed"]);torch.manual_seed(lock["seed"]);torch.cuda.manual_seed_all(lock["seed"])
 tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);model,model_dir=load_model();replacement=original=optimizer=None;training={"steps":0}
 if candidate=="base":model.requires_grad_(False);audit={"branch":"base","trainable_parameters":0,"net_model_parameter_reduction":0}
 else:
  state=torch.load(START,map_location="cpu",weights_only=False)["replacement_state_dict"]
  branch="quantum" if candidate in {"noent","qh045"} else "classical";replacement,original,audit=install_qh045(model,state,branch)
  if candidate=="noent":replacement.requires_grad_(False)
  else:
   params=[replacement.core.gate_raw];optimizer=torch.optim.AdamW(params,lr=.01,weight_decay=0.0);losses=[];seconds=[];before=int(replacement.core.circuit_calls.item())
   for step in range(1,513):
    index=lock["train_indices"][(step-1)%384];inputs=block(tokens,index,model.device);model.model.language_model.layers[31].mlp=original
    with torch.inference_mode():teacher=model(input_ids=inputs,use_cache=False).logits[:,:-1].detach()
    model.model.language_model.layers[31].mlp=replacement;optimizer.zero_grad(set_to_none=True);started=time.perf_counter();student=model(input_ids=inputs,use_cache=False).logits[:,:-1];loss=F.kl_div(F.log_softmax(student.float(),-1),F.softmax(teacher.float(),-1),reduction="sum")/(student.shape[0]*student.shape[1]);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);optimizer.step();torch.cuda.synchronize();losses.append(float(loss));seconds.append(time.perf_counter()-started)
    if step%128==0:print(json.dumps({"candidate":candidate,"step":step,"recent_kl":float(np.mean(losses[-128:])),"gate_l2":float(torch.linalg.vector_norm(torch.tanh(replacement.core.gate_raw)).detach())}),flush=True)
   frozen=sum(int(parameter.grad is not None) for parameter in model.parameters() if not parameter.requires_grad);calls=int(replacement.core.circuit_calls.item())-before
   gates=torch.tanh(replacement.core.gate_raw.detach()).cpu().numpy();training={"steps":512,"kl_first64":float(np.mean(losses[:64])),"kl_last64":float(np.mean(losses[-64:])),"mean_step_seconds":float(np.mean(seconds)),"simulator_calls":calls,"frozen_gradient_tensor_count":frozen,"gate_l2":float(np.linalg.norm(gates)),"gate_max_abs":float(np.max(np.abs(gates))),"gate_abs_below_0p05":int(np.sum(np.abs(gates)<.05)),"gate_count":int(gates.size)}
   if frozen or calls!=512:raise AssertionError({"frozen":frozen,"calls":calls})
 result=evaluate(model,tokens,lock["eval_indices"]);OUT.mkdir(parents=True,exist_ok=True)
 if candidate in {"qh045","cc045"}:torch.save({"candidate":candidate,"data_lock_sha256":sha(LOCK),"source_sha256":sha(SOURCE),"entrypoint_sha256":sha(ENTRY),"start_checkpoint_sha256":sha(START),"base_model_path":str(model_dir),"base_config_sha256":sha(model_dir/"config.json"),"steps":512,"precision":{"base":"bfloat16","simulator":"float32 exact statevector","device":"CUDA GPU only"},"replacement_state_dict":{key:value.detach().cpu() for key,value in replacement.state_dict().items()},"optimizer_state_dict":optimizer.state_dict(),"training_summary":training},OUT/f"{candidate}-checkpoint.pt")
 payload={"status":"ok","candidate":candidate,"lock_sha256":sha(LOCK),"audit":audit,"training":training,"evaluation":result,"validation_rows_inspected":0,"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()};path=OUT/f"screen-{candidate}-s{lock['seed']}-n384-e64.json";path.write_text(json.dumps(payload,indent=2));print(json.dumps({"candidate":candidate,"nll":result["metrics"]["mean_token_nll"],"training":training,"sha256":sha(path)},indent=2))

def analyze():
 def load(name):return json.loads(next(OUT.glob(f"screen-{name}-*.json")).read_text())
 def comparison(a,b,seed):
  delta=np.asarray([x["mean_token_nll"] for x in a["evaluation"]["rows"]])-np.asarray([x["mean_token_nll"] for x in b["evaluation"]["rows"]]);rng=np.random.default_rng(seed);means=delta[rng.integers(0,len(delta),size=(20000,len(delta)))].mean(1);return {"mean_delta_nll":float(delta.mean()),"descriptive_95_ci":[float(np.quantile(means,.025)),float(np.quantile(means,.975))]}
 data={name:load(name) for name in ("base","noent","qh045","cc045")};nll={name:value["evaluation"]["metrics"]["mean_token_nll"] for name,value in data.items()};comparisons={"qh_minus_noent":comparison(data["qh045"],data["noent"],1),"qh_minus_cc":comparison(data["qh045"],data["cc045"],2),"qh_minus_base":comparison(data["qh045"],data["base"],3)};gates={"qh_better_than_noent":nll["qh045"]<nll["noent"],"qh_better_than_cc":nll["qh045"]<nll["cc045"],"qh_within_base_plus_0p005":nll["qh045"]-nll["base"]<=.005};gates["promote_to_new_c4"]=all(gates.values());payload={"status":"ok","analysis":"QH045 selective entanglement train-only screen","point_nll":nll,"comparisons":comparisons,"gate_diagnostics":{"qh":data["qh045"]["training"],"cc":data["cc045"]["training"]},"gates":gates,"verdict":"promote_to_new_c4" if gates["promote_to_new_c4"] else "reject_or_mutate","claim_limit":"train-only; both arms share frozen QH044 no-ent simulator scaffold","created_at":datetime.now(timezone.utc).isoformat()};(OUT/"qh045-selective-entanglement-analysis.json").write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))

def main():
 parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["lock","run","analyze"]);parser.add_argument("--candidate",choices=["base","noent","qh045","cc045"]);args=parser.parse_args();{"lock":make_lock,"run":lambda:run(args.candidate),"analyze":analyze}[args.mode]()
if __name__=="__main__":main()
