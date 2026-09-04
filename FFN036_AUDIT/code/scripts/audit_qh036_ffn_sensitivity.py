#!/usr/bin/env python3
"""Rank Qwen3.8 FFN blocks by paired NLL damage when the entire MLP is zeroed."""
from __future__ import annotations
import hashlib,json,math,random,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from torch import nn
from transformers import AutoModelForMultimodalLM
PROJECT=Path(__file__).resolve().parents[2];ART=PROJECT/"artifacts";LOCK=ART/"qh036-ffn-sensitivity-lock.json";TOKENS=PROJECT/"datasets/processed/wikitext2_qwen38/train-tokens.pt";OUT=ART/"qh036-ffn-sensitivity-audit.json"
def sha(path):
 d=hashlib.sha256()
 with path.open("rb") as h:
  for c in iter(lambda:h.read(8<<20),b""):d.update(c)
 return d.hexdigest()
class ZeroFFN(nn.Module):
 def forward(self,x):return torch.zeros_like(x)
def block(tokens,i,n,device):return tokens[i*n:(i+1)*n].to(device=device,dtype=torch.long).unsqueeze(0)
@torch.inference_mode()
def evaluate(model,tokens,indices,n):
 rows=[]
 for i in indices:
  x=block(tokens,int(i),n,model.device);started=time.perf_counter();loss=model(input_ids=x,labels=x,use_cache=False).loss;torch.cuda.synchronize();rows.append({"block_index":int(i),"mean_token_nll":float(loss),"seconds":time.perf_counter()-started})
 return rows
def paired_ci(delta,seed,n_boot=20000):
 a=np.asarray(delta,dtype=np.float64);rng=np.random.default_rng(seed);means=a[rng.integers(0,len(a),size=(n_boot,len(a)))].mean(1)
 return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]
def main():
 lock=json.loads(LOCK.read_text())
 if lock["status"]!="locked_before_qh036_ffn_sensitivity_results":raise RuntimeError("invalid lock")
 random.seed(lock["seed"]);np.random.seed(lock["seed"]);torch.manual_seed(lock["seed"]);torch.cuda.manual_seed_all(lock["seed"])
 tokens=torch.load(TOKENS,map_location="cpu",weights_only=True);model_dir=Path((PROJECT/"records/active_model_path.txt").read_text().strip())
 model=AutoModelForMultimodalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,device_map="balanced",low_cpu_mem_usage=True,attn_implementation="sdpa")
 model.eval();model.requires_grad_(False);model.config.use_cache=False;layers=model.model.language_model.layers;indices=lock["block_indices"];n=lock["sequence_length"]
 base_rows=evaluate(model,tokens,indices,n);base=np.asarray([r["mean_token_nll"] for r in base_rows]);results=[]
 for li,layer in enumerate(layers):
  original=layer.mlp;removed=sum(p.numel() for p in original.parameters());layer.mlp=ZeroFFN()
  try:rows=evaluate(model,tokens,indices,n)
  finally:layer.mlp=original
  values=np.asarray([r["mean_token_nll"] for r in rows]);delta=values-base
  item={"layer_index":li,"mlp_class":original.__class__.__name__,"original_mlp_parameters":removed,"mean_nll":float(values.mean()),"mean_delta_nll":float(delta.mean()),"paired_delta_95_ci":paired_ci(delta,lock["seed"]+li),"per_block_delta":delta.tolist(),"mean_seconds":float(np.mean([r["seconds"] for r in rows]))}
  results.append(item);print(json.dumps({"layer":li,"delta_nll":item["mean_delta_nll"],"ci":item["paired_delta_95_ci"]}),flush=True)
 ordered=sorted(results,key=lambda x:x["mean_delta_nll"]);payload={"status":"complete_train_only_no_optimization","purpose":"rank whole-FFN deletion sensitivity before any activation fit",
  "lock_sha256":sha(LOCK),"model_config_sha256":"191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
  "base":{"mean_nll":float(base.mean()),"perplexity":math.exp(float(base.mean())),"rows":base_rows},"layers":results,"top8":[{k:v for k,v in x.items() if k!="per_block_delta"} for x in ordered[:8]],
  "selection_rule":"Use top deletion-tolerant layers only as candidates; do not infer linear recoverability from deletion alone",
  "dataset":{"block_indices":indices,"validation_rows_inspected":0,"test_rows_inspected":0},"created_at":datetime.now(timezone.utc).isoformat()}
 OUT.write_text(json.dumps(payload,indent=2));print(json.dumps({"status":payload["status"],"base_nll":payload["base"]["mean_nll"],"top8":[[x["layer_index"],x["mean_delta_nll"]] for x in ordered[:8]],"sha256":sha(OUT)},indent=2))
if __name__=="__main__":main()
