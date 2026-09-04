#!/usr/bin/env python3
"""Lock a fresh train-only partition for QH034 spectral modulation."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
PROJECT=Path(__file__).resolve().parents[2]; ART=PROJECT/"artifacts"
TOKENS=PROJECT/"datasets/processed/wikitext2_qwen38/train-tokens.pt"
STATIC=ART/"qh034-static-validation.json"; SMOKE=ART/"qh034-full-model-smoke.json"; KL_SMOKE=ART/"qh034-logit-kl-smoke.json"
SOURCE=PROJECT/"code/src/quantum_qwen38/qh034_spectral_modulation_replacement.py"
OUTPUT=ART/"qh034-spectral-screen-lock.json"; SEED=20260859; LENGTH=256; TRAIN_COUNT=256; EVAL_COUNT=64
def sha256(path):
    d=hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda:h.read(8<<20),b""): d.update(chunk)
    return d.hexdigest()
def used_indices():
    used=set(); parsed=0
    for path in ART.rglob("*.json"):
        if path==OUTPUT: continue
        try: value=json.loads(path.read_text())
        except Exception: continue
        parsed+=1
        def walk(x):
            if isinstance(x,dict):
                for key,child in x.items():
                    if key in {"train_indices","train_block_indices"} and isinstance(child,list): used.update(int(i) for i in child)
                    elif key=="train_block_index" and isinstance(child,int): used.add(child)
                    walk(child)
            elif isinstance(x,list):
                for child in x: walk(child)
        walk(value)
    return used,parsed
def main():
    for path in (TOKENS,STATIC,SMOKE,KL_SMOKE,SOURCE):
        if not path.exists(): raise FileNotFoundError(path)
    validations={name:json.loads(path.read_text()) for name,path in (("static",STATIC),("full_model",SMOKE),("logit_kl",KL_SMOKE))}
    if any(value["status"]!="pass" for value in validations.values()): raise RuntimeError("QH034 engineering gate failed")
    tokens=torch.load(TOKENS,map_location="cpu",weights_only=True); complete=len(tokens)//LENGTH
    used,parsed=used_indices(); available=np.asarray(sorted(set(range(complete))-used),dtype=np.int64)
    selected=np.random.default_rng(SEED).permutation(available)[:TRAIN_COUNT+EVAL_COUNT].tolist()
    if len(selected)!=TRAIN_COUNT+EVAL_COUNT: raise RuntimeError("insufficient unused train blocks")
    train_indices=selected[:TRAIN_COUNT]; eval_indices=selected[TRAIN_COUNT:]
    if set(train_indices)&set(eval_indices) or set(selected)&used: raise AssertionError("QH034 overlap")
    payload={"status":"locked_before_qh034_train_only_screen_results",
      "candidate":"QH-034 runtime quantum shared-SVD spectral modulation",
      "architecture":{"layers":[55,59],"rank":1024,"qubits":10,"depth":2,"offsets":[1,2,4],
        "circuit_parameters":80,"layer_gammas":2,"trainable":82,"spectral_scales":1024,"positive_mean_one":True,
        "net_model_parameter_reduction":3145646,"zero_angles":"exact shared-SVD identity"},
      "controls":{"equal_parameter_classical":"80 fixed Walsh modes plus two layer gammas; positive mean-one scales",
        "own_zero":"same shared-SVD with branch disabled","no_ent":"same circuit angles with controlled rotations removed",
        "dequantized":"independent tensor-axis exact gate contraction"},
      "evidence":{name:{"path":str(path),"sha256":sha256(path)} for name,path in (("static",STATIC),("full_model",SMOKE),("logit_kl",KL_SMOKE))},
      "source":{"path":str(SOURCE),"sha256":sha256(SOURCE)},
      "dataset":"Salesforce/wikitext wikitext-2-raw-v1 train only","tokens_sha256":sha256(TOKENS),
      "sequence_length":LENGTH,"seed":SEED,"train_indices":train_indices,"eval_indices":eval_indices,
      "train_count":TRAIN_COUNT,"eval_count":EVAL_COUNT,"excluded_prior_train_indices_count":len(used),
      "available_before_selection":len(available),"prior_json_parsed":parsed,"overlap":0,
      "training_protocol":{"objective":"mean-token KL(teacher dense Qwen || compressed student) at temperature 1",
        "initialization":"zero theta, gamma one; exact shared-SVD","optimizer":"AdamW","learning_rate":0.001,
        "weight_decay":0.0,"gradient_clip_norm":1.0,"steps":TRAIN_COUNT,
        "teacher":"same frozen model with original dense layers55/59 v_proj","all_other_parameters_frozen":True},
      "screen_rule":{"continue_to_new_C4_only_if":["QH034 point NLL lower than shared-SVD own-zero","QH034 point NLL lower than equal-parameter CC034"],
        "otherwise":"reject or structurally mutate before any C4","no_claim_from_train_screen":True},
      "validation_rows_inspected":0,"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()}
    OUTPUT.write_text(json.dumps(payload,indent=2)); print(json.dumps({"status":payload["status"],"train_count":TRAIN_COUNT,
      "eval_count":EVAL_COUNT,"excluded":len(used),"available":len(available),"sha256":sha256(OUTPUT)},indent=2))
if __name__=="__main__": main()
