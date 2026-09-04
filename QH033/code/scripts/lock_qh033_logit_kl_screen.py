#!/usr/bin/env python3
"""Lock a new train-only partition for QH033 teacher-logit KL continuation."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch

PROJECT=Path(__file__).resolve().parents[2]; ART=PROJECT/"artifacts"
TOKENS=PROJECT/"datasets/processed/wikitext2_qwen38/train-tokens.pt"
PARENT=ART/"qh032_train_screen/qh032-train-screen-analysis.json"
SMOKE=ART/"qh033-logit-kl-smoke.json"
QH=ART/"qh032_train_screen/qh032-screen-checkpoint.pt"; CC=ART/"qh032_train_screen/cc032-screen-checkpoint.pt"
OUTPUT=ART/"qh033-logit-kl-screen-lock.json"; SEED=20260853; LENGTH=256; TRAIN_COUNT=256; EVAL_COUNT=64

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
    for path in (TOKENS,PARENT,SMOKE,QH,CC):
        if not path.exists(): raise FileNotFoundError(path)
    parent=json.loads(PARENT.read_text()); smoke=json.loads(SMOKE.read_text())
    if parent["verdict"]!="reject_or_mutate_before_c4" or smoke["status"]!="pass": raise RuntimeError("QH033 parent gates invalid")
    tokens=torch.load(TOKENS,map_location="cpu",weights_only=True); complete=len(tokens)//LENGTH
    used,parsed=used_indices(); available=np.asarray(sorted(set(range(complete))-used),dtype=np.int64)
    selected=np.random.default_rng(SEED).permutation(available)[:TRAIN_COUNT+EVAL_COUNT].tolist()
    if len(selected)!=TRAIN_COUNT+EVAL_COUNT: raise RuntimeError("insufficient unused train blocks")
    train_indices=selected[:TRAIN_COUNT]; eval_indices=selected[TRAIN_COUNT:]
    if set(train_indices)&set(eval_indices) or set(selected)&used: raise AssertionError("QH033 overlap")
    payload={
      "status":"locked_before_qh033_train_only_screen_results",
      "candidate":"QH-033 fixed QH032 architecture plus teacher-logit KL stage",
      "single_mutation":"Continue frozen QH032/CC032 checkpoints with token-distribution KL; architecture and parameter count unchanged.",
      "parent":{"analysis":str(PARENT),"sha256":sha256(PARENT),"verdict":parent["verdict"]},
      "smoke":{"path":str(SMOKE),"sha256":sha256(SMOKE)},
      "starting_checkpoints":{"qh032":{"path":str(QH),"sha256":sha256(QH)},"cc032":{"path":str(CC),"sha256":sha256(CC)}},
      "architecture":{"layers":[55,59],"rank":1024,"qubits":10,"observables":60,"trainable":82,"net_model_parameter_reduction":3145646},
      "dataset":"Salesforce/wikitext wikitext-2-raw-v1 train only","tokens_sha256":sha256(TOKENS),
      "sequence_length":LENGTH,"seed":SEED,"train_indices":train_indices,"eval_indices":eval_indices,
      "train_count":TRAIN_COUNT,"eval_count":EVAL_COUNT,"excluded_prior_train_indices_count":len(used),
      "available_before_selection":len(available),"prior_json_parsed":parsed,"overlap":0,
      "training_protocol":{"objective":"mean-token KL(teacher dense Qwen || compressed student) at temperature 1",
          "optimizer":"AdamW","learning_rate":0.001,"weight_decay":0.0,"gradient_clip_norm":1.0,"steps":TRAIN_COUNT,
          "teacher":"same frozen model with original dense layers55/59 v_proj","all_other_parameters_frozen":True},
      "screen_rule":{"continue_to_new_C4_only_if":["QH033 point NLL lower than shared-SVD own-zero","QH033 point NLL lower than equal-parameter CC033"],
          "no_claim_from_train_screen":True},
      "validation_rows_inspected":0,"test_rows_inspected":0,"created_at":datetime.now(timezone.utc).isoformat()}
    OUTPUT.write_text(json.dumps(payload,indent=2)); print(json.dumps({"status":payload["status"],"train_count":TRAIN_COUNT,"eval_count":EVAL_COUNT,
        "excluded":len(used),"available":len(available),"sha256":sha256(OUTPUT)},indent=2))

if __name__=="__main__": main()

