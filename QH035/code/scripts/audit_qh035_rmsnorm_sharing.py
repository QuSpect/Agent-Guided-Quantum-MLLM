#!/usr/bin/env python3
"""Weight-only clustering audit for shared RMSNorm candidates in QH035."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import torch
from transformers import AutoModelForMultimodalLM
PROJECT=Path(__file__).resolve().parents[2]
OUTPUT=PROJECT/"artifacts/qh035-rmsnorm-sharing-audit.json"
def best_cluster(rows,k):
    matrix=torch.stack([row["weight"] for row in rows]).float()
    distances=torch.cdist(matrix,matrix)
    best=None
    for center in range(len(rows)):
        ids=torch.argsort(distances[center])[:k]
        selected=matrix[ids]; mean=selected.mean(0)
        relative=(selected-mean).norm(dim=1)/selected.norm(dim=1).clamp_min(1e-12)
        score=float(relative.square().mean().sqrt())
        item={"medoid_layer":rows[center]["layer"],"layers":[rows[int(i)]["layer"] for i in ids],
          "rms_relative_weight_error":score,"max_relative_weight_error":float(relative.max()),
          "mean_weight_mean":float(mean.mean()),"mean_weight_std":float(mean.std()),
          "net_parameter_reduction_if_qh035":k*matrix.shape[1]-matrix.shape[1]-104-k}
        if best is None or score<best["rms_relative_weight_error"]: best=item
    return best
def main():
    if not torch.cuda.is_available(): raise RuntimeError("QH035 audit expects the GPU model environment")
    model_dir=Path((PROJECT/"records/active_model_path.txt").read_text().strip())
    model=AutoModelForMultimodalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,device_map="balanced",low_cpu_mem_usage=True,attn_implementation="sdpa")
    layers=model.model.language_model.layers; families={"input_layernorm":[],"post_attention_layernorm":[]}
    individual=[]
    for index,layer in enumerate(layers):
        for kind in families:
            module=getattr(layer,kind)
            raw_weight=module.weight.detach().float().cpu()
            weight=1.0+raw_weight
            relative_one=float((weight-1).norm()/weight.norm().clamp_min(1e-12))
            row={"layer":index,"kind":kind,"weight":weight}
            families[kind].append(row); individual.append({"layer":index,"kind":kind,"raw_parameterization":"effective_scale=1+stored_weight",
              "stored_weight_mean":float(raw_weight.mean()),"mean":float(weight.mean()),
              "std":float(weight.std()),"minimum":float(weight.min()),"maximum":float(weight.max()),
              "relative_l2_from_ones":relative_one})
    clusters={kind:{str(k):best_cluster(rows,k) for k in (4,8,16)} for kind,rows in families.items()}
    payload={"status":"complete_weight_only_no_dataset","candidate":"QH035 shared RMSNorm plus runtime 13q correction",
      "model_dir":str(model_dir),"hidden_size":int(families["input_layernorm"][0]["weight"].numel()),
      "layer_count":len(layers),"individual_weights":individual,"best_same_family_clusters":clusters,
      "selection_rule":"Cluster the effective scales (1+stored_weight); choose the largest k whose RMS relative shared-mean error is <=0.01, without activation or language data.",
      "quantum_ledger":{"qubits":13,"state_dimension":8192,"depth":2,"offsets":[1,2,4],"circuit_parameters":104,
        "operation":"pad each normalized 5120-vector to 8192, exact statevector rotations, crop first 5120, residual layer gamma",
        "training_and_inference_gpu_simulator":True},"train_rows_inspected":0,"validation_rows_inspected":0,"test_rows_inspected":0,
      "created_at":datetime.now(timezone.utc).isoformat()}
    OUTPUT.write_text(json.dumps(payload,indent=2)); print(json.dumps({"status":payload["status"],"clusters":clusters},indent=2))
if __name__=="__main__": main()
