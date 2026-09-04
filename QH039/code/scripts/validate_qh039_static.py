#!/usr/bin/env python3
"""Static exact/DQ and frozen-scaffold validation for QH039."""
from __future__ import annotations
import json,sys
from datetime import datetime,timezone
from pathlib import Path
import torch
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh037_grouped_qvaf_ffn_replacement import EqualParameterClassicalQVAFCore,GroupedQVAFReplacement
from quantum_qwen38.qh039_frozen_scaffold_quantum_residual import *
from quantum_qwen38.qh037_grouped_qvaf_ffn_replacement import ExactGroupedQVAFCore,DequantizedGroupedQVAFCore
def main():
 if not torch.cuda.is_available():raise RuntimeError("CUDA required")
 d=torch.device("cuda:0");cfg=QH039Config(hidden_size=16,intermediate_size=24,target_layer=0,latent_size=8,group_size=4,num_qubits=4,depth=2);torch.manual_seed(39);base=GroupedQVAFReplacement(cfg,EqualParameterClassicalQVAFCore(cfg).to(d),d,torch.float32);state={k:v.detach().cpu() for k,v in base.state_dict().items()};q=FrozenScaffoldResidualReplacement(cfg,state,ExactGroupedQVAFCore(cfg),d,torch.float32);dq=FrozenScaffoldResidualReplacement(cfg,state,DequantizedGroupedQVAFCore(cfg),d,torch.float32);dq.load_state_dict(q.state_dict());cc=FrozenScaffoldResidualReplacement(cfg,state,EqualParameterClassicalQVAFCore(cfg),d,torch.float32)
 with torch.no_grad():q.core.theta.normal_(0,.2);dq.core.theta.copy_(q.core.theta);q.alpha.normal_(0,.01);dq.alpha.copy_(q.alpha)
 x=torch.randn(3,5,cfg.hidden_size,device=d,requires_grad=True);x2=x.detach().clone().requires_grad_(True);y=q(x);y2=dq(x2);probe=torch.randn_like(y);(y*probe).sum().backward();(y2*probe).sum().backward();diffs={"forward":float((y-y2).abs().max()),"input_grad":float((x.grad-x2.grad).abs().max()),"theta_grad":float((q.core.theta.grad-dq.core.theta.grad).abs().max()),"alpha_grad":float((q.alpha.grad-dq.alpha.grad).abs().max())}
 if max(diffs.values())>2e-5:raise AssertionError(diffs)
 z=torch.randn(2,4,cfg.hidden_size,device=d);q.correction_enabled=False;a=q(z);q.correction_enabled=True;q.alpha.data.zero_();b=q(z);zero=float((a-b).abs().max());loss=cc(z).square().mean();loss.backward();dead=int((cc.core.theta.grad.abs()>0).sum())!=cc.core.theta.numel() or int((cc.alpha.grad.abs()>0).sum())!=cc.alpha.numel();frozen=sum(int(p.grad is not None) for p in list(cc.down.parameters())+list(cc.base_core.parameters())+list(cc.up.parameters()))
 cpu=False
 try:q.core(torch.randn(2,cfg.latent_size))
 except RuntimeError:cpu=True
 payload={"status":"pass","config":cfg.__dict__,"q_dq_max_abs":diffs,"alpha_zero_vs_branch_off_max_abs":zero,"cc_dead_trainables":dead,"frozen_scaffold_gradient_tensor_count":frozen,"cpu_rejected":cpu,"parameter_audit":qh039_parameter_audit(cfg),"created_at":datetime.now(timezone.utc).isoformat()}
 if zero!=0 or dead or frozen or not cpu:raise AssertionError(payload)
 out=PROJECT/"artifacts/qh039-static-validation.json";out.write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
