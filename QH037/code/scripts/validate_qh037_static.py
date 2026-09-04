#!/usr/bin/env python3
"""Static exact-vs-independent validation for QH037."""
from __future__ import annotations
import json,sys
from datetime import datetime,timezone
from pathlib import Path
import torch
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh037_grouped_qvaf_ffn_replacement import *
def main():
 if not torch.cuda.is_available():raise RuntimeError("CUDA required")
 device=torch.device("cuda:0");cfg=QH037Config(hidden_size=16,intermediate_size=24,target_layer=0,latent_size=8,group_size=4,num_qubits=4,depth=2);q=ExactGroupedQVAFCore(cfg).to(device);dq=DequantizedGroupedQVAFCore(cfg).to(device);dq.load_state_dict(q.state_dict());torch.manual_seed(37)
 with torch.no_grad():q.theta.normal_(0,.2);dq.theta.copy_(q.theta)
 x=torch.randn(3,5,cfg.latent_size,device=device,requires_grad=True);x2=x.detach().clone().requires_grad_(True);y=q(x);y2=dq(x2);probe=torch.randn_like(y);(y*probe).sum().backward();(y2*probe).sum().backward()
 diffs={"forward":float((y-y2).abs().max()),"input_grad":float((x.grad-x2.grad).abs().max()),"theta_grad":float((q.theta.grad-dq.theta.grad).abs().max())}
 if diffs["forward"]>2e-6 or diffs["input_grad"]>3e-6 or diffs["theta_grad"]>2e-5:raise AssertionError(diffs)
 cc=EqualParameterClassicalQVAFCore(cfg).to(device);z=torch.randn(2,4,cfg.latent_size,device=device,requires_grad=True);cc(z).square().mean().backward()
 if int((q.theta.grad.abs()>0).sum())!=q.theta.numel() or int((cc.theta.grad.abs()>0).sum())!=cc.theta.numel():raise AssertionError("dead parameters")
 replacement=GroupedQVAFReplacement(cfg,q,device,torch.float32);replacement.branch_enabled=False;identity=float(replacement(torch.randn(2,3,cfg.hidden_size,device=device)).abs().max())
 cpu_rejected=False
 try:q(torch.randn(2,cfg.latent_size))
 except RuntimeError:cpu_rejected=True
 audit=qh037_parameter_audit(cfg);payload={"status":"pass","config":cfg.__dict__,"q_dq_max_abs":diffs,"q_theta_nonzero_grad":int((q.theta.grad.abs()>0).sum()),"cc_theta_nonzero_grad":int((cc.theta.grad.abs()>0).sum()),"branch_off_max_abs":identity,"cpu_rejected":cpu_rejected,"parameter_audit":audit,"created_at":datetime.now(timezone.utc).isoformat()}
 if identity!=0 or not cpu_rejected:raise AssertionError(payload)
 out=PROJECT/"artifacts/qh037-static-validation.json";out.write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
