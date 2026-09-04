#!/usr/bin/env python3
"""Static exact/DQ validation for QH042 signed XZ correlators."""
from __future__ import annotations
import json,sys
from datetime import datetime,timezone
from pathlib import Path
import torch
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh042_signed_xz_correlator_residual import *
def main():
 d=torch.device("cuda:0");cfg=QH042Config(hidden_size=16,intermediate_size=24,target_layer=0,latent_size=8,group_size=4,num_qubits=4,depth=2);q=ExactGroupedXZReadoutCore(cfg).to(d);dq=DequantizedGroupedXZReadoutCore(cfg).to(d);dq.load_state_dict(q.state_dict());torch.manual_seed(42)
 with torch.no_grad():q.theta.normal_(0,.2);dq.theta.copy_(q.theta)
 x=torch.randn(3,5,cfg.latent_size,device=d,requires_grad=True);x2=x.detach().clone().requires_grad_(True);y=q(x);y2=dq(x2);p=torch.randn_like(y);(y*p).sum().backward();(y2*p).sum().backward();diffs={"forward":float((y-y2).abs().max()),"input_grad":float((x.grad-x2.grad).abs().max()),"theta_grad":float((q.theta.grad-dq.theta.grad).abs().max())}
 # At zero theta/no entanglers, flipping only X-coordinate q flips <X_q Z_q+1>.
 q0=ExactGroupedXZReadoutCore(cfg,use_entanglers=False).to(d);a=torch.tensor([.4,-.7,.2,.1, .3,.6,-.2,.5],device=d).reshape(1,8);b=a.clone();b[0,0]*=-1;ya=q0(a);yb=q0(b);signed=float((ya[0,0]+yb[0,0]).abs());cpu=False
 try:q(torch.randn(2,cfg.latent_size))
 except RuntimeError:cpu=True
 payload={"status":"pass","q_dq_max_abs":diffs,"single_coordinate_sign_flip_error":signed,"theta_nonzero":int((q.theta.grad.abs()>0).sum()),"cpu_rejected":cpu,"parameter_audit":qh042_parameter_audit(cfg),"created_at":datetime.now(timezone.utc).isoformat()}
 if max(diffs.values())>2e-5 or signed>2e-6 or payload["theta_nonzero"]!=q.theta.numel() or not cpu:raise AssertionError(payload)
 out=PROJECT/"artifacts/qh042-static-validation.json";out.write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
