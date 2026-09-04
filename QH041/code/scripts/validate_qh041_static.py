#!/usr/bin/env python3
"""Static exact/DQ signed-X validation for QH041."""
from __future__ import annotations
import json,sys
from datetime import datetime,timezone
from pathlib import Path
import torch
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh041_signed_x_readout_residual import *
def main():
 d=torch.device("cuda:0");cfg=QH041Config(hidden_size=16,intermediate_size=24,target_layer=0,latent_size=8,group_size=4,num_qubits=4,depth=2);q=ExactGroupedXReadoutCore(cfg).to(d);dq=DequantizedGroupedXReadoutCore(cfg).to(d);dq.load_state_dict(q.state_dict());torch.manual_seed(41)
 with torch.no_grad():q.theta.normal_(0,.2);dq.theta.copy_(q.theta)
 x=torch.randn(3,5,cfg.latent_size,device=d,requires_grad=True);x2=x.detach().clone().requires_grad_(True);y=q(x);y2=dq(x2);p=torch.randn_like(y);(y*p).sum().backward();(y2*p).sum().backward();diffs={"forward":float((y-y2).abs().max()),"input_grad":float((x.grad-x2.grad).abs().max()),"theta_grad":float((q.theta.grad-dq.theta.grad).abs().max())};probe=torch.tensor([-.5,-.25,.25,.5,0,0,0,0],device=d).reshape(1,8);q0=ExactGroupedXReadoutCore(cfg).to(d);signed=q0(probe).detach().float().flatten();odd=float((signed[:2]+signed[2:4].flip(0)).abs().max());cpu=False
 try:q(torch.randn(2,cfg.latent_size))
 except RuntimeError:cpu=True
 payload={"status":"pass","q_dq_max_abs":diffs,"zero_theta_signed_odd_pair_error":odd,"theta_nonzero":int((q.theta.grad.abs()>0).sum()),"cpu_rejected":cpu,"parameter_audit":qh041_parameter_audit(cfg),"created_at":datetime.now(timezone.utc).isoformat()}
 if max(diffs.values())>2e-5 or odd>2e-6 or payload["theta_nonzero"]!=q.theta.numel() or not cpu:raise AssertionError(payload)
 out=PROJECT/"artifacts/qh041-static-validation.json";out.write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
