#!/usr/bin/env python3
"""Static exact/DQ and trust-region validation for QH040."""
from __future__ import annotations
import json,sys
from datetime import datetime,timezone
from pathlib import Path
import torch
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh037_grouped_qvaf_ffn_replacement import EqualParameterClassicalQVAFCore,GroupedQVAFReplacement,ExactGroupedQVAFCore,DequantizedGroupedQVAFCore
from quantum_qwen38.qh040_bounded_quantum_residual import *
def main():
 d=torch.device("cuda:0");cfg=QH040Config(hidden_size=16,intermediate_size=24,target_layer=0,latent_size=8,group_size=4,num_qubits=4,depth=2);base=GroupedQVAFReplacement(cfg,EqualParameterClassicalQVAFCore(cfg).to(d),d,torch.float32);state={k:v.detach().cpu() for k,v in base.state_dict().items()};q=BoundedScaffoldResidualReplacement(cfg,state,ExactGroupedQVAFCore(cfg),d,torch.float32);dq=BoundedScaffoldResidualReplacement(cfg,state,DequantizedGroupedQVAFCore(cfg),d,torch.float32);dq.load_state_dict(q.state_dict());cc=BoundedScaffoldResidualReplacement(cfg,state,EqualParameterClassicalQVAFCore(cfg),d,torch.float32);torch.manual_seed(40)
 with torch.no_grad():q.core.theta.normal_(0,.2);dq.core.theta.copy_(q.core.theta);q.alpha.uniform_(-2,2);dq.alpha.copy_(q.alpha)
 x=torch.randn(3,5,cfg.hidden_size,device=d,requires_grad=True);x2=x.detach().clone().requires_grad_(True);y=q(x);y2=dq(x2);probe=torch.randn_like(y);(y*probe).sum().backward();(y2*probe).sum().backward();diffs={"forward":float((y-y2).abs().max()),"input_grad":float((x.grad-x2.grad).abs().max()),"theta_grad":float((q.core.theta.grad-dq.core.theta.grad).abs().max()),"alpha_grad":float((q.alpha.grad-dq.alpha.grad).abs().max())}
 z=torch.randn(2,4,cfg.hidden_size,device=d);q.correction_enabled=False;a=q(z);q.correction_enabled=True;q.alpha.data.fill_(-100);lo=q(z);q.alpha.data.fill_(100);hi=q(z);max_eff=float(q.effective_alpha().abs().max());cc(z).square().mean().backward();frozen=sum(int(p.grad is not None) for p in list(cc.down.parameters())+list(cc.base_core.parameters())+list(cc.up.parameters()));payload={"status":"pass","q_dq_max_abs":diffs,"branch_off_finite":bool(torch.isfinite(a).all()),"saturated_outputs_finite":bool(torch.isfinite(lo).all() and torch.isfinite(hi).all()),"max_effective_alpha":max_eff,"cap":cfg.residual_alpha_cap,"cc_theta_nonzero":int((cc.core.theta.grad.abs()>0).sum()),"cc_alpha_nonzero":int((cc.alpha.grad.abs()>0).sum()),"frozen_scaffold_gradient_tensor_count":frozen,"parameter_audit":qh040_parameter_audit(cfg),"created_at":datetime.now(timezone.utc).isoformat()}
 if max(diffs.values())>2e-5 or max_eff>cfg.residual_alpha_cap+1e-7 or frozen or payload["cc_theta_nonzero"]!=cc.core.theta.numel() or payload["cc_alpha_nonzero"]!=cc.alpha.numel():raise AssertionError(payload)
 out=PROJECT/"artifacts/qh040-static-validation.json";out.write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
