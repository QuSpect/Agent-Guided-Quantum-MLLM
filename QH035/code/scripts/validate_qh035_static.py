#!/usr/bin/env python3
"""Static identity, DQ, gradient, and ledger validation for QH035."""
from __future__ import annotations
import json,sys
from datetime import datetime,timezone
from pathlib import Path
import torch
PROJECT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(PROJECT/"code/src"))
from quantum_qwen38.qh035_shared_rmsnorm_replacement import DequantizedNormCore,EqualParameterClassicalNormCore,ExactQuantumNormCore,QH035Config,qh035_parameter_audit
def rel(a,b): return float((a-b).float().norm()/b.float().norm().clamp_min(1e-12))
def main():
    if not torch.cuda.is_available(): raise RuntimeError("QH035 static requires CUDA")
    torch.manual_seed(20260860); device=torch.device("cuda:0")
    cfg=QH035Config(hidden_size=48,state_dimension=64,layer_indices=(0,1,2,3),num_qubits=6,depth=2,entangling_offsets=(1,2,3))
    qh=ExactQuantumNormCore(cfg).to(device); dq=DequantizedNormCore(cfg).to(device); cc=EqualParameterClassicalNormCore(cfg).to(device); no=ExactQuantumNormCore(cfg,use_entanglers=False).to(device)
    if {m.parameter_count for m in (qh,dq,cc,no)}!={48}: raise AssertionError("parameter mismatch")
    x=torch.randn(2,3,cfg.hidden_size,device=device); identities={n:float((m(x)-x).float().abs().max()) for n,m in (("qh",qh),("dq",dq),("cc",cc),("no_ent",no))}
    if max(identities.values())>2e-7: raise AssertionError(f"identity failed {identities}")
    with torch.no_grad(): qh.theta.uniform_(-.2,.2); dq.theta.copy_(qh.theta)
    xq=torch.randn(2,3,cfg.hidden_size,device=device,requires_grad=True); xd=xq.detach().clone().requires_grad_(True); yq=qh(xq); yd=dq(xd); probe=torch.randn_like(yq)
    (yq*probe).sum().backward(); (yd*probe).sum().backward(); checks={"forward_max_abs":float((yq-yd).float().abs().max()),
      "input_gradient_max_abs":float((xq.grad-xd.grad).float().abs().max()),"theta_gradient_max_abs":float((qh.theta.grad-dq.theta.grad).abs().max()),
      "forward_relative_l2":rel(yq,yd),"input_gradient_relative_l2":rel(xq.grad,xd.grad),"theta_gradient_relative_l2":rel(qh.theta.grad,dq.theta.grad)}
    if checks["forward_max_abs"]>3e-5 or checks["input_gradient_max_abs"]>3e-5 or checks["theta_gradient_max_abs"]>5e-4: raise AssertionError(f"DQ mismatch {checks}")
    grads={}
    for name,module in (("qh",qh),("dq",dq),("cc",cc),("no_ent",no)):
        module.zero_grad(set_to_none=True); module.theta.data.uniform_(-.15,.15); module(torch.randn(4,cfg.hidden_size,device=device)).float().square().mean().backward()
        grads[name]={"norm":float(module.theta.grad.norm()),"nonzero":int((module.theta.grad!=0).sum()),"finite":bool(torch.isfinite(module.theta.grad).all())}
        if not grads[name]["finite"] or grads[name]["norm"]<=0: raise AssertionError(f"gradient failed {name}")
    rejected=False
    try: ExactQuantumNormCore(cfg)(torch.randn(1,cfg.hidden_size))
    except RuntimeError: rejected=True
    if not rejected: raise AssertionError("CPU path accepted")
    audit=qh035_parameter_audit(QH035Config())
    if audit["trainable_parameters"]!=108 or audit["net_parameter_reduction"]!=15252: raise AssertionError(f"ledger changed {audit}")
    payload={"status":"pass","candidate":"QH-035 shared RMSNorm quantum residual","small_config":cfg.__dict__,"zero_angle_identity":identities,
      "dequantization_checks":checks,"gradient_audit":grads,"cpu_quantum_path_rejected":rejected,"full_parameter_audit":audit,
      "created_at":datetime.now(timezone.utc).isoformat()}
    out=PROJECT/"artifacts/qh035-static-validation.json"; out.write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
