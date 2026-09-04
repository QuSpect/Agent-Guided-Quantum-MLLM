"""QH040: bounded trust-region mutation of the QH039 residual."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math,torch
from torch import Tensor
from .qh037_grouped_qvaf_ffn_replacement import DequantizedGroupedQVAFCore,EqualParameterClassicalQVAFCore,ExactGroupedQVAFCore
from .qh039_frozen_scaffold_quantum_residual import QH039Config,FrozenScaffoldResidualReplacement,load_qh037_cc_state,qh039_parameter_audit,unique_trainable_parameters

@dataclass(frozen=True)
class QH040Config(QH039Config):
    init_seed:int=20260870
    residual_alpha_cap:float=0.05

class BoundedScaffoldResidualReplacement(FrozenScaffoldResidualReplacement):
    def __init__(self,config:QH040Config,*args,**kwargs):
        super().__init__(config,*args,**kwargs)
        ratio=config.residual_alpha_init/config.residual_alpha_cap
        if not 0<ratio<1:raise ValueError("alpha init must be inside trust region")
        with torch.no_grad():self.alpha.fill_(math.atanh(ratio))
    def effective_alpha(self):return self.config.residual_alpha_cap*torch.tanh(self.alpha)
    def forward(self,x:Tensor):
        latent=self.down(x);base=self.base_core(latent)
        if not self.correction_enabled:return self.up(base)
        correction=self.core(latent).float()*self.effective_alpha()
        return self.up((base.float()+correction).to(base.dtype))

def qh040_parameter_audit(config:QH040Config):return qh039_parameter_audit(config)

def install_qh040(model,backbone_checkpoint:Path,config:QH040Config=QH040Config(),branch:str="quantum"):
    config.validate();model.requires_grad_(False);layer=model.model.language_model.layers[config.target_layer];original=layer.mlp;device=next(original.parameters()).device;dtype=next(original.parameters()).dtype
    classes={"quantum":ExactGroupedQVAFCore,"dq":DequantizedGroupedQVAFCore,"classical":EqualParameterClassicalQVAFCore,"no_ent":lambda cfg:ExactGroupedQVAFCore(cfg,use_entanglers=False)}
    if branch not in classes:raise ValueError(f"unknown QH040 branch {branch}")
    state,meta=load_qh037_cc_state(backbone_checkpoint);base=sum(p.numel() for p in model.parameters());rep=BoundedScaffoldResidualReplacement(config,state,classes[branch](config),device,dtype);layer.mlp=rep;deployed=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad);audit=qh040_parameter_audit(config)
    if base-deployed!=audit["net_parameter_reduction"] or trainable!=audit["trainable_parameters"]:raise AssertionError("QH040 parameter audit mismatch")
    return rep,original,{**audit,"base_model_parameters":base,"deployed_model_parameters":deployed,"net_model_parameter_reduction":audit["net_parameter_reduction"],"trainable_parameter_count":trainable,"branch":branch,"target_layer":config.target_layer,"backbone_candidate":meta["candidate"],"backbone_steps":meta.get("steps"),"effective_alpha_cap_per_coordinate":config.residual_alpha_cap}

__all__=["QH040Config","BoundedScaffoldResidualReplacement","install_qh040","qh040_parameter_audit","unique_trainable_parameters"]
