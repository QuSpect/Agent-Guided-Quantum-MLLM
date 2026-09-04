"""QH039: quantum residual on a frozen, trained QH037 classical FFN scaffold."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping
import torch
from torch import Tensor,nn
from .qh037_grouped_qvaf_ffn_replacement import (
    DequantizedGroupedQVAFCore,EqualParameterClassicalQVAFCore,
    ExactGroupedQVAFCore,QH037Config,_Core,unique_trainable_parameters,
)

@dataclass(frozen=True)
class QH039Config(QH037Config):
    init_seed:int=20260868
    residual_alpha_init:float=1.0e-3

def load_qh037_cc_state(path:Path):
    obj=torch.load(path,map_location="cpu",weights_only=False)
    if obj.get("candidate")!="cc037":raise RuntimeError("QH039 requires the registered CC037 checkpoint")
    state=obj["replacement_state_dict"]
    for key in ("down.weight","core.theta","up.weight"):
        if key not in state:raise KeyError(key)
    return state,obj

class FrozenScaffoldResidualReplacement(nn.Module):
    def __init__(self,config:QH039Config,backbone_state:dict,residual_core:_Core,device:torch.device,dtype:torch.dtype):
        super().__init__();config.validate();self.config=config
        self.down=nn.Linear(config.hidden_size,config.latent_size,bias=False,device=device,dtype=dtype)
        self.base_core=EqualParameterClassicalQVAFCore(config).to(device)
        self.core=residual_core.to(device)
        self.alpha=nn.Parameter(torch.full((config.latent_size,),config.residual_alpha_init,device=device,dtype=torch.float32))
        self.up=nn.Linear(config.latent_size,config.hidden_size,bias=False,device=device,dtype=dtype)
        with torch.no_grad():
            self.down.weight.copy_(backbone_state["down.weight"].to(device=device,dtype=dtype))
            self.base_core.theta.copy_(backbone_state["core.theta"].to(device=device,dtype=torch.float32))
            self.up.weight.copy_(backbone_state["up.weight"].to(device=device,dtype=dtype))
        self.down.requires_grad_(False);self.base_core.requires_grad_(False);self.up.requires_grad_(False);self.correction_enabled=True
    def forward(self,x:Tensor):
        latent=self.down(x);base=self.base_core(latent)
        if not self.correction_enabled:return self.up(base)
        correction=self.core(latent).float()*self.alpha
        return self.up((base.float()+correction).to(base.dtype))

def qh039_parameter_audit(config:QH039Config)->MutableMapping[str,int]:
    removed=3*config.hidden_size*config.intermediate_size;low_rank=2*config.hidden_size*config.latent_size;base_core=config.circuit_parameter_count;residual_core=config.circuit_parameter_count;alpha=config.latent_size;deployed=low_rank+base_core+residual_core+alpha
    return {"original_parameters_removed":removed,"low_rank_parameters":low_rank,"frozen_classical_core_parameters":base_core,"residual_circuit_or_control_parameters":residual_core,"residual_alpha_parameters":alpha,"trainable_parameters":residual_core+alpha,"total_replacement_parameters":deployed,"net_parameter_reduction":removed-deployed}

def install_qh039(model,backbone_checkpoint:Path,config:QH039Config=QH039Config(),branch:str="quantum"):
    config.validate();model.requires_grad_(False);layer=model.model.language_model.layers[config.target_layer];original=layer.mlp;device=next(original.parameters()).device;dtype=next(original.parameters()).dtype
    classes={"quantum":ExactGroupedQVAFCore,"dq":DequantizedGroupedQVAFCore,"classical":EqualParameterClassicalQVAFCore,"no_ent":lambda cfg:ExactGroupedQVAFCore(cfg,use_entanglers=False)}
    if branch not in classes:raise ValueError(f"unknown QH039 branch {branch}")
    state,meta=load_qh037_cc_state(backbone_checkpoint);base=sum(p.numel() for p in model.parameters());replacement=FrozenScaffoldResidualReplacement(config,state,classes[branch](config),device,dtype);layer.mlp=replacement;deployed=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad);audit=qh039_parameter_audit(config)
    if base-deployed!=audit["net_parameter_reduction"] or trainable!=audit["trainable_parameters"]:raise AssertionError("QH039 parameter audit mismatch")
    return replacement,original,{**audit,"base_model_parameters":base,"deployed_model_parameters":deployed,"net_model_parameter_reduction":audit["net_parameter_reduction"],"trainable_parameter_count":trainable,"branch":branch,"target_layer":config.target_layer,"backbone_candidate":meta["candidate"],"backbone_lock_sha256":meta.get("lock_sha256"),"backbone_steps":meta.get("steps")}

__all__=["QH039Config","FrozenScaffoldResidualReplacement","install_qh039","load_qh037_cc_state","qh039_parameter_audit","unique_trainable_parameters"]
