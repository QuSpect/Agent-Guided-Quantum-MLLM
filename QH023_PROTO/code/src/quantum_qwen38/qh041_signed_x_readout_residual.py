"""QH041: bounded QH040 residual with signed local-X quantum readout."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import Tensor
from .qh037_grouped_qvaf_ffn_replacement import DequantizedGroupedQVAFCore,EqualParameterClassicalQVAFCore,ExactGroupedQVAFCore,_require_cuda
from .qh039_frozen_scaffold_quantum_residual import load_qh037_cc_state,unique_trainable_parameters
from .qh040_bounded_quantum_residual import BoundedScaffoldResidualReplacement,QH040Config,qh040_parameter_audit

@dataclass(frozen=True)
class QH041Config(QH040Config):
    init_seed:int=20260872

class ExactGroupedXReadoutCore(ExactGroupedQVAFCore):
    def forward(self,latent:Tensor):
        _require_cuda(latent);shape,v,encoded=self._inputs(latent);n=v.shape[0];state=torch.zeros(n,self.config.groups,self.state_dimension,device=latent.device,dtype=torch.float32);state[...,0]=1;state=state.reshape(n*self.config.groups,self.state_dimension)
        for layer in range(self.config.depth):
            for q in range(self.config.num_qubits):
                state=self._rotate(state,getattr(self,f"z_{q}"),getattr(self,f"o_{q}"),encoded[...,q].reshape(-1));angle=self.theta[layer,:,q,0].unsqueeze(0).expand(n,-1).reshape(-1);state=self._rotate(state,getattr(self,f"z_{q}"),getattr(self,f"o_{q}"),angle)
            if self.use_entanglers:
                for q in range(self.config.num_qubits):
                    angle=self.theta[layer,:,q,1].unsqueeze(0).expand(n,-1).reshape(-1);state=self._rotate(state,getattr(self,f"cz_{q}"),getattr(self,f"co_{q}"),angle)
        out=torch.stack([2.0*(state.index_select(-1,getattr(self,f"z_{q}"))*state.index_select(-1,getattr(self,f"o_{q}"))).sum(-1) for q in range(self.config.num_qubits)],-1).reshape(n,self.config.latent_size);self.circuit_calls.add_(1);return out.reshape(shape).to(latent.dtype)

class DequantizedGroupedXReadoutCore(DequantizedGroupedQVAFCore):
    def forward(self,latent:Tensor):
        _require_cuda(latent);shape,v,encoded=self._inputs(latent);n=v.shape[0];state=torch.zeros(n*self.config.groups,*([2]*self.config.num_qubits),device=latent.device,dtype=torch.float32);state[(slice(None),)+(0,)*self.config.num_qubits]=1
        for layer in range(self.config.depth):
            for q in range(self.config.num_qubits):
                state=self._single(state,q,encoded[...,q].reshape(-1));angle=self.theta[layer,:,q,0].unsqueeze(0).expand(n,-1).reshape(-1);state=self._single(state,q,angle)
            if self.use_entanglers:
                for q in range(self.config.num_qubits):
                    angle=self.theta[layer,:,q,1].unsqueeze(0).expand(n,-1).reshape(-1);state=self._controlled(state,q,(q+1)%self.config.num_qubits,angle)
        vals=[]
        for q in range(self.config.num_qubits):
            axis=1+(self.config.num_qubits-1-q);m=state.movedim(axis,-1);prod=2.0*m[...,0]*m[...,1];vals.append(prod.reshape(prod.shape[0],-1).sum(-1))
        out=torch.stack(vals,-1).reshape(n,self.config.latent_size);self.circuit_calls.add_(1);return out.reshape(shape).to(latent.dtype)

def qh041_parameter_audit(config:QH041Config):return qh040_parameter_audit(config)
def install_qh041(model,backbone_checkpoint:Path,config:QH041Config=QH041Config(),branch:str="quantum"):
    config.validate();model.requires_grad_(False);layer=model.model.language_model.layers[config.target_layer];original=layer.mlp;device=next(original.parameters()).device;dtype=next(original.parameters()).dtype;classes={"quantum":ExactGroupedXReadoutCore,"dq":DequantizedGroupedXReadoutCore,"classical":EqualParameterClassicalQVAFCore,"no_ent":lambda cfg:ExactGroupedXReadoutCore(cfg,use_entanglers=False)}
    if branch not in classes:raise ValueError(branch)
    state,meta=load_qh037_cc_state(backbone_checkpoint);base=sum(p.numel() for p in model.parameters());rep=BoundedScaffoldResidualReplacement(config,state,classes[branch](config),device,dtype);layer.mlp=rep;deployed=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad);audit=qh041_parameter_audit(config)
    if base-deployed!=audit["net_parameter_reduction"] or trainable!=audit["trainable_parameters"]:raise AssertionError("QH041 parameter audit mismatch")
    return rep,original,{**audit,"base_model_parameters":base,"deployed_model_parameters":deployed,"net_model_parameter_reduction":audit["net_parameter_reduction"],"trainable_parameter_count":trainable,"branch":branch,"target_layer":config.target_layer,"backbone_candidate":meta["candidate"],"backbone_steps":meta.get("steps"),"observable":"local Pauli-X","effective_alpha_cap_per_coordinate":config.residual_alpha_cap}

__all__=["QH041Config","ExactGroupedXReadoutCore","DequantizedGroupedXReadoutCore","install_qh041","qh041_parameter_audit","unique_trainable_parameters"]
