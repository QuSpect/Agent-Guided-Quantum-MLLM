"""QH042: QH041 with signed nearest-neighbour Pauli-X tensor Pauli-Z readout."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import Tensor
from .qh037_grouped_qvaf_ffn_replacement import EqualParameterClassicalQVAFCore,ExactGroupedQVAFCore,_require_cuda
from .qh039_frozen_scaffold_quantum_residual import load_qh037_cc_state,unique_trainable_parameters
from .qh040_bounded_quantum_residual import BoundedScaffoldResidualReplacement,QH040Config,qh040_parameter_audit
from .qh041_signed_x_readout_residual import DequantizedGroupedXReadoutCore

@dataclass(frozen=True)
class QH042Config(QH040Config):
    init_seed:int=20260874

class _XZReadout:
    @staticmethod
    def _xz(state:Tensor,config,owner):
        vals=[]
        for q in range(config.num_qubits):
            z=getattr(owner,f"z_{q}");o=getattr(owner,f"o_{q}");nxt=(q+1)%config.num_qubits
            sign=torch.where(z.bitwise_and(1<<nxt)==0,1.0,-1.0).to(state.device)
            vals.append(2.0*(state.index_select(-1,z)*state.index_select(-1,o)*sign).sum(-1))
        return torch.stack(vals,-1)

class ExactGroupedXZReadoutCore(ExactGroupedQVAFCore,_XZReadout):
    def forward(self,latent:Tensor):
        _require_cuda(latent);shape,v,encoded=self._inputs(latent);n=v.shape[0];state=torch.zeros(n,self.config.groups,self.state_dimension,device=latent.device,dtype=torch.float32);state[...,0]=1;state=state.reshape(n*self.config.groups,self.state_dimension)
        for layer in range(self.config.depth):
            for q in range(self.config.num_qubits):
                state=self._rotate(state,getattr(self,f"z_{q}"),getattr(self,f"o_{q}"),encoded[...,q].reshape(-1));angle=self.theta[layer,:,q,0].unsqueeze(0).expand(n,-1).reshape(-1);state=self._rotate(state,getattr(self,f"z_{q}"),getattr(self,f"o_{q}"),angle)
            if self.use_entanglers:
                for q in range(self.config.num_qubits):
                    angle=self.theta[layer,:,q,1].unsqueeze(0).expand(n,-1).reshape(-1);state=self._rotate(state,getattr(self,f"cz_{q}"),getattr(self,f"co_{q}"),angle)
        out=self._xz(state,self.config,self).reshape(n,self.config.latent_size);self.circuit_calls.add_(1);return out.reshape(shape).to(latent.dtype)

class DequantizedGroupedXZReadoutCore(DequantizedGroupedXReadoutCore,_XZReadout):
    def forward(self,latent:Tensor):
        _require_cuda(latent);shape,v,encoded=self._inputs(latent);n=v.shape[0];state=torch.zeros(n*self.config.groups,*([2]*self.config.num_qubits),device=latent.device,dtype=torch.float32);state[(slice(None),)+(0,)*self.config.num_qubits]=1
        for layer in range(self.config.depth):
            for q in range(self.config.num_qubits):
                state=self._single(state,q,encoded[...,q].reshape(-1));angle=self.theta[layer,:,q,0].unsqueeze(0).expand(n,-1).reshape(-1);state=self._single(state,q,angle)
            if self.use_entanglers:
                for q in range(self.config.num_qubits):
                    angle=self.theta[layer,:,q,1].unsqueeze(0).expand(n,-1).reshape(-1);state=self._controlled(state,q,(q+1)%self.config.num_qubits,angle)
        flat=state.reshape(n*self.config.groups,-1);basis=torch.arange(1<<self.config.num_qubits,device=latent.device)
        vals=[]
        for q in range(self.config.num_qubits):
            z=basis[(basis.bitwise_and(1<<q))==0];o=z.bitwise_or(1<<q);nxt=(q+1)%self.config.num_qubits;sign=torch.where(z.bitwise_and(1<<nxt)==0,1.0,-1.0)
            vals.append(2.0*(flat.index_select(-1,z)*flat.index_select(-1,o)*sign).sum(-1))
        out=torch.stack(vals,-1).reshape(n,self.config.latent_size);self.circuit_calls.add_(1);return out.reshape(shape).to(latent.dtype)

def qh042_parameter_audit(config:QH042Config):return qh040_parameter_audit(config)
def install_qh042(model,backbone_checkpoint:Path,config:QH042Config=QH042Config(),branch:str="quantum"):
    config.validate();model.requires_grad_(False);layer=model.model.language_model.layers[config.target_layer];original=layer.mlp;device=next(original.parameters()).device;dtype=next(original.parameters()).dtype
    classes={"quantum":ExactGroupedXZReadoutCore,"dq":DequantizedGroupedXZReadoutCore,"classical":EqualParameterClassicalQVAFCore,"no_ent":lambda cfg:ExactGroupedXZReadoutCore(cfg,use_entanglers=False)}
    if branch not in classes:raise ValueError(branch)
    state,meta=load_qh037_cc_state(backbone_checkpoint);base=sum(p.numel() for p in model.parameters());rep=BoundedScaffoldResidualReplacement(config,state,classes[branch](config),device,dtype);layer.mlp=rep;deployed=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad);audit=qh042_parameter_audit(config)
    if base-deployed!=audit["net_parameter_reduction"] or trainable!=audit["trainable_parameters"]:raise AssertionError("QH042 parameter audit mismatch")
    return rep,original,{**audit,"base_model_parameters":base,"deployed_model_parameters":deployed,"net_model_parameter_reduction":audit["net_parameter_reduction"],"trainable_parameter_count":trainable,"branch":branch,"target_layer":config.target_layer,"backbone_candidate":meta["candidate"],"observable":"nearest-neighbour Pauli-X tensor Pauli-Z","effective_alpha_cap_per_coordinate":config.residual_alpha_cap}

__all__=["QH042Config","ExactGroupedXZReadoutCore","DequantizedGroupedXZReadoutCore","install_qh042","qh042_parameter_audit","unique_trainable_parameters"]
