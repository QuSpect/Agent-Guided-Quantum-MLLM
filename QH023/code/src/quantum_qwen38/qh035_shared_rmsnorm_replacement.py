"""QH035: shared RMSNorm replacement with a runtime 13-qubit residual.

Four Qwen3.8 post-attention RMSNorm modules share one frozen effective scale.
Their normalized token vectors are padded from 5120 to 8192 amplitudes, passed
through a shallow exact CUDA statevector circuit, cropped back to 5120, and
fused through a layer-specific scalar. Zero angles exactly reproduce the
shared classical RMSNorm. The original per-layer Norm parameters are removed.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Tuple
import torch
from torch import Tensor, nn

@dataclass(frozen=True)
class QH035Config:
    hidden_size:int=5120
    state_dimension:int=8192
    layer_indices:Tuple[int,...]=(11,12,13,15)
    norm_name:str="post_attention_layernorm"
    num_qubits:int=13
    depth:int=2
    entangling_offsets:Tuple[int,...]=(1,2,4)
    eps:float=1.0e-6
    def validate(self):
        if self.state_dimension != 1<<self.num_qubits: raise ValueError("state_dimension must equal 2**num_qubits")
        if self.hidden_size>self.state_dimension: raise ValueError("hidden_size exceeds state dimension")
        if len(self.layer_indices)<2: raise ValueError("QH035 requires shared layers")
        if any(not 0<o<self.num_qubits for o in self.entangling_offsets): raise ValueError("invalid offset")
    @property
    def circuit_parameter_count(self): return self.depth*self.num_qubits*(1+len(self.entangling_offsets))
def _require_cuda(x):
    if x.device.type!="cuda": raise RuntimeError(f"QH035 exact simulation is GPU-only; received {x.device}")
def _ry(theta):
    h=theta*.5; c,s=torch.cos(h),torch.sin(h); return torch.stack((torch.stack((c,-s)),torch.stack((s,c))))
class _Core(nn.Module):
    def __init__(self,config):
        super().__init__(); config.validate(); self.config=config
        self.theta=nn.Parameter(torch.zeros(config.depth,config.num_qubits,1+len(config.entangling_offsets),dtype=torch.float32))
        self.register_buffer("circuit_calls",torch.zeros((),dtype=torch.long),persistent=False)
    @property
    def parameter_count(self): return self.theta.numel()
    def reset_circuit_calls(self): self.circuit_calls.zero_()
    def _pad_normalize(self,values):
        flat=values.reshape(-1,self.config.hidden_size).float()
        padded=torch.nn.functional.pad(flat,(0,self.config.state_dimension-self.config.hidden_size))
        norm=padded.square().sum(-1,keepdim=True).clamp_min(1e-12).sqrt()
        return flat,padded/norm,norm
class ExactQuantumNormCore(_Core):
    def __init__(self,config,use_entanglers=True):
        super().__init__(config); self.use_entanglers=use_entanglers; basis=torch.arange(config.state_dimension,dtype=torch.long)
        for w in range(config.num_qubits):
            z=basis[(basis.bitwise_and(1<<w))==0]; self.register_buffer(f"z_{w}",z,persistent=False); self.register_buffer(f"o_{w}",z.bitwise_or(1<<w),persistent=False)
        for c in range(config.num_qubits):
            for offset in config.entangling_offsets:
                t=(c+offset)%config.num_qubits; mask=((basis.bitwise_and(1<<c))!=0)&((basis.bitwise_and(1<<t))==0)
                z=basis[mask]; self.register_buffer(f"cz_{c}_{offset}",z,persistent=False); self.register_buffer(f"co_{c}_{offset}",z.bitwise_or(1<<t),persistent=False)
    @staticmethod
    def _rotate(state,z,o,theta):
        a0=state.index_select(-1,z); a1=state.index_select(-1,o); h=theta*.5; c,s=torch.cos(h),torch.sin(h); out=state.clone()
        out.index_copy_(-1,z,c*a0-s*a1); out.index_copy_(-1,o,s*a0+c*a1); return out
    def forward(self,values):
        _require_cuda(values); shape=values.shape; _,state,norm=self._pad_normalize(values)
        for layer in range(self.config.depth):
            for w in range(self.config.num_qubits): state=self._rotate(state,getattr(self,f"z_{w}"),getattr(self,f"o_{w}"),self.theta[layer,w,0])
            if self.use_entanglers:
                for oi,offset in enumerate(self.config.entangling_offsets):
                    for c in range(self.config.num_qubits): state=self._rotate(state,getattr(self,f"cz_{c}_{offset}"),getattr(self,f"co_{c}_{offset}"),self.theta[layer,c,1+oi])
        self.circuit_calls.add_(1); out=(state*norm)[...,:self.config.hidden_size]
        return out.reshape(shape).to(values.dtype)
class DequantizedNormCore(_Core):
    def __init__(self,config,use_entanglers=True): super().__init__(config); self.use_entanglers=use_entanglers
    def _single(self,state,w,theta):
        axis=1+(self.config.num_qubits-1-w); moved=state.movedim(axis,-1); return torch.matmul(moved,_ry(theta).transpose(0,1)).movedim(-1,axis)
    def _controlled(self,state,c,t,theta):
        ca=1+(self.config.num_qubits-1-c); ta=1+(self.config.num_qubits-1-t); moved=state.movedim((ca,ta),(-2,-1)); shape=moved.shape
        pairs=moved.reshape(*shape[:-2],4); gate=torch.eye(4,dtype=state.dtype,device=state.device); gate[2:,2:]=_ry(theta).to(state)
        return torch.matmul(pairs,gate.transpose(0,1)).reshape(shape).movedim((-2,-1),(ca,ta))
    def forward(self,values):
        _require_cuda(values); shape=values.shape; _,state,norm=self._pad_normalize(values); state=state.reshape(-1,*([2]*self.config.num_qubits))
        for layer in range(self.config.depth):
            for w in range(self.config.num_qubits): state=self._single(state,w,self.theta[layer,w,0])
            if self.use_entanglers:
                for oi,offset in enumerate(self.config.entangling_offsets):
                    for c in range(self.config.num_qubits): state=self._controlled(state,c,(c+offset)%self.config.num_qubits,self.theta[layer,c,1+oi])
        self.circuit_calls.add_(1); out=state.reshape(-1,self.config.state_dimension)*norm
        return out[...,:self.config.hidden_size].reshape(shape).to(values.dtype)
class EqualParameterClassicalNormCore(_Core):
    def __init__(self,config):
        super().__init__(config); basis=torch.arange(config.state_dimension,dtype=torch.long)
        for w in range(config.num_qubits):
            self.register_buffer(f"sign_{w}",torch.where(basis.bitwise_and(1<<w)==0,1.0,-1.0),persistent=False)
            for offset in config.entangling_offsets: self.register_buffer(f"perm_{w}_{offset}",basis.bitwise_xor(1<<((w+offset)%config.num_qubits)),persistent=False)
    def forward(self,values):
        _require_cuda(values); shape=values.shape; _,state,norm=self._pad_normalize(values); denom=float(self.config.num_qubits*len(self.config.entangling_offsets))**.5
        for layer in range(self.config.depth):
            for w in range(self.config.num_qubits): state=state+torch.tanh(self.theta[layer,w,0])*state*getattr(self,f"sign_{w}")/denom
            for oi,offset in enumerate(self.config.entangling_offsets):
                for w in range(self.config.num_qubits): state=state+torch.tanh(self.theta[layer,w,1+oi])*state.index_select(-1,getattr(self,f"perm_{w}_{offset}"))/denom
        out=state*norm; return out[...,:self.config.hidden_size].reshape(shape).to(values.dtype)
class SharedEffectiveScale(nn.Module):
    def __init__(self,scale): super().__init__(); self.scale=nn.Parameter(scale.detach().clone(),requires_grad=False)
class SharedQuantumRMSNorm(nn.Module):
    def __init__(self,shared,core,eps):
        super().__init__(); self.shared=shared; self.core=core; self.eps=float(eps); self.gamma=nn.Parameter(torch.ones((),dtype=torch.float32)); self.branch_enabled=True
    def forward(self,x):
        normalized=x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+self.eps)
        base=(normalized*self.shared.scale.float()).type_as(x)
        if self.branch_enabled:
            transformed=self.core(base); base=base+self.gamma.to(base.dtype)*(transformed-base)
        return base
def build_qh035_replacements(norms:Mapping[int,nn.Module],config:QH035Config,branch="quantum")->Dict[int,SharedQuantumRMSNorm]:
    config.validate();
    if tuple(norms)!=config.layer_indices: raise ValueError("norm order mismatch")
    ordered=[norms[i] for i in config.layer_indices]; device=ordered[0].weight.device; _require_cuda(ordered[0].weight)
    scales=[]
    for module in ordered:
        if module.weight.numel()!=config.hidden_size: raise ValueError("norm shape mismatch")
        if abs(float(module.eps)-config.eps)>1e-12: raise ValueError("norm eps mismatch")
        scales.append(1.0+module.weight.detach().float())
    shared=SharedEffectiveScale(torch.stack(scales).mean(0)).to(device)
    classes={"quantum":ExactQuantumNormCore,"dq":DequantizedNormCore,"classical":EqualParameterClassicalNormCore,
      "no_ent":lambda cfg:ExactQuantumNormCore(cfg,use_entanglers=False)}
    if branch not in classes: raise ValueError(f"unknown QH035 branch {branch}")
    core=classes[branch](config).to(device); return {i:SharedQuantumRMSNorm(shared,core,config.eps).to(device) for i in config.layer_indices}
def qh035_parameter_audit(config)->MutableMapping[str,int]:
    config.validate(); removed=len(config.layer_indices)*config.hidden_size; frozen=config.hidden_size; trainable=config.circuit_parameter_count+len(config.layer_indices)
    return {"original_parameters_removed":removed,"frozen_replacement_parameters":frozen,"trainable_parameters":trainable,
      "total_replacement_parameters":frozen+trainable,"net_parameter_reduction":removed-frozen-trainable}
def install_qh035(model,config=QH035Config(),branch="quantum"):
    config.validate(); model.requires_grad_(False); layers=model.model.language_model.layers; originals={}
    for i in config.layer_indices: originals[i]=getattr(layers[i],config.norm_name)
    base=sum(p.numel() for p in model.parameters()); replacements=build_qh035_replacements(originals,config,branch)
    for i,m in replacements.items(): setattr(layers[i],config.norm_name,m)
    deployed=sum(p.numel() for p in model.parameters()); trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); audit=qh035_parameter_audit(config)
    if base-deployed!=audit["net_parameter_reduction"] or trainable!=audit["trainable_parameters"]: raise AssertionError("QH035 parameter audit mismatch")
    return replacements,originals,{**audit,"base_model_parameters":base,"deployed_model_parameters":deployed,"net_model_parameter_reduction":audit["net_parameter_reduction"],
      "trainable_parameter_count":trainable,"branch":branch,"layer_indices":list(config.layer_indices),"norm_name":config.norm_name,"hidden_size":config.hidden_size,
      "state_dimension":config.state_dimension,"num_qubits":config.num_qubits,"depth":config.depth,"entangling_offsets":list(config.entangling_offsets)}
def unique_trainable_parameters(replacements):
    seen=set()
    for module in replacements.values():
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen: seen.add(id(p)); yield p
