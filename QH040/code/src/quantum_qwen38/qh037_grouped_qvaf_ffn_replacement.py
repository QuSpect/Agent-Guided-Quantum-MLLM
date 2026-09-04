"""QH037: replace one complete Qwen3.8 SwiGLU FFN with a grouped QVAF bottleneck.

The replacement deletes layer 31's three 5120x17408/17408x5120 matrices.
A 5120->64->5120 low-rank path surrounds sixteen independent four-qubit,
depth-two data-reuploading circuits.  Every active forward executes the exact
real statevector simulator on CUDA.  Branch-off is an exact zero FFN.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import MutableMapping
import torch
from torch import Tensor,nn

@dataclass(frozen=True)
class QH037Config:
    hidden_size:int=5120
    intermediate_size:int=17408
    target_layer:int=31
    latent_size:int=64
    group_size:int=4
    num_qubits:int=4
    depth:int=2
    eps:float=1.0e-6
    init_seed:int=20260864
    def validate(self):
        if self.group_size!=self.num_qubits:raise ValueError("one latent scalar per qubit is required")
        if self.latent_size%self.group_size:raise ValueError("latent_size must divide into groups")
        if self.num_qubits<2:raise ValueError("QH037 requires entanglement-capable groups")
    @property
    def groups(self):return self.latent_size//self.group_size
    @property
    def circuit_parameter_count(self):return self.depth*self.groups*self.num_qubits*2

def _require_cuda(x:Tensor):
    if x.device.type!="cuda":raise RuntimeError(f"QH037 exact simulation is GPU-only; received {x.device}")

class _Core(nn.Module):
    def __init__(self,config:QH037Config):
        super().__init__();config.validate();self.config=config
        self.theta=nn.Parameter(torch.zeros(config.depth,config.groups,config.num_qubits,2,dtype=torch.float32))
        self.register_buffer("circuit_calls",torch.zeros((),dtype=torch.long),persistent=False)
    def _inputs(self,latent:Tensor):
        shape=latent.shape;v=latent.reshape(-1,self.config.groups,self.config.group_size).float()
        v=v*torch.rsqrt(v.square().mean(-1,keepdim=True)+self.config.eps)
        return shape,v,2.0*torch.atan(v)

class ExactGroupedQVAFCore(_Core):
    def __init__(self,config:QH037Config,use_entanglers:bool=True):
        super().__init__(config);self.use_entanglers=use_entanglers;dim=1<<config.num_qubits;basis=torch.arange(dim,dtype=torch.long)
        self.state_dimension=dim
        for q in range(config.num_qubits):
            z=basis[(basis.bitwise_and(1<<q))==0];self.register_buffer(f"z_{q}",z,persistent=False);self.register_buffer(f"o_{q}",z.bitwise_or(1<<q),persistent=False)
            t=(q+1)%config.num_qubits;mask=((basis.bitwise_and(1<<q))!=0)&((basis.bitwise_and(1<<t))==0);cz=basis[mask]
            self.register_buffer(f"cz_{q}",cz,persistent=False);self.register_buffer(f"co_{q}",cz.bitwise_or(1<<t),persistent=False)
        signs=[]
        for q in range(config.num_qubits):signs.append(torch.where(basis.bitwise_and(1<<q)==0,1.0,-1.0))
        self.register_buffer("z_signs",torch.stack(signs),persistent=False)
    @staticmethod
    def _rotate(state:Tensor,z:Tensor,o:Tensor,theta:Tensor):
        a0=state.index_select(-1,z);a1=state.index_select(-1,o);c=torch.cos(theta*.5).unsqueeze(-1);s=torch.sin(theta*.5).unsqueeze(-1);out=state.clone()
        out.index_copy_(-1,z,c*a0-s*a1);out.index_copy_(-1,o,s*a0+c*a1);return out
    def forward(self,latent:Tensor):
        _require_cuda(latent);shape,v,encoded=self._inputs(latent);n=v.shape[0];state=torch.zeros(n,self.config.groups,self.state_dimension,device=latent.device,dtype=torch.float32);state[...,0]=1
        state=state.reshape(n*self.config.groups,self.state_dimension)
        for layer in range(self.config.depth):
            for q in range(self.config.num_qubits):
                state=self._rotate(state,getattr(self,f"z_{q}"),getattr(self,f"o_{q}"),encoded[...,q].reshape(-1))
                angle=self.theta[layer,:,q,0].unsqueeze(0).expand(n,-1).reshape(-1);state=self._rotate(state,getattr(self,f"z_{q}"),getattr(self,f"o_{q}"),angle)
            if self.use_entanglers:
                for q in range(self.config.num_qubits):
                    angle=self.theta[layer,:,q,1].unsqueeze(0).expand(n,-1).reshape(-1);state=self._rotate(state,getattr(self,f"cz_{q}"),getattr(self,f"co_{q}"),angle)
        probs=state.square();out=torch.matmul(probs,self.z_signs.transpose(0,1)).reshape(n,self.config.latent_size);self.circuit_calls.add_(1)
        return out.reshape(shape).to(latent.dtype)

class DequantizedGroupedQVAFCore(_Core):
    def __init__(self,config:QH037Config,use_entanglers:bool=True):super().__init__(config);self.use_entanglers=use_entanglers
    @staticmethod
    def _rot_pair(a0:Tensor,a1:Tensor,theta:Tensor):
        view=(theta.shape[0],)+(1,)*(a0.ndim-1);c=torch.cos(theta*.5).view(view);s=torch.sin(theta*.5).view(view);return c*a0-s*a1,s*a0+c*a1
    def _single(self,state:Tensor,q:int,theta:Tensor):
        axis=1+(self.config.num_qubits-1-q);m=state.movedim(axis,-1);a0,a1=self._rot_pair(m[...,0],m[...,1],theta);return torch.stack((a0,a1),-1).movedim(-1,axis)
    def _controlled(self,state:Tensor,cq:int,tq:int,theta:Tensor):
        ca=1+(self.config.num_qubits-1-cq);ta=1+(self.config.num_qubits-1-tq);m=state.movedim((ca,ta),(-2,-1));a0,a1=self._rot_pair(m[...,1,0],m[...,1,1],theta);out=torch.stack((m[...,0,:],torch.stack((a0,a1),-1)),-2);return out.movedim((-2,-1),(ca,ta))
    def forward(self,latent:Tensor):
        _require_cuda(latent);shape,v,encoded=self._inputs(latent);n=v.shape[0];state=torch.zeros(n*self.config.groups,*([2]*self.config.num_qubits),device=latent.device,dtype=torch.float32);state[(slice(None),)+(0,)*self.config.num_qubits]=1
        for layer in range(self.config.depth):
            for q in range(self.config.num_qubits):
                state=self._single(state,q,encoded[...,q].reshape(-1));angle=self.theta[layer,:,q,0].unsqueeze(0).expand(n,-1).reshape(-1);state=self._single(state,q,angle)
            if self.use_entanglers:
                for q in range(self.config.num_qubits):
                    angle=self.theta[layer,:,q,1].unsqueeze(0).expand(n,-1).reshape(-1);state=self._controlled(state,q,(q+1)%self.config.num_qubits,angle)
        probs=state.reshape(n*self.config.groups,-1).square();basis=torch.arange(1<<self.config.num_qubits,device=latent.device)
        signs=torch.stack([torch.where(basis.bitwise_and(1<<q)==0,1.0,-1.0) for q in range(self.config.num_qubits)]);out=torch.matmul(probs,signs.transpose(0,1)).reshape(n,self.config.latent_size);self.circuit_calls.add_(1)
        return out.reshape(shape).to(latent.dtype)

class EqualParameterClassicalQVAFCore(_Core):
    def forward(self,latent:Tensor):
        _require_cuda(latent);shape,v,_=self._inputs(latent)
        for layer in range(self.config.depth):
            a=torch.tanh(self.theta[layer,...,0]).unsqueeze(0);b=torch.tanh(self.theta[layer,...,1]).unsqueeze(0);rolled=torch.roll(v,1,dims=-1)
            v=torch.tanh(v+a*torch.sin(v)+b*torch.sin(v*rolled))
        return v.reshape(shape).to(latent.dtype)

class GroupedQVAFReplacement(nn.Module):
    def __init__(self,config:QH037Config,core:_Core,device:torch.device,dtype:torch.dtype):
        super().__init__();self.config=config;self.down=nn.Linear(config.hidden_size,config.latent_size,bias=False,device=device,dtype=dtype);self.core=core.to(device);self.up=nn.Linear(config.latent_size,config.hidden_size,bias=False,device=device,dtype=dtype);self.branch_enabled=True
        gen=torch.Generator(device=device);gen.manual_seed(config.init_seed)
        with torch.no_grad():
            self.down.weight.normal_(0.0,1.0/(config.hidden_size**.5),generator=gen);self.up.weight.normal_(0.0,1.0e-4,generator=gen)
    def forward(self,x:Tensor):
        if not self.branch_enabled:return torch.zeros_like(x)
        return self.up(self.core(self.down(x)))

def qh037_parameter_audit(config:QH037Config)->MutableMapping[str,int]:
    config.validate();removed=3*config.hidden_size*config.intermediate_size;low_rank=2*config.hidden_size*config.latent_size;core=config.circuit_parameter_count;deployed=low_rank+core
    return {"original_parameters_removed":removed,"low_rank_parameters":low_rank,"circuit_or_control_parameters":core,"trainable_parameters":deployed,"total_replacement_parameters":deployed,"net_parameter_reduction":removed-deployed}

def install_qh037(model,config:QH037Config=QH037Config(),branch:str="quantum"):
    config.validate();model.requires_grad_(False);layer=model.model.language_model.layers[config.target_layer];original=layer.mlp;device=next(original.parameters()).device;dtype=next(original.parameters()).dtype
    classes={"quantum":ExactGroupedQVAFCore,"dq":DequantizedGroupedQVAFCore,"classical":EqualParameterClassicalQVAFCore,"no_ent":lambda cfg:ExactGroupedQVAFCore(cfg,use_entanglers=False)}
    if branch not in classes:raise ValueError(f"unknown QH037 branch {branch}")
    base=sum(p.numel() for p in model.parameters());replacement=GroupedQVAFReplacement(config,classes[branch](config),device,dtype);layer.mlp=replacement;deployed=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad);audit=qh037_parameter_audit(config)
    if base-deployed!=audit["net_parameter_reduction"] or trainable!=audit["trainable_parameters"]:raise AssertionError("QH037 parameter audit mismatch")
    return replacement,original,{**audit,"base_model_parameters":base,"deployed_model_parameters":deployed,"net_model_parameter_reduction":audit["net_parameter_reduction"],"trainable_parameter_count":trainable,"branch":branch,"target_layer":config.target_layer,"latent_size":config.latent_size,"groups":config.groups,"group_size":config.group_size,"num_qubits_per_group":config.num_qubits,"depth":config.depth}

def unique_trainable_parameters(module:nn.Module):return (p for p in module.parameters() if p.requires_grad)
