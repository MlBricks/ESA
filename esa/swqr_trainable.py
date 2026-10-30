"""MLBricks SWQR Fast/Mem -- trainable from scratch, self-contained PyTorch reference.

Preserves Original SWQR A/B/R recurrence and B+4 initialization; extends its
model to full-sequence causal logits and an *optional* trainable feature archive.
Mem is an EXPERIMENTAL general-feature archive, not the old trained 4-token-ID
synthetic archive. Its parameters require training for each new model/task.

SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 (SWQR-derived reference)
Copyright 2026 Zameer Hussain and Akhtar Hussain (original SWQR)
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import math
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_TIMESCALES = (16.,64.,256.,1024.,4096.,8192.)
VALID_SLOTS = (8,16,32,64,128)


@dataclass(frozen=True)
class SWQRConfig:
    vocab_size: int = 256
    dim: int = 384
    heads: int = 6
    layers: int = 6
    mode: str = "fast"                  # fast | mem
    slots: int = 0                       # 8|16|32|64|128 in mem mode
    memory_dim: int = 64                 # projected archival K/V width
    gate_rank: int = 16
    write_bias_init: float = 4.0
    read_bias_init: float = 4.0
    chunk_size: int = 32
    ffn_ratio: int = 4
    dropout: float = 0.

    def __post_init__(self):
        if self.mode not in ("fast", "mem"): raise ValueError("mode must be fast or mem")
        if self.mode == "mem" and self.slots not in VALID_SLOTS:
            raise ValueError(f"mem slots must be one of {VALID_SLOTS}")
        if self.mode == "fast" and self.slots != 0:
            raise ValueError("fast mode must have slots=0")
        if self.dim < 1 or self.heads < 1 or self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if min(self.vocab_size, self.layers, self.memory_dim, self.gate_rank, self.chunk_size) < 1:
            raise ValueError("vocab_size/layers/memory_dim/gate_rank/chunk_size must be positive")
        if self.ffn_ratio < 1: raise ValueError("ffn_ratio must be positive")
        if not 0<=self.dropout<1: raise ValueError("dropout must be in [0,1)")


def timescale_init(n_heads: int, *, device=None):
    """Reference six heads exactly; otherwise log-interpolate initial timescales."""
    if n_heads == 6: return torch.tensor(DEFAULT_TIMESCALES, device=device).log()
    x = torch.linspace(0, 1, n_heads,device=device)
    # Monotonic geometric spread for arbitrary head counts
    return math.log(16.) + x*(math.log(8192.)-math.log(16.))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.eps=eps
    def forward(self,x):
        return F.rms_norm(x,(x.shape[-1],),self.weight,self.eps)


def stable_scan(a, v, initial=None, chunk_size=32):
    """Differentiable division-free associative scan of E_t=A_t*E_(t-1)+V_t.

    Uses doubling within each bounded chunk; carries state between chunks.
    Both a and v are [batch,time,heads,channels].
    """
    b,t,h,d = a.shape
    if t == 0: raise ValueError("Empty sequence")
    state=(v.new_zeros((b,h,d)) if initial is None else initial).float()
    result=[]
    for st in range(0,t,chunk_size):
        aa=a[:,st:st+chunk_size].float(); uu=v[:,st:st+chunk_size].float()
        offset=1
        while offset < aa.shape[1]:
            prev_a = torch.cat((torch.ones_like(aa[:,:offset]),aa[:,:-offset]),1)
            prev_u = torch.cat((torch.zeros_like(uu[:,:offset]),uu[:,:-offset]),1)
            next_u=uu + aa*prev_u
            next_a=aa*prev_a
            aa,uu=next_a,next_u
            offset*=2
        chunk_states=aa*state.unsqueeze(1)+uu
        state=chunk_states[:,-1]
        result.append(chunk_states)
    return torch.cat(result,dim=1),state


class SWQRLayer(nn.Module):
    def __init__(self,c: SWQRConfig):
        super().__init__()
        self.dim=c.dim;self.heads=c.heads;self.dh=c.dim//c.heads;self.chunk_size=c.chunk_size
        self.W_A=nn.Linear(c.dim,c.dim,bias=False)
        self.W_V=nn.Linear(c.dim,c.dim,bias=False)
        self.W_O=nn.Linear(c.dim,c.dim,bias=False)
        self.state_norm=RMSNorm(c.dim)
        self.log_half_life=nn.Parameter(timescale_init(c.heads))
        self.raw_decay_bias=nn.Parameter(torch.full((c.heads,),math.log(math.expm1(1.))))
        self.write_down=nn.Linear(c.dim,c.gate_rank,bias=False)
        self.write_up=nn.Linear(c.gate_rank,c.dim,bias=False)
        self.write_bias=nn.Parameter(torch.full((c.heads,),c.write_bias_init))
        self.read_down=nn.Linear(c.dim,c.gate_rank,bias=False)
        self.read_up=nn.Linear(c.gate_rank,c.dim,bias=False)
        self.read_bias=nn.Parameter(torch.full((c.heads,),c.read_bias_init))
    def _gate(self,x,part):
        b,t,_=x.shape
        z=getattr(self,part+'_up')(F.silu(getattr(self,part+'_down')(x))).float()
        return (z.view(b,t,self.heads,self.dh)+getattr(self,part+'_bias')[None,None,:,None]).sigmoid()
    def forward(self,x,state=None):
        b,t,_=x.shape
        z=self.W_A(x).float().reshape(b,t,self.heads,self.dh)
        decay=F.softplus(z+self.raw_decay_bias[None,None,:,None])
        a=torch.exp(-math.log(2.)*decay/self.log_half_life.exp()[None,None,:,None])
        v=self.W_V(x).float().reshape(b,t,self.heads,self.dh)*self._gate(x,'write')
        if t==1:
            states=a*(torch.zeros_like(v) if state is None else state.float().unsqueeze(1))+v
            final=states[:,0]
        else:
            states,final=stable_scan(a,v,state,self.chunk_size)
        states=states*self._gate(x,'read')
        return self.W_O(self.state_norm(states.reshape(b,t,self.dim)).to(x.dtype)),final


class SWQRBlock(nn.Module):
    def __init__(self,c:SWQRConfig):
        super().__init__()
        # Legacy-compatible module naming (pre_esa, esa, pre_ffn, ffn)
        self.pre_esa=RMSNorm(c.dim)
        self.esa=SWQRLayer(c)
        self.pre_ffn=RMSNorm(c.dim)
        width=c.dim*c.ffn_ratio - 2*c.gate_rank
        if width<=0:raise ValueError("FFN too narrow; increase dim/ffn_ratio or decrease gate_rank")
        self.ffn=nn.Sequential(nn.Linear(c.dim,width),nn.GELU(),nn.Linear(width,c.dim))
        self.dropout=nn.Dropout(c.dropout)
    def forward(self,x,state=None):
        y,state=self.esa(self.pre_esa(x),state)
        x=x+self.dropout(y)
        return x+self.dropout(self.ffn(self.pre_ffn(x))),state


@dataclass
class ArchiveState:
    keys: torch.Tensor            # [batch,K,memory_dim]
    values: torch.Tensor          # [batch,K,memory_dim]
    salience: torch.Tensor        # [batch,K]
    filled: torch.Tensor          # [batch,K], boolean
    positions: torch.Tensor       # [batch,K], long
    position: int                 # processed tokens
    def detach(self):
        return ArchiveState(self.keys.detach(),self.values.detach(),self.salience.detach(),
                            self.filled.detach(),self.positions.detach(),self.position)


class LearnedArchive(nn.Module):
    """Optional causal hidden-state archive, trainable from scratch.

    Hard top-K selection is discrete: gradients pass through stored values and
    salience, but *not* through rejected indices. Optional salience supervision
    can teach selection; this is not a fully differentiable top-K algorithm.
    """
    def __init__(self,c:SWQRConfig):
        super().__init__()
        d=c.dim; m=c.memory_dim
        self.slots=c.slots;self.memory_dim=m
        self.score=nn.Linear(d,1)
        self.key=nn.Linear(d,m,bias=False)
        self.value=nn.Linear(d,m,bias=False)
        self.query=nn.Linear(d,m,bias=False)
        self.out=nn.Linear(m,d,bias=False)
        self.alpha=nn.Parameter(torch.tensor(-2.)) # sigmoid(alpha) initial ~0.119
    def empty(self,b:int,device,dtype):
        k=self.slots;m=self.memory_dim
        return ArchiveState(torch.zeros(b,k,m,device=device,dtype=dtype),
             torch.zeros(b,k,m,device=device,dtype=dtype),
             torch.full((b,k),-1e4,device=device,dtype=dtype),
             torch.zeros(b,k,device=device,dtype=torch.bool),
             torch.full((b,k),-1,device=device,dtype=torch.long),0)
    def read(self,x,state:ArchiveState):
        q=F.normalize(self.query(x),dim=-1)
        k=F.normalize(state.keys,dim=-1,eps=1e-6)
        scores=torch.einsum('bd,bkd->bk',q,k)*math.sqrt(self.memory_dim)
        scores=scores+state.salience.sigmoid().log().nan_to_num(neginf=-100.)
        scores=scores.masked_fill(~state.filled,-1e4)
        att=F.softmax(scores.float(),dim=-1).to(state.values.dtype)
        # Empty slots hold zero values, so an empty archive reads as zero
        # without a GPU-to-CPU sync (important for streaming decode).
        ctx=torch.einsum('bk,bkd->bd',att,state.values)
        return self.out(ctx)*self.alpha.sigmoid()
    def write(self,x,state:ArchiveState):
        raw_score=self.score(x).squeeze(-1)
        k=self.key(x).unsqueeze(1)
        v=self.value(x).unsqueeze(1)
        s=torch.cat([state.salience,raw_score.unsqueeze(1)],dim=1)
        filled=torch.cat([state.filled,torch.ones_like(raw_score,dtype=torch.bool).unsqueeze(1)],dim=1)
        pos=torch.full((x.size(0),1),state.position,device=x.device,dtype=torch.long)
        positions=torch.cat((state.positions,pos),dim=1)
        allk=torch.cat((state.keys,k),dim=1)
        allv=torch.cat((state.values,v),dim=1)
        ranks=s.masked_fill(~filled,-1e5)
        idx=torch.topk(ranks,self.slots,dim=1,sorted=True).indices
        gather=idx.unsqueeze(-1).expand(-1,-1,self.memory_dim)
        return ArchiveState(allk.gather(1,gather),allv.gather(1,gather),
            s.gather(1,idx),filled.gather(1,idx),positions.gather(1,idx),state.position+1),raw_score


class SWQR(nn.Module):
    """A single architecture to TRAIN or INFER many independent models.

    - fast: original SWQR A/B/R recurrence, no archive parameters allocated.
    - mem: same SWQR core plus optional learned hidden-state top-K archive.
    - arbitrary vocab/dim/heads/layers; default 6 heads matches original B+4.
    """
    def __init__(self,config:SWQRConfig):
        super().__init__()
        self.config=config
        self.emb=nn.Embedding(config.vocab_size,config.dim)
        self.blocks=nn.ModuleList([SWQRBlock(config) for _ in range(config.layers)])
        self.final_norm=RMSNorm(config.dim)
        self.lm_head=nn.Linear(config.dim,config.vocab_size,bias=False)
        self.lm_head.weight=self.emb.weight
        self.archive=LearnedArchive(config) if config.mode=='mem' else None
        self.apply(self._init)
        with torch.no_grad():self.emb.weight.normal_(0,0.02)
    @staticmethod
    def _init(mod):
        if isinstance(mod,nn.Linear):
            nn.init.normal_(mod.weight,0,0.02)
            if mod.bias is not None:nn.init.zeros_(mod.bias)
        elif isinstance(mod,nn.Embedding):nn.init.normal_(mod.weight,0,0.02)
    @classmethod
    def from_scratch(cls,*,mode='fast',slots=0,**kwargs):
        return cls(SWQRConfig(mode=mode,slots=slots,**kwargs))
    def forward(self,tokens:torch.Tensor,cached_states=None,archive_state:Optional[ArchiveState]=None,
                salience_labels:Optional[torch.Tensor]=None):
        """Returns (all_causal_logits, recurrent_states, archive_state, auxiliary_loss).

        Query at t reads ONLY archive entries from tokens < t, then current x_t is
        considered for archival. For next-token loss, logits at t predict t+1.
        salience_labels [B,T] optional; must be genuinely supplied by the task.
        """
        if tokens.ndim!=2:raise ValueError('tokens must be [batch,time]')
        if tokens.shape[1]==0:raise ValueError('tokens sequence cannot be empty')
        x=self.emb(tokens)
        new_states=[]
        if cached_states is not None and len(cached_states)!=len(self.blocks):
            raise ValueError('cached_states length != decoder layers')
        for i,block in enumerate(self.blocks):
            previous=None if cached_states is None else cached_states[i]
            x,updated=block(x,previous);new_states.append(updated)
        aux=x.new_zeros(())
        if self.archive is not None:
            arc=archive_state if archive_state is not None else self.archive.empty(x.shape[0],x.device,x.dtype)
            reads=[]; score_seq=[]
            for t in range(x.size(1)):
                xt=x[:,t,:]
                reads.append(self.archive.read(xt,arc))
                arc,score=self.archive.write(xt,arc)
                score_seq.append(score)
            x=x+torch.stack(reads,dim=1)
            if salience_labels is not None:
                if salience_labels.shape!=tokens.shape:
                    raise ValueError('salience_labels shape mismatch')
                aux=F.binary_cross_entropy_with_logits(torch.stack(score_seq,dim=1),
                                                         salience_labels.float())
        else:
            if salience_labels is not None:raise ValueError('salience_labels only applies to mem mode')
            arc=None
        return self.lm_head(self.final_norm(x)),new_states,arc,aux
    def next_token_loss(self,tokens,*,salience_labels=None,aux_weight=0.2):
        """Teacher-forced causal LM loss, e.g. UTF-8 byte-level IDs (0..255)."""
        if tokens.shape[1]<2:raise ValueError('need at least two tokens for loss')
        labels=None if salience_labels is None else salience_labels[:,:-1]
        logits,_,_,aux=self(tokens[:,:-1],salience_labels=labels)
        loss=F.cross_entropy(logits.reshape(-1,self.config.vocab_size),tokens[:,1:].reshape(-1))
        return loss+aux_weight*aux, {'lm_loss':loss.detach(),'salience_loss':aux.detach()}
    @torch.no_grad()
    def prefill(self,tokens,chunk_size=128):
        """Streaming prefix; returns next-token logits and both caches."""
        if tokens.ndim!=2:raise ValueError('tokens must be [batch,time]')
        state=None;archive=None;logits=None
        for st in range(0,tokens.shape[1],chunk_size):
            logits,state,archive,_=self(tokens[:,st:st+chunk_size],state,archive)
        return logits[:,-1],state,archive
    @torch.no_grad()
    def decode_one(self,token,states,archive_state=None):
        if token.ndim==1:token=token.unsqueeze(1)
        logits,states,archive_state,_=self(token,states,archive_state)
        return logits[:,-1],states,archive_state
    @torch.no_grad()
    def generate(self,prompt,steps=32,temperature=0.):
        self.eval();lg,s,a=self.prefill(prompt)
        outputs=[]
        for _ in range(steps):
            if temperature>0:
                token=torch.multinomial(F.softmax(lg/temperature,dim=-1),1)
            else:token=lg.argmax(dim=-1,keepdim=True)
            outputs.append(token)
            lg,s,a=self.decode_one(token,s,a)
        return torch.cat([prompt,*outputs],dim=1)
    def save(self,path,optimizer=None,step=None):
        dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'format':'swqr-fast-mem-v2','config':asdict(self.config),
                    'model':self.state_dict(),'optimizer':optimizer.state_dict() if optimizer is not None else None,
                    'step':step},dest)
    @classmethod
    def load(cls,path,map_location='cpu'):
        obj=torch.load(path,map_location=map_location,weights_only=False)
        if obj.get('format')!='swqr-fast-mem-v2':raise ValueError('Expected from-scratch SWQR v2 checkpoint')
        model=cls(SWQRConfig(**obj['config']))
        model.load_state_dict(obj['model'],strict=True)
        return model,obj
    def load_legacy_b4_weights(self,path):
        """Optional transfer: load 60K Original SWQR B+4 core; never required.
        Only compatible with original 6H/6L/384D/vocab256/rank16.
        """
        c=self.config
        if (c.heads,c.layers,c.dim,c.vocab_size,c.gate_rank,c.ffn_ratio)!=(6,6,384,256,16,4):
            raise ValueError('Original B+4 transfer requires legacy 6H/6L/384D/vocab256/rank16/ffn4')
        obj=torch.load(path,map_location='cpu',weights_only=False)
        sd=obj.get('model',obj)
        missing,unexpected=self.load_state_dict(sd,strict=False)
        if unexpected or [k for k in missing if not k.startswith('archive.')]:
            raise ValueError(f'Legacy checkpoint mismatch: missing={missing}, unexpected={unexpected}')
        return self

__all__=['SWQR','SWQRConfig','ArchiveState','LearnedArchive','stable_scan']
