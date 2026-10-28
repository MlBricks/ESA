# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Original ESA-SWQR research implementation (reference, unoptimized).

Exact architecture from the original Kaggle SWQR experiment: learned head
half-lives, low-rank selective write/read gates, and **read before RMSNorm**.
Production Thunder ESA is unchanged. The reference scan is intentionally
kept unchanged for checkpoint reproducibility; optimize only after validation.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
DEFAULT_TIMESCALES = [16, 64, 256, 1024, 4096, 8192]

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def chunk_scan(a, v, initial=None, chunk_size=32):
    """Exact-in-real-arithmetic associative affine recurrence, chunked FP32 scan.

    E_t = a_t * E_(t-1) + v_t. a and v shaped [B,T,H,Dh].
    Zero-length chunks not allowed. Preserve recurrent state across chunks.
    """
    B, T, H, Dh = a.shape
    state = (v.new_zeros(B, H, Dh) if initial is None else initial).float()
    result = []
    for i in range(0, T, chunk_size):
        aa = a[:, i:i+chunk_size].float()
        vv = v[:, i:i+chunk_size].float()
        # A >= ~0.9 initially, so product of 32 terms stays well in FP32 range.
        pref = torch.cumprod(aa, dim=1)
        val = pref * (state[:, None] + torch.cumsum(vv / pref, dim=1))
        state = val[:, -1]
        result.append(val)
    return torch.cat(result, dim=1), state


class OriginalSWQRLayer(nn.Module):
    def __init__(self, dim, heads, variant, chunk_size=32, hard_max=0.9999):
        super().__init__()
        if dim % heads:
            raise ValueError('dim must be divisible by heads')
        if variant not in {'long_memory', 'selective_write_read', 'write_only', 'read_only'}:
            raise ValueError('Unknown ESA variant')
        if heads != len(DEFAULT_TIMESCALES):
            raise ValueError('This controlled experiment requires 6 heads')
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.variant, self.chunk_size, self.hard_max = variant, chunk_size, hard_max
        self.W_A = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)
        self.W_O = nn.Linear(dim, dim, bias=False)
        self.state_norm = RMSNorm(dim)
        # All variants use the identical long-memory decay equation.
        self.log_half_life = nn.Parameter(
            torch.tensor(DEFAULT_TIMESCALES, dtype=torch.float32).log())
        self.raw_decay_bias = nn.Parameter(torch.full(
            (heads,), math.log(math.expm1(1.0))))
        # Low-rank feature gates. FFN width is reduced in the enhanced model
        # to offset their parameters and keep the model size nearly identical.
        self.use_write = variant in {'selective_write_read', 'write_only'}
        self.use_read = variant in {'selective_write_read', 'read_only'}
        rank = 16
        if self.use_write:
            self.write_down = nn.Linear(dim, rank, bias=False)
            self.write_up = nn.Linear(rank, dim, bias=False)
            self.write_bias = nn.Parameter(torch.full((heads,), 4.0))
        if self.use_read:
            self.read_down = nn.Linear(dim, rank, bias=False)
            self.read_up = nn.Linear(rank, dim, bias=False)
            self.read_bias = nn.Parameter(torch.full((heads,), 4.0))

    def gates(self, x):
        B, T, _ = x.shape
        z = self.W_A(x).reshape(B, T, self.heads, self.dh).float()
        half_life = self.log_half_life.exp()
        decay = F.softplus(z + self.raw_decay_bias[None, None, :, None])
        return torch.exp(-math.log(2)*decay / half_life[None, None, :, None])

    def feature_gate(self, x, kind):
        down = getattr(self, kind+'_down')
        up = getattr(self, kind+'_up')
        bias = getattr(self, kind+'_bias')
        B,T,_ = x.shape
        gate = up(F.silu(down(x))).float().reshape(B,T,self.heads,self.dh)
        return (gate + bias[None,None,:,None]).sigmoid()

    def forward(self, x, state=None):
        B, T, _ = x.shape
        a = self.gates(x)
        v = self.W_V(x).reshape(B, T, self.heads, self.dh)
        if self.use_write:
            v = v * self.feature_gate(x, 'write').to(v.dtype)
        states, final = chunk_scan(a, v, initial=state, chunk_size=self.chunk_size)
        if self.use_read:
            states = states * self.feature_gate(x, 'read')
        y = self.state_norm(states.reshape(B, T, self.dim)).to(x.dtype)
        return self.W_O(y), final

    @torch.no_grad()
    def stats(self):
        result = {
            'half_life': self.log_half_life.exp().detach().cpu().tolist(),
            'decay_bias': self.raw_decay_bias.detach().cpu().tolist()}
        if self.use_write:
            result['write_gate_bias_only'] = self.write_bias.sigmoid().detach().cpu().tolist()
        if self.use_read:
            result['read_gate_bias_only'] = self.read_bias.sigmoid().detach().cpu().tolist()
        return result


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, variant, chunk_size):
        super().__init__()
        self.pre_esa = RMSNorm(dim)
        self.esa = OriginalSWQRLayer(dim, heads, variant, chunk_size)
        self.pre_ffn = RMSNorm(dim)
        # Each low-rank feature gate adds 2*D*16 + H trainable parameters.
        # Remove 16 FFN neurons per added gate to offset their cost.
        gates_count = int(self.esa.use_write) + int(self.esa.use_read)
        ffn_width = 4*dim - 16*gates_count
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_width), nn.GELU(),
                                 nn.Linear(ffn_width, dim))

    def forward(self, x, state=None):
        y, state = self.esa(self.pre_esa(x), state)
        x = x + y
        return x + self.ffn(self.pre_ffn(x)), state


class OriginalSWQRRecallModel(nn.Module):
    def __init__(self, vocab_size=256, dim=384, heads=6, layers=6,
                 variant='long_memory', chunk_size=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([DecoderBlock(dim, heads, variant, chunk_size)
                                     for _ in range(layers)])
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.emb.weight
        self.apply(self._init)
        with torch.no_grad():
            self.emb.weight.normal_(0, 0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, tokens, cached_states=None):
        x = self.emb(tokens)
        new_states = []
        for i, layer in enumerate(self.blocks):
            s = None if cached_states is None else cached_states[i]
            x, updated = layer(x, s)
            new_states.append(updated)
        # Return only the final token logits: reduces memory dramatically.
        return self.lm_head(self.final_norm(x[:, -1, :])), new_states

    @torch.no_grad()
    def all_stats(self):
        return [blk.esa.stats() for blk in self.blocks]



class OriginalSWQR(OriginalSWQRLayer):
    """Single original selective-write/read layer (returns output, final state)."""
    def __init__(self, embd: int, head: int = 6, chunk_size: int = 32):
        super().__init__(embd, head, 'selective_write_read', chunk_size)


def original_swqr_recall_model(vocab_size: int = 256, embd: int = 384,
                               head: int = 6, layers: int = 6,
                               chunk_size: int = 32) -> OriginalSWQRRecallModel:
    """Reference checkpoint-compatible causal language model for synthetic recall."""
    return OriginalSWQRRecallModel(vocab_size=vocab_size, dim=embd, heads=head,
                                   layers=layers, variant='selective_write_read',
                                   chunk_size=chunk_size)


__all__ = ['OriginalSWQR', 'OriginalSWQRLayer', 'OriginalSWQRRecallModel',
           'original_swqr_recall_model', 'chunk_scan']
