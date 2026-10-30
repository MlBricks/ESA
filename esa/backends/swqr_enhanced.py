# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.
"""ESA-SWQR V3 Packed+BMM mixer, preserving the original SWQR A/B/R recurrence.

A/B/R projections remain independent *trainable* parameters. For inference,
A/V and both gate-down weights are cached in packed non-persistent buffers;
write/read gate-up operations are grouped in one batched GEMM. The original
SWQR differentiable reference path is retained for training.

The packed weights are derived (not checkpoint weights), invalidated whenever
parameters are loaded / model is moved / training resumes, and rebuilt on
entering eval mode or the first inference call. Native Thunder kernels are not
used here because their recurrence differs from SWQR.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from ..swqr_trainable import SWQRConfig, SWQRLayer
from .thunder import associative_chunk_scan


class EnhancedSWQRLayer(SWQRLayer):
    """Drop-in [B,T,D] SWQR mixer with separate inference/training execution."""

    def __init__(self, *, embd: int, head: int, dropout: float = 0.0,
                 eps: float = 1e-5, gate_rank: int = 16, chunk_size: int = 128):
        cfg = SWQRConfig(dim=embd, heads=head, layers=1, mode="fast", slots=0,
                         dropout=dropout, gate_rank=gate_rank, chunk_size=chunk_size)
        super().__init__(cfg)
        self.state_norm.eps = float(eps)
        self.inference_chunk_size = int(chunk_size)
        self.register_buffer("_packed_av", torch.empty(0), persistent=False)
        self.register_buffer("_packed_gate_down", torch.empty(0), persistent=False)
        self.register_buffer("_packed_gate_up", torch.empty(0), persistent=False)
        self._packs_valid = False
        self.register_load_state_dict_post_hook(self._on_state_loaded)

    def _on_state_loaded(self, module, incompatible_keys):
        self._packs_valid = False

    def _apply(self, fn):
        result = super()._apply(fn)
        self._packs_valid = False
        return result

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._packs_valid = False
        else:
            self.refresh_packs()
        return self

    @torch.no_grad()
    def refresh_packs(self):
        """Rebuild inference-only packed weight caches from source Parameters."""
        self._packed_av = torch.cat((self.W_A.weight, self.W_V.weight), dim=0).contiguous().detach()
        self._packed_gate_down = torch.cat(
            (self.write_down.weight, self.read_down.weight), dim=0
        ).contiguous().detach()
        self._packed_gate_up = torch.stack(
            (self.write_up.weight.T, self.read_up.weight.T), dim=0
        ).contiguous().detach()
        self._packs_valid = True
        return self

    def _inference_scan(self, x: torch.Tensor, state: torch.Tensor | None):
        """Packed+BMM SWQR recurrence, including carried state across chunks."""
        if not self._packs_valid:
            self.refresh_packs()
        b, total, _ = x.shape
        if total < 1:
            raise ValueError("ESA-SWQR expects a non-empty sequence.")
        carry = (torch.zeros((b, self.heads, self.dh), device=x.device, dtype=torch.float32)
                 if state is None else state.float().reshape(b, self.heads, self.dh))
        outputs = []
        for start in range(0, total, self.inference_chunk_size):
            z = x[:, start:start + self.inference_chunk_size]
            b, t, _ = z.shape
            av = F.linear(z, self._packed_av).float()
            proj_a, proj_v = av.chunk(2, dim=-1)
            proj_a = proj_a.reshape(b, t, self.heads, self.dh)
            proj_v = proj_v.reshape(b, t, self.heads, self.dh)
            decay = F.softplus(proj_a + self.raw_decay_bias[None, None, :, None])
            a = torch.exp(-math.log(2.0) * decay /
                          self.log_half_life.exp()[None, None, :, None])

            hidden = F.silu(F.linear(z, self._packed_gate_down))
            rank = self.write_down.out_features
            grouped = hidden.reshape(b * t, 2, rank).transpose(0, 1).contiguous()
            gates = torch.bmm(grouped, self._packed_gate_up).transpose(0, 1)
            w = gates[:, 0].reshape(b, t, self.heads, self.dh).float()
            r = gates[:, 1].reshape(b, t, self.heads, self.dh).float()
            write = (w + self.write_bias[None, None, :, None]).sigmoid()
            read = (r + self.read_bias[None, None, :, None]).sigmoid()

            u = proj_v * write
            if t == 1:
                recurrent = a * carry.unsqueeze(1) + u
            else:
                prefix_a, prefix_u = associative_chunk_scan(a, u)
                recurrent = prefix_u + prefix_a * carry.unsqueeze(1)
            carry = recurrent[:, -1].contiguous()
            gated = recurrent * read
            out = self.W_O(self.state_norm(gated.reshape(b, t, self.dim)).to(z.dtype))
            outputs.append(out)
        return torch.cat(outputs, dim=1), carry

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        # During training (or explicit autograd) use the original differentiable
        # SWQR recurrence; frozen inference packs MUST never drop gradients.
        if self.training or torch.is_grad_enabled():
            return super().forward(x, state)
        return self._inference_scan(x, state)

    @torch.no_grad()
    def prefill(self, x: torch.Tensor, state: torch.Tensor | None = None):
        # Full-sequence prefill always uses Packed+BMM, including when called
        # by the original ESAModel prefill API.
        return self._inference_scan(x, state)

    def decode_step(self, x: torch.Tensor, state: torch.Tensor):
        squeeze = x.ndim == 2
        flat_state = state.ndim == 2
        if squeeze:
            x = x[:, None, :]
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError("decode_step expects [B,D] or [B,1,D]")
        y, next_state = self.forward(x, state)
        if flat_state:
            next_state = next_state.reshape(x.size(0), self.dim).contiguous()
        return (y[:, 0] if squeeze else y), next_state
