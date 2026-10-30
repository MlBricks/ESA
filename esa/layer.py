# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

import torch
import torch.nn as nn

from .runtime import normalize_backend
from .config import _normalize_variant


class ESA(nn.Module):
    """Public Entangled State Attention sequence-mixing layer.

    Thunder is the single production ESA backend. ``compass="auto"`` is the
    default and lets MLBricks choose a validated scan configuration. Advanced
    users can pass a positive integer such as ``compass=16`` or ``compass=32``.
    """

    def __init__(
        self,
        embd: int,
        head: int = 4,
        batch: int | None = None,
        block: int | None = None,
        backend: str = "auto",
        precision: str = "fp16",
        *,
        compass: int | str | None = "auto",
        dropout: float = 0.0,
        gate_min: float = 0.80,
        gate_max: float = 0.995,
        eps: float = 1e-5,
        device: str | torch.device | None = "auto",
        auto_compile: bool = False,
        compile_mode: str = "default",
        auto_move_input: bool = True,
        strict_checks: bool = False,
        variant: str = "enhanced",
    ):
        super().__init__()

        if embd <= 0:
            raise ValueError(f"embd must be positive, got {embd}.")
        if head <= 0:
            raise ValueError(f"head must be positive, got {head}.")
        if embd % head != 0:
            raise ValueError(
                f"embd must be divisible by head, got embd={embd}, head={head}."
            )
        if batch is not None and batch <= 0:
            raise ValueError(f"batch must be positive when set, got {batch}.")
        if block is not None and block <= 0:
            raise ValueError(f"block must be positive when set, got {block}.")

        self.variant = _normalize_variant(variant)
        self.embd = int(embd)
        self.head = int(head)
        self.batch = None if batch is None else int(batch)
        self.block = None if block is None else int(block)
        self.backend = normalize_backend(backend, warn_legacy=True)
        self.precision = str(precision).lower()
        self.dropout = float(dropout)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.eps = float(eps)
        self.auto_move_input = bool(auto_move_input)
        self.strict_checks = bool(strict_checks)
        self.compiled = False

        if compass is None:
            self.compass: int | str = "auto"
        elif isinstance(compass, str):
            if compass.strip().lower() != "auto":
                raise ValueError("compass must be a positive integer or 'auto'.")
            self.compass = "auto"
        else:
            self.compass = int(compass)
            if self.compass <= 0:
                raise ValueError(f"compass must be positive, got {self.compass}.")

        if self.variant == "base":
            from .backends.thunder import ThunderESA
            self.layer = ThunderESA(
                embd=self.embd,
                head=self.head,
                compass=self.compass,
                dropout=self.dropout,
                precision=self.precision,
                gate_min=self.gate_min,
                gate_max=self.gate_max,
                eps=self.eps,
                runtime_backend=self.backend,
            )
        else:
            if self.backend == "native":
                raise ValueError("variant='enhanced' does not have a native Thunder kernel; use backend='auto' or 'pytorch'.")
            from .backends.swqr_enhanced import EnhancedSWQRLayer
            self.layer = EnhancedSWQRLayer(
                embd=self.embd,
                head=self.head,
                dropout=self.dropout,
                eps=self.eps,
            )

        self.register_buffer("_esa_device_ref", torch.empty(0), persistent=False)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device is not None:
            self.to(device)
        if auto_compile:
            self.compile(mode=compile_mode)

    def set_backend(self, backend: str, *, recursive: bool = True):
        del recursive
        value = normalize_backend(backend, warn_legacy=True)
        if self.variant == "enhanced" and value == "native":
            raise ValueError("variant='enhanced' does not support native Thunder kernels.")
        self.backend = value
        target = getattr(self.layer, "_orig_mod", self.layer)
        if hasattr(target, "runtime_backend"):
            target.runtime_backend = self.backend
        return self

    def resolved_backend(self) -> str:
        if self.variant == "enhanced":
            return "pytorch"
        if self.backend == "pytorch":
            return "pytorch"
        if self.backend == "native":
            return "native-required"
        try:
            from .native import available
            return "native" if available() else "pytorch"
        except Exception:
            return "pytorch"

    def compile(self, mode: str = "default"):
        """Compile this ESA layer with torch.compile and return the module."""
        try:
            self.layer = torch.compile(self.layer, mode=mode, fullgraph=False)
            self.compiled = True
        except Exception:
            self.compiled = False
        return self

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.auto_move_input:
            target_device = self._esa_device_ref.device
            if x.device != target_device:
                x = x.to(target_device, non_blocking=True)
        if x.dim() != 3:
            raise ValueError(f"ESA input must be [batch, seq, embd], got {tuple(x.shape)}.")
        B, T, C = x.shape
        if C != self.embd:
            raise ValueError(f"ESA expected embd={self.embd}, got {C}.")
        if self.strict_checks:
            if self.batch is not None and B != self.batch:
                raise ValueError(f"ESA expected batch={self.batch}, got {B}.")
            if self.block is not None and T > self.block:
                raise ValueError(f"ESA block limit is {self.block}, got {T}.")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.layer(self._prepare_input(x))
        return result[0] if self.variant == "enhanced" else result

    def init_state(self, batch: int, *, device=None, dtype=None, layout: str = "heads") -> torch.Tensor:
        if self.variant == "enhanced":
            if batch <= 0:
                raise ValueError("batch must be positive.")
            if layout not in {"heads", "flat"}:
                raise ValueError("layout must be heads or flat")
            target = self._esa_device_ref.device if device is None else torch.device(device)
            state_dtype = torch.float32 if dtype is None else dtype
            shape = (batch, self.head, self.embd // self.head) if layout == "heads" else (batch, self.embd)
            return torch.zeros(shape, device=target, dtype=state_dtype)
        from .generation import lightning_init_state
        return lightning_init_state(self, batch, device=device, dtype=dtype, layout=layout)

    @torch.no_grad()
    def prefill(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        backend: str | None = None,
        compass: int | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a full sequence and return outputs plus final recurrent state."""
        if self.variant == "enhanced":
            if backend is not None and normalize_backend(backend, warn_legacy=False) == "native":
                raise ValueError("Native Thunder prefill cannot be used with variant='enhanced'.")
            target = getattr(self.layer, "_orig_mod", self.layer)
            return target.prefill(self._prepare_input(x), state)
        from .generation import esa_prefill
        return esa_prefill(
            self,
            self._prepare_input(x),
            state=state,
            backend=backend,
            compass=compass,
        )

    def decode_step(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.variant == "enhanced":
            target = getattr(self.layer, "_orig_mod", self.layer)
            return target.decode_step(self._prepare_input(x) if x.ndim == 3 else x, state)
        from .generation import lightning_decode_step
        return lightning_decode_step(self, x, state)

    lightning_prefill = prefill
    lightning_step = decode_step
