# Original SWQR opt-in patch (reference, not optimized)

This package adds `esa.swqr.OriginalSWQR` and `original_swqr_recall_model`.
It does not change `ESA` or Thunder, so preexisting production callers keep
identical behavior. The original SWQR experiment uses read gating BEFORE
RMSNorm, a rank-16 selective-write/read network, six learned half-life heads,
and the original FP32 chunked prefix-product scan.

```python
from esa import OriginalSWQR, original_swqr_recall_model
import torch
layer = OriginalSWQR(embd=384, head=6, chunk_size=32)
y, final_state = layer(torch.randn(2, 128, 384))
model = original_swqr_recall_model()
logits, states = model(torch.randint(0, 256, (2, 128)))
```

The benchmark architecture has 9,849,168 tied-weight parameters.
For loading trained original-SWQR checkpoints, pass the checkpoint's
**model state dict** to `model.load_state_dict(state_dict, strict=True)`;
validate keys and configuration rather than silently filtering missing keys.

Run `python -m pytest tests/test_original_swqr.py -q` before attempting
optimization. This patch does not claim production-quality/long-context
benchmark improvements until trained weights are tested independently.
