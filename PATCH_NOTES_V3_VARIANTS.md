# MLBricks ESA: V3 Packed+BMM variant patch

This patch is based on the uploaded `ESA-main (2).zip`. It **does not replace `ESA` (the sequence mixer) with a language model**. It does not remove or rename `ESAModel`, `ESAModelConfig`, `Trainer`, `ThunderESA`, or the original APIs.

## Public API (only one new optional argument)

```python
from esa import ESA, ESAModel, ESAModelConfig
import torch

# ESA is a MIXER. These calls construct only a single sequence-mixing layer.
enhanced = ESA(embd=384, head=6)  # NEW DEFAULT: SWQR V3 Packed+BMM
explicit = ESA(embd=384, head=6, variant='enhanced')
base = ESA(embd=384, head=6, variant='base')  # untouched original ThunderESA

# ESAModel remains a COMPLETE language model; old constructor names unchanged.
config = ESAModelConfig(
    vocab_size=50257, block=512, n_layer=6,
    head=6, embd=384, precision='fp16',
    variant='enhanced',  # optional, new default
)
model = ESAModel(config, device='cuda')

old_model = ESAModel(
    ESAModelConfig(vocab_size=50257, variant='base'), device='cuda'
)
```

The original `ESAConfig(embd=384, head=6, ...)` dataclass is retained, with the optional `variant` field. The original `ESA` constructor **still takes `embd`, `head`, etc. as before**; it has not been converted to accept an `ESAConfig` positional object.

## Implementations

- `base`: untouched `esa/backends/thunder.py` and original generation logic.
- `enhanced`: independent trainable SWQR A/B/R recurrence. Inference uses **V3 Packed+BMM**: prepacked A/V projection, prepacked write/read gate-down projection, grouped gate-up `torch.bmm`, and associative parallel scan in chunks of 128. No Triton V4 added.
- `ESA.forward` maintains `[batch, seq, embd] -> [batch, seq, embd]` for either mixer.
- `ESA.prefill` maintains `(output_seq, final_state)`; `decode_step` maintains `(output_token, next_state)` with both `[B,C]` and `[B,1,C]` inputs and flat/head state layouts.
- **Training on enhanced** uses the original differentiable SWQR recurrence. Optimized *packed* weights are used when doing inference under `torch.no_grad()` / `torch.inference_mode()`. The extra temporary packed tensors are **not** saved in checkpoints.
- Packed weights are automatically regenerated on `eval()`, first inference after `load_state_dict()`, or first inference after a device/dtype move. They are invalidated when switching to training. Manual mutation of model Parameters while remaining in eval mode requires `.layer.refresh_packs()` or a fresh `.eval()` call; optimizers should train with `.train()` as usual.
- The `native` runtime backend is **not available for enhanced SWQR** and is rejected explicitly; `auto` and `pytorch` use the supported SWQR computation. For original native Thunder, set `variant='base'`.
- The existing `model.prefill(engine='thunder')` and `model.generate(prefill='thunder_16')` method signatures are still accepted for the new default. For enhanced models, the engine's historical name does **not** change SWQR back into Thunder; the variant chooses the architecture.
- Thunder and SWQR have **different learned parameter structures**. Don't load an old Thunder state dict directly into a newly created enhanced model.

## Legacy checkpoints

- New `ESAModelConfig(...)` uses `variant='enhanced'` by default.
- Saved models include the variant in `config.json`.
- `ESAModel.load()` or `ESAModelConfig.from_dict()` from an old config **without** the variant assumes `variant='base'`, preserving the old Thunder state dict.
- If loading a **standalone old `ESA` mixer** state dict, explicitly construct `ESA(..., variant='base')` before calling `load_state_dict()`; standalone layer state dicts don't encode variant choice.
- The old `Trainer` interface is preserved. To resume a checkpoint, instantiate the same architecture as the checkpoint before calling `Trainer.load_checkpoint`.

## Tests performed (CPU)

```bash
python -m pip install -e .
python -m pytest -q tests/test_variants.py tests/test_smoke.py
```

18 regression tests passed: default/base selection, original Thunder parameter parity, SWQR forward/prefill/state parity at 1/17/128/129/256/257 tokens, training gradients and pack refresh after optimizer/load, both input/state decode layouts, model generation, checkpoint save/load including legacy variant-free configs, full-model forward↔prefill↔decode parity, and compiled-wrapper routing with `torch.compile(backend='eager',fullgraph=True)`.

**Hardware limitation:** This environment has no CUDA GPU. The integration and regression tests are CPU tests; CUDA/Triton or native binary performance was not measured. Previous V3 Packed+BMM T4 benchmark figures are external experiment results, not newly measured by this patch.

## Changed files

- `esa/config.py`: optional ESAConfig variant + validation.
- `esa/layer.py`: optional ESA mixer variant selection and enhanced state/prefill/decode routing.
- `esa/model.py`: optional ESAModelConfig variant + legacy-load rule + pass variant into mixers.
- `esa/backends/swqr_enhanced.py`: new V3 packed/BMM inference implementation (training uses reference SWQR).
- `esa/swqr_trainable.py`: original SWQR reference source used for exactly matching A/B/R parameters and gradients.
- `tests/test_variants.py`: regression tests.
- `PATCH_NOTES_V3_VARIANTS.md`: these notes.

All other source files remain byte-for-byte unchanged from the uploaded repository. Apply `MLBricks_ESA_V3_variant.patch` from the root of the original repository using `patch -p1 < MLBricks_ESA_V3_variant.patch`, or use the prepatched source archive.
