# ESA API Reference — v1.0.0

Standalone Python package: `esa`  
Distribution: `mlbricks-esa`

## Primary API

### `ESA`
```python
ESA(embd, head=4, batch=None, block=None, backend="auto", precision="fp16", *,
    compass="auto", dropout=0.0, gate_min=0.8, gate_max=0.995, eps=1e-5,
    device="auto", auto_compile=False, compile_mode="default",
    auto_move_input=True, strict_checks=False)
```
Creates an Entangled State Attention layer. Input is normally a tensor shaped `[batch, sequence, embd]`; output preserves the model dimension.

```python
from esa import ESA
layer = ESA(embd=384, head=6, backend="auto")
y = layer(x)
```

### `ESAConfig`
Configuration dataclass for an ESA layer. Important fields include `embd`, `head`, `backend`, `compass`, `precision`, `dropout`, and gate limits.

### `ESAModel`
```python
ESAModel(config=None, *, device="cuda", **kwargs)
```
Builds the standalone ESA language-model stack. Pass either `ESAModelConfig` or the corresponding keyword arguments.

```python
from esa import ESAModel, ESAModelConfig
cfg = ESAModelConfig(vocab_size=50257, block=256, n_layer=6, head=6, embd=384)
model = ESAModel(cfg, device="cuda")
```

### `ESAModelConfig`
Main language-model configuration. Core fields are `vocab_size`, `block`, `n_layer`, `head`, `embd`, `dropout`, `backend`, `precision`, `compass`, `ffn`, and `residual`.

### `Trainer`
```python
Trainer(model, *, optimizer=None, scheduler=None, scaler=None,
        checkpoint_dir="checkpoints", save_every=None, save_at=None,
        save_best=True, save_last=True, keep_last_n=3)
```
Training/checkpoint helper for `ESAModel`.

### `ThunderESA`
Optimized ESA backend with standalone native/PyTorch routing.

### `compass`
```python
compass(*, evaluate_fn, c_candidates=(8,16,32,64), precision="fp16",
        quality_tolerance=0.02, metric="ppl", speed_metric="tok_per_sec", **kwargs)
```
Evaluates candidate compass/chunk settings and returns `CompassResult`.

### `thunderBoost`
Optimization helper for a PyTorch module, with optional compilation, AMP, forward/backward benchmarking, and device selection.

## Generation API
The package exports `GenerationResult`, `GenerationStats`, and generation helpers from `esa.generation`, including prefill, lightning decode, and token sampling utilities.

## Backend/planner API
Every ESA repo is self-contained. Planner/runtime utilities live in `esa.planner` and `esa.runtime`; no installation of the umbrella `mlbricks` package is required. `backend="auto"` selects an available execution route; explicit routes can be requested where supported.

## Native acceleration
Native C++/CUDA support is optional. The PyTorch path remains available if the native extension cannot be built or loaded.
