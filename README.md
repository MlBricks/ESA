# ESA — Entangled State Attention

Standalone ESA library extracted from MLBricks. It contains its own planner/runtime and the small model-support components needed by `ESAModel`, so it does not require the umbrella `mlbricks` package.

```python
from esa import ESA, ESAModel, ESAModelConfig
```

`backend="auto"` uses the local planner and falls back to PyTorch whenever native kernels are unavailable.

## Release

Version: **1.0.0**

See [API.md](API.md) for the standalone API reference and [LICENSE.md](LICENSE.md) for this component's license terms.
