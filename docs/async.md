# Asynchronous support

`JaxForge` provides non-blocking async wrappers for every primary generation, preview, refinement, training, and evaluation method so web services, Jupyter kernels, and orchestration frameworks remain responsive.

## Async session methods

Instantiate an isolated `JaxForge` session and await its async methods cleanly:

```python
import asyncio
import numpy as np
from jaxforge import JaxForge, JaxForgeConfig

async def run_pipeline():
    agent = JaxForge(JaxForgeConfig(model_name="gemini-3.1-pro"))
    plan = await agent.apreview("Build a 2-layer Flax NNX MLP")
    model = await agent.abuild("Build a 2-layer Flax NNX MLP")
    model = await agent.amodify("Add LayerNorm between dense blocks")

    X = np.random.randn(64, 8).astype(np.float32)
    y = np.random.randint(0, 2, size=(64,))
    model, metrics = await agent.atrain("Train binary classifier", dataset=(X, y))
    eval_metrics = await agent.aevaluate("Evaluate binary accuracy", dataset=(X, y))
    return model, metrics, eval_metrics

asyncio.run(run_pipeline())
```

## Parity with synchronous APIs

| Synchronous method | Asynchronous equivalent | Return value |
| --- | --- | --- |
| `agent.preview(...)` | `await agent.apreview(...)` | `GeneratedPlan` |
| `agent.build(...)` | `await agent.abuild(...)` | Model instance (or `GeneratedPlan` with `dry_run=True`) |
| `agent.modify(...)` | `await agent.amodify(...)` | Refined model instance (or `GeneratedPlan` with `dry_run=True`) |
| `agent.train(...)` | `await agent.atrain(...)` | `(model, metrics)` tuple |
| `agent.evaluate(...)` | `await agent.aevaluate(...)` | Evaluation metrics dict |

For concurrent multi-user services, allocate one `JaxForge` instance per request or user session instead of mutating the module-level `jf.config` singleton.
