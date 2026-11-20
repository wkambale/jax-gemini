# Preview and diffs

Instead of blindly executing model-generated Python code, Jax Forge defaults to an inspect-first workflow built around `GeneratedPlan`, `approve` callbacks, `DatasetProfile`, and unified diff patching.

## Preview before execution

Call `jf.preview()` or pass `dry_run=True` to `build()`, `modify()`, `train()`, `evaluate()`, or `run()` to generate source code, run `CodeValidator`, and return a structured `GeneratedPlan` without allocating accelerators or running compute:

```python
import jaxforge as jf

plan = jf.preview("Build a residual MLP with layer normalization and GELU")
print("Intent:", plan.intent)
print("Valid:", plan.validation_passed)
print(plan.code)
```

In Jupyter and IPython notebooks, `GeneratedPlan` implements `_repr_html_()` to render intent metadata, validation status, optional diffs, warnings, and formatted source automatically.

## Enforcing human or policy approval gates

Set `auto_approve=False` in `JaxForgeConfig` to require an explicit `approve(plan) -> bool` callback whenever code is executed:

```python
from jaxforge import JaxForge, JaxForgeConfig, JaxForgeApprovalError

agent = JaxForge(JaxForgeConfig(gemini_api_key="your-key", auto_approve=False))

def review_gate(plan):
    print(plan.code)
    return input("Approve execution? [y/N]: ").strip().lower() == "y"

try:
    model = agent.build("Build a two-layer classifier", approve=review_gate)
except JaxForgeApprovalError as err:
    print("Execution paused or rejected:", err)
```

## Unified diff refinement

When iterating on an existing model with `modify(..., use_diff=True)`, Jax Forge prompts the LLM to emit focused unified diff hunks (`diff`) when appropriate and applies them cleanly against `_last_code` via `CodeParser` and `apply_unified_diff()`. Both before/after code and the computed unified diff are attached to `GeneratedPlan.diff` for auditing.

## Privacy-preserving dataset profiling

Passing `(X, y)` or NumPy arrays to `preview()`, `train()`, `evaluate()`, or `explore()` computes a structural `DatasetProfile` without transmitting raw sample rows:

```python
import numpy as np
from jaxforge.plan import DatasetProfile

X = np.random.randn(256, 32).astype(np.float32)
y = np.random.randint(0, 5, size=(256,))
profile = DatasetProfile.from_dataset((X, y))
print(profile.to_prompt_text())
```

The generated summary injects tensor shapes, dtypes, class counts, NaN/Inf detection, and value bounds (`min`/`max`) into the prompt context.
