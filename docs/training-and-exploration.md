# Training and exploration

Jax Forge supports both imperative single-shot training loops (`jf.train()`, `jf.evaluate()`) and metric-guided branching search (`jf.explore()`).

## Single-run training and evaluation

Pass training tensors or custom loops to `jf.train()` and validate held-out data with `jf.evaluate()`:

```python
import jaxforge as jf
import numpy as np

X_train = np.random.randn(128, 16).astype(np.float32)
y_train = np.random.randint(0, 4, size=(128,))

model = jf.build("Build a 3-layer Flax NNX classifier for 16 features and 4 classes")
model, train_metrics = jf.train(
    "Train for 10 epochs with Optax AdamW and cross-entropy loss",
    dataset=(X_train, y_train),
)
eval_metrics = jf.evaluate("Compute classification accuracy and loss", dataset=(X_train, y_train))
print(train_metrics, eval_metrics)
```

## Metric-guided branching exploration

`jf.explore()` evaluates bounded optimization strategies (baseline AdamW convergence, stronger regularization with cosine decay, and deeper residual architectures with gradient clipping), logs each trial into `ExperimentResult`, and promotes the best-scoring candidate model onto the session automatically:

```python
experiment = jf.explore(
    "Train image classifier with strong generalization",
    dataset=(X_train, y_train),
    metric="accuracy",
    maximize=True,
    candidates=3,
)

print("Best candidate:", experiment.best_candidate.index, experiment.best_candidate.score)
for trial in experiment.candidates:
    print(trial.index, trial.metrics, trial.error)
```

If an individual trial fails validation or execution, `explore()` captures the diagnostic error on that candidate and preserves the base model cleanly for the remaining branches.

## Contrib helpers and metric tracking

Use `jaxforge.contrib` utilities inside custom scripts or inspections:

- **`eval_shape_summary(fn, *specs)`**: Inspect output pytree shapes and dtypes using `jax.eval_shape` without allocating device memory.
- **`create_cosine_schedule(...)`**: Build Optax linear warmup plus cosine decay schedules cleanly.
- **`InMemoryTracker` and `JsonlTracker`**: Record structured step metrics alongside agent training runs.
