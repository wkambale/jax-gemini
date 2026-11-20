# Quickstart guide

This guide walks through installing `jax-gemini` and executing common model workflows.

## Prerequisites

Install version `0.1.3` from [PyPI](https://pypi.org/project/jax-gemini/0.1.3/):

```bash
pip install jax-gemini
```

If you plan to generate plots with `jg.visualize_data()`, include the optional visualization extra:

```bash
pip install "jax-gemini[viz]"
```

Configure your Gemini API key via the `GEMINI_API_KEY` environment variable or explicitly through `jg.config.set()`:

```python
import os
import jax_gemini as jg

jg.config.set({
    "gemini_api_key": os.environ.get("GEMINI_API_KEY"),
    "model_name": "gemini-3.1-pro",
    "temperature": 0.2,
})
```

## Build a model

Call `jg.build()` with a natural-language architecture specification. `jax-gemini` generates `flax.nnx` code, validates it, executes it in the sandbox, and returns the instantiated `flax.nnx.Module`.

```python
my_model = jg.build(
    "Design a CNN for 32x32 RGB image classification using two convolutional blocks, "
    "max pooling, and dense classification layers."
)
```

## Refine iteratively

Session memory tracks prior turns and the active model state so follow-up modifications build on your existing architecture:

```python
my_model = jg.modify(
    "Add dropout with rate 0.25 between the convolutional blocks to reduce overfitting."
)
```

## Inspect generated code

You can review the validated Python source and Gemini's explanation at any point after an operation succeeds:

```python
print(jg.show_code())
print(jg.explain())
```

## Train and evaluate

Pass `(X, y)` NumPy array tuples into `jg.train()` and `jg.evaluate()`:

```python
trained_model, metrics = jg.train(
    prompt="Train for 15 epochs with categorical cross-entropy loss and Adam",
    dataset=(X_train, y_train),
)
print("Training loss:", metrics["loss"])

results = jg.evaluate(
    "Compute test set accuracy and loss",
    dataset=(X_test, y_test),
)
```

## Save and load checkpoints

Checkpoints are stored via Orbax strictly inside `config.checkpoint_dir` using relative names:

```python
ckpt_path = jg.save("v1_cnn_model_production")
restored_model = jg.load("v1_cnn_model_production")
```

## Automatic intent routing

If you prefer a single entrypoint, `jg.run()` classifies the prompt with `IntentClassifier` and dispatches to the corresponding operation:

```python
results = jg.run("Evaluate model accuracy on test set", dataset=(X_test, y_test))
```
