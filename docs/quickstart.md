# Quickstart guide

This guide walks through installing `jaxforge` (**Jax Forge**), choosing an LLM provider, previewing generated code, training, exploring, and exporting standalone packages.

## Prerequisites

Install version `0.2.0` from [PyPI](https://pypi.org/project/jaxforge/0.2.0/):

```bash
pip install jaxforge
```

Optional extras add plotting, provider SDKs, Equinox/Grain/Safetensors, remote sandboxes, or MCP support:

```bash
pip install "jaxforge[viz,equinox,grain,safetensors]"
```

Configure credentials for Google Gemini, OpenAI, Anthropic, OpenRouter, or point `base_url` at local Ollama:

```python
import jaxforge as jf

jf.config.set({
    "model_name": "openai:gpt-5",  # or "gemini-3.1-pro", "anthropic:claude-sonnet-4-5", "ollama:llama3"
    "temperature": 0.2,
})
```

## Preview code before execution

Call `jf.preview()` (or pass `dry_run=True` to `build()`, `modify()`, or `train()`) to inspect validated source code and diffs without executing compute:

```python
plan = jf.preview(
    "Design a CNN for 32x32 RGB image classification using two convolutional blocks and dense heads."
)
print(plan.code)
model = jf._default_agent.execute_plan(plan)
```

## Build, refine, and inspect

Call `jf.build()` directly when auto-approval is enabled, then iterate with `jf.modify()` (which requests targeted unified diffs when possible):

```python
my_model = jf.build("Build a 3-layer Flax NNX classifier")
my_model = jf.modify("Add dropout with rate 0.25 between hidden layers")
print(jf.show_code())
print(jf.explain())
```

## Train, evaluate, and explore

Pass `(X, y)` NumPy array tuples or streaming batches into `jf.train()`, `jf.evaluate()`, or metric-guided `jf.explore()`:

```python
trained_model, metrics = jf.train(
    prompt="Train for 15 epochs with categorical cross-entropy loss and AdamW",
    dataset=(X_train, y_train),
)

experiment = jf.explore(
    "Tune regularization and optimizer schedule",
    dataset=(X_train, y_train),
    metric="accuracy",
    candidates=3,
)
print("Best score:", experiment.best_candidate.score)
```

## Save, load, and export standalone packages

Save Orbax checkpoints inside `config.checkpoint_dir`, or export a clean installable project with unit smoke tests:

```python
ckpt_path = jf.save("v1_cnn_model")
restored_model = jf.load("v1_cnn_model")

jf.export_project("./my_exported_model", package_name="my_exported_model")
```
