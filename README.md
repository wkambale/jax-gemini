<div align="center">
  <h1>Jax Gemini</h1>
  <p><b>Natural language-driven JAX and Flax model building powered by Google Gemini</b></p>

  [![PyPI version](https://img.shields.io/pypi/v/jax-gemini.svg)](https://pypi.org/project/jax-gemini/0.1.3/)
  [![Python Versions](https://img.shields.io/pypi/pyversions/jax-gemini.svg)](https://pypi.org/project/jax-gemini/0.1.3/)
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![CI](https://github.com/wkambale/jax-gemini/actions/workflows/ci.yml/badge.svg)](https://github.com/wkambale/jax-gemini/actions/workflows/ci.yml)
</div>

Package version: `0.1.3` ([PyPI](https://pypi.org/project/jax-gemini/0.1.3/)) | Supported Python: 3.10 to 3.12 | [Documentation site](https://wkambale.github.io/jax-gemini) | License: [MIT](LICENSE)

`jax-gemini` enables machine learning engineers and researchers to build, train, evaluate, and persist JAX and Flax neural networks from plain English prompts without writing repetitive boilerplate.

Unlike text-only LLM wrappers, `jax-gemini` executes validated generated code inside an isolated sandbox and returns live `flax.nnx.Module` instances, trained weights, evaluation dictionaries, and confined checkpoints.

## Core features

- **Executable JAX objects**: Returns instantiated `flax.nnx.Module` models, trained Optax states, and structured metrics dictionaries.
- **Bounded conversational memory**: Retains multi-turn refinement context and the active model state so follow-up calls such as `jg.modify()` and `jg.train()` build upon prior steps cleanly.
- **Layered sandbox validation**: Enforces AST size and node budgets, intent-scoped import allowlists, read-only module proxies, confined filesystem I/O, and execution timeouts.
- **Self-healing retry loop**: Sanitizes tracebacks and redacts paths and credentials before returning failure context to Gemini for correction, with per-attempt `runtime_args` rollback.
- **Pipeline and Flax/Optax utilities**: Includes composable pipeline wrappers in `jax_gemini.pipeline` and inspection helpers in `jax_gemini.contrib`.

## Quickstart

### Installation

Install the core package from [PyPI](https://pypi.org/project/jax-gemini/0.1.3/):

```bash
pip install jax-gemini
```

Install optional plotting support for `jg.visualize_data()`:

```bash
pip install "jax-gemini[viz]"
```

Export your [Gemini API key](https://aistudio.google.com/):

```bash
export GEMINI_API_KEY="your-api-key-here"
```

### Conversational model workflow

```python
import jax_gemini as jg
import numpy as np

# Configuration reads GEMINI_API_KEY from the environment when left unset
jg.config.set({"model_name": "gemini-3.1-pro"})

# Build a model from natural language
model = jg.build("Build a 4-layer MLP for handwritten digit classification")

# Refine the architecture across turns
model = jg.modify("Add dropout with rate 0.2 between each hidden layer")

# Train the active model
X_train = np.random.randn(100, 28, 28, 1).astype(np.float32)
y_train = np.random.randint(0, 10, size=(100,))

model, metrics = jg.train(
    "Train for 10 epochs with Adam and cross-entropy loss",
    dataset=(X_train, y_train),
)
print(f"Accuracy: {metrics['accuracy']:.2%}")

# Persist checkpoints inside config.checkpoint_dir
checkpoint_path = jg.save("digit_classifier_v1")

# Dispatch by intent automatically
eval_metrics = jg.run("Evaluate accuracy on the test set", dataset=(X_train, y_train))
```

See [examples/](examples/) for end-to-end Python scripts and Jupyter notebooks.

## Documentation

- [Quickstart guide](docs/quickstart.md): Installation, session lifecycle, training, and checkpointing.
- [API reference](docs/api-reference.md): Public functions, configuration fields, pipeline stages, and contrib helpers.
- [Architecture internals](docs/architecture.md): Module boundaries across generation, validation, sandbox execution, and memory.
- [Security protocol](docs/security.md): AST rules, import scoping, runtime proxies, path confinement, and timeout enforcement.

## License and contributing

Read the [contribution guidelines](CONTRIBUTING.md) before opening issues or pull requests. Distributed under the MIT License; see [LICENSE](LICENSE) for full terms.
