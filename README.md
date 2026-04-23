<div align="center">
  <h1>Jax-Gemini</h1>
  <p><b>Natural language-driven JAX/Flax model building powered by Google Gemini.</b></p>
  
  [![PyPI version](https://img.shields.io/pypi/v/jax-gemini.svg)](https://pypi.org/project/jax-gemini/)
  [![Python Versions](https://img.shields.io/pypi/pyversions/jax-gemini.svg)](https://pypi.org/project/jax-gemini/)
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
</div>

<br/>

`jax-gemini` is a Python library that allows AI/ML practitioners, researchers, and domain experts to build, train, evaluate, and snapshot JAX/Flax neural network models using conversational plain English prompts without writing JAX boilerplate code.

Unlike traditional descriptive LLM wrappers, **jax-gemini returns real, executable Python/JAX `flax.nnx.Module` objects**. Your prompts dictate instructions; Gemini generates the respective code; jax-gemini runs it securely inside a sandboxed namespace and hands the resulting object back to you.

## Features

- **Real Objects, Not Text**: Every operation yields a genuine Python object (e.g. `flax.nnx.Module`), trained weights, or dictionaries natively integrated with JAX environments.
- **Conversational Memory**: Retains the context of modifications multi-turn iteratively. Add dropout layers, reshape variables or tweak hyper-parameters step-by-step.
- **Safe Execution by Default**: All parsed LLM code generation is restricted through an AST Abstract Syntax Tree Validator restricting arbitrary script execution and imports. 
- **Auto-Healing**: Encountering exceptions? jax-gemini features an adaptive auto-correction mechanism which relays failed code and Python tracebacks directly back to Gemini for automatic pipeline repair.

## Quickstart

### Installation

Install via [PyPI](https://pypi.org/project/jax-gemini/):
```bash
pip install jax-gemini
```

Setup your [Gemini API Key](https://aistudio.google.com/):
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### Conversational Model Building

```python
import jax_gemini as jg
import numpy as np

# Note: jg automatically picks up GEMINI_API_KEY from env
jg.config.set({"model_name": "gemini-3.1-pro"})

# Build a model exclusively purely from text
model = jg.build("Build a 4-layer MLP for handwritten digit classification")

# Refine the architecture interactively
model = jg.modify("Ah, wait, add dropout with rate 0.2 between each layer for regularization")

# Train your model
X_train = np.random.randn(100, 28, 28, 1).astype(np.float32)
y_train = np.random.randint(0, 10, size=(100,))

model, metrics = jg.train(
    "Train for 10 epochs with Adam optimizer and Cross Entropy Loss",
    dataset=(X_train, y_train)
)
print(f"Accuracy: {metrics['accuracy']:.2%}")

# Checkpoint Persistence
checkpoint_path = jg.save("digit_classifier_v1")
```

See [examples/](examples/) for more comprehensive workflows such as Jupyter Notebook deployments end-to-end setups.

## Documentation
- [Quickstart Guide](docs/quickstart.md): Extensive introduction and core philosophies.
- [API Reference](docs/api-reference.md): Detailed API coverage and methods parameters.
- [Architecture Insights](docs/architecture.md): Design elements under the hood. 
- [Security Protocol](docs/security.md): Code validation and restricted runtime policies.

## License and Contributing
This repository thrives on community input. Check out our [Contribution Guidelines](CONTRIBUTING.md) to log issues or prepare Pull Requests.

Distributed under the **MIT License**. See `LICENSE` for more information.
