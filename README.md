# jax-gemini

![PyPI version](https://img.shields.io/pypi/v/jax-gemini)
![Python Versions](https://img.shields.io/pypi/pyversions/jax-gemini)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

Natural language-driven JAX/Flax model building powered by Gemini.

## Quick Demo

```python
import jax_gemini as jg

jg.config.set({"gemini_api_key": "YOUR_KEY"})

# Build a model from natural language
model = jg.build("Build a 4-layer MLP for handwritten digit classification")

# Train it on your data
model, metrics = jg.train("Train for 10 epochs with Adam", dataset=(X_train, y_train))
print(f"Accuracy: {metrics['accuracy']:.2%}")   # Accuracy: 94.30%

# Refine the architecture conversationally
model = jg.modify("Add dropout with rate 0.2 between each layer")

# Save the checkpoint
jg.save("digit_classifier_v1")
```

## Installation

```bash
pip install jax-gemini
```

## Usage

### Configuration
```python
jg.config.set({
    "gemini_api_key": "...",
    "model_name": "gemini-3.1-pro",
    "temperature": 0.2,
    "max_retries": 3,
})
```

## How it works
jax-gemini parses your natural language input, passes it to the Gemini API which returns executable Python code making use of Flax NNX. It validates the code securely blocking bad imports, and executes it in a sandboxed namespace to instantiate the model object returning it to you.

## Contributing
See `CONTRIBUTING.md`

## License
MIT
