# Quickstart Guide

This guide introduces you to the core workflows for using `jax-gemini`.

## Prerequisites
Ensure that you have installed the package via [PyPI (`jax-gemini`)](https://pypi.org/project/jax-gemini/0.1.0/) either in your virtual environment or Jupyter notebook.
Before digging deeply into the logic, ensure that you have initialized the system credentials. 
Obtain a valid **Google Gemini API Key** and set it safely inside operations:
```python
import jax_gemini as jg
import os

# Automatically pulls from os environment by default or overrides manually
jg.config.set({
    "gemini_api_key": os.environ.get("GEMINI_API_KEY"),
    "model_name": "gemini-3.1-pro", 
    "temperature": 0.2
})
```

## Step 1: Building a Model
Building an architecture does not require you to write typical JAX syntax classes. Instead, we call the `jg.build()` methodology outlining specific needs.

```python
# Instantiates a valid jax.flax.nnx.Module!
my_model = jg.build(
    "Design a CNN model built for 32x32 color image classifications. Ensure it utilizes 2 Convolutional Blocks followed by a Maxpool and Dense heads."
)
```

## Step 2: Refining Architecture Iteratively
Once the object is created, we can modify it through iterative follow-ups keeping preceding steps in conversation memory contexts:
```python
# Modifies the active my_model in the session tree.
my_model = jg.modify("Refactor the model by integrating a Spatial Dropout wrapper between the CNN blocks to mitigate overfitting.")
```

## Step 3: Inspecting Outputs 
`jax-gemini` isn't a blackbox. If you ever feel curious about what the underlying code executing your objects looks like or what the LLM concluded, use our transparent methods:
```python
# Outputs plain Python String logic
print(jg.show_code())

# Outputs descriptive natural-language summaries representing changes
print(jg.explain())
```

## Step 4: Training & Evaluation
Training directly passes dataset tuple metrics returning validated models:
```python
tuned_model, metrics = jg.train(
    prompt="Train for 15 epochs leveraging categorical cross-entropy loss alongside learning rate schedulers",
    dataset=(X_train, y_train)
)

print("Loss observed: ", metrics['loss'])
```

Evaluate testing payloads:
```python
results = jg.evaluate("Run accuracy checks matching targets.", dataset=(X_test, y_test))
```

## Step 5: Checkpointing
Use the underlying state checkpoint methodologies powered natively through Orbax seamlessly.
```python
# Saves state to local project /jax_gemini_checkpoints path defaults.
jg.save("v1_cnn_model_production")
```
