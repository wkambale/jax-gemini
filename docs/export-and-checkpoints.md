# Export and checkpoints

Once you have prototyped and tuned a model in Jax Forge, you can persist checkpoints cleanly or materialize the generated modules as a standalone, installable Python repository.

## Exporting a standalone package

`jf.export_project()` writes generated build, training, and evaluation modules into an isolated Python project layout with packaging metadata and smoke tests:

```python
import jaxforge as jf

jf.build("Build an Equinox or Flax classifier")
project_dir = jf.export_project("./my_model_repo", package_name="my_model")
print("Exported to:", project_dir)
```

Generated output layout:

```text
my_model_repo/
  pyproject.toml
  README.md
  src/
    my_model/
      __init__.py
      model.py
      train.py
      eval.py
  tests/
    test_smoke.py
```

## Confined checkpoint persistence

`jf.save(name)` and `jf.load(name)` persist and restore model state inside `config.checkpoint_dir` (`./jaxforge_checkpoints` by default):

```python
path = jf.save("mnist_v1")
restored = jf.load("mnist_v1")
```

All checkpoint names are resolved relative to the configured checkpoint root. Path traversal segments (`..`), null bytes, URI schemes, absolute paths, and symlink escapes outside `checkpoint_dir` raise `JaxForgeCheckpointError` before touching disk.

## Optional Equinox and Safetensors support

When generating Equinox architectures (`framework="equinox"`) or restoring weights from `.safetensors` archives, Jax Forge validates allowed imports and applies confined directory checks on `safetensors` loaders inside the sandbox namespace.
