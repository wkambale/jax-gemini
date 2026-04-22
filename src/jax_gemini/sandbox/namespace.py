import jax
import jax.numpy as jnp
import flax
import flax.nnx as nnx
import optax
import numpy as np

try:
    import orbax.checkpoint as ocp
    _orbax_available = True
except ImportError:
    _orbax_available = False

import builtins

SAFE_BUILTINS = {
    name: getattr(builtins, name, None)
    for name in [
        "print", "range", "len", "enumerate", "zip", "map", "filter",
        "sorted", "reversed", "min", "max", "sum", "abs", "round",
        "int", "float", "str", "bool", "list", "dict", "tuple", "set",
        "isinstance", "hasattr", "getattr", "type", "repr",
        "True", "False", "None", "__import__",
        "Exception", "ValueError", "TypeError", "RuntimeError", "KeyError"
    ]
    if getattr(builtins, name, None) is not None
}

BASE_NAMESPACE: dict = {
    "jax": jax,
    "jnp": jnp,
    "flax": flax,
    "nnx": nnx,
    "optax": optax,
    "np": np,
    "__builtins__": SAFE_BUILTINS,
}

if _orbax_available:
    import orbax.checkpoint as ocp # type: ignore
    BASE_NAMESPACE["ocp"] = ocp
    BASE_NAMESPACE["orbax"] = __import__("orbax")


def get_execution_namespace() -> dict:
    """Return a fresh copy of the safe namespace for each execution."""
    return dict(BASE_NAMESPACE)
