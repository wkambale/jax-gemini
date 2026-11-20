import flax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

try:
    import orbax.checkpoint as ocp

    _orbax_available = True
except ImportError:
    _orbax_available = False

try:
    import matplotlib.pyplot as plt

    _matplotlib_available = True
except ImportError:
    _matplotlib_available = False

import builtins
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any

from jax_gemini.codegen.validator import WHITELIST_IMPORTS


def _safe_import(
    name: str,
    globals: Mapping[str, object] | None = None,
    locals: Mapping[str, object] | None = None,
    fromlist: Sequence[str] = (),
    level: int = 0,
) -> ModuleType:
    """Restricted __import__ enforcing WHITELIST_IMPORTS and disallowing relative imports."""
    if level != 0:
        raise ImportError("Relative imports are not allowed.")
    root = name.split(".")[0]
    if root not in WHITELIST_IMPORTS:
        raise ImportError(f"Import '{name}' is not allowed.")
    return builtins.__import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(builtins, name, None)
    for name in [
        "print",
        "range",
        "len",
        "enumerate",
        "zip",
        "map",
        "filter",
        "sorted",
        "reversed",
        "min",
        "max",
        "sum",
        "abs",
        "round",
        "int",
        "float",
        "str",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
        "isinstance",
        "hasattr",
        "repr",
        "True",
        "False",
        "None",
        "__build_class__",
        "Exception",
        "ValueError",
        "TypeError",
        "RuntimeError",
        "KeyError",
    ]
    if getattr(builtins, name, None) is not None
}
SAFE_BUILTINS["__import__"] = _safe_import

BASE_NAMESPACE: dict[str, Any] = {
    "jax": jax,
    "jnp": jnp,
    "flax": flax,
    "nnx": nnx,
    "optax": optax,
    "np": np,
    "__name__": "jax_gemini_sandbox",
    "__builtins__": SAFE_BUILTINS,
}

if _orbax_available:
    import orbax.checkpoint as ocp

    BASE_NAMESPACE["ocp"] = ocp
    BASE_NAMESPACE["orbax"] = __import__("orbax")

if _matplotlib_available:
    BASE_NAMESPACE["plt"] = plt
    BASE_NAMESPACE["matplotlib"] = __import__("matplotlib")


def get_execution_namespace() -> dict[str, Any]:
    """Return a fresh copy of the safe namespace for each execution."""
    return dict(BASE_NAMESPACE)
