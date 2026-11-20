"""Flax NNX helper utilities for inspecting jax-gemini generated models."""

from typing import Any

import flax.nnx as nnx
import jax


def count_parameters(model: nnx.Module) -> int:
    """Return the total number of scalar parameters in a ``flax.nnx.Module``."""
    if not isinstance(model, nnx.Module):
        raise TypeError(f"Expected a flax.nnx.Module instance, got {type(model).__name__}.")
    params = nnx.state(model, nnx.Param)
    total = 0
    for leaf in jax.tree_util.tree_leaves(params):
        if hasattr(leaf, "size"):
            total += int(leaf.size)
    return total


def summarize_module(model: nnx.Module) -> dict[str, Any]:
    """Return a lightweight structural summary for a ``flax.nnx.Module``."""
    total_params = count_parameters(model)
    submodules = [
        name for name, value in vars(model).items() if isinstance(value, nnx.Module)
    ]
    return {
        "module_type": type(model).__name__,
        "parameter_count": total_params,
        "submodules": sorted(submodules),
    }
