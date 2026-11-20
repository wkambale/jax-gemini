"""Flax NNX inspection and abstract shape-evaluation helpers."""

from typing import Any

import flax.nnx as nnx
import jax


def count_parameters(model: Any) -> int:
    """Return the total number of scalar parameters in an NNX or PyTree module."""
    params: Any
    if isinstance(model, nnx.Module):
        params = nnx.state(model, nnx.Param)
    elif callable(model):
        params = jax.tree_util.tree_leaves(model)
    else:
        raise TypeError(
            f"Expected a flax.nnx.Module or callable PyTree, got {type(model).__name__}."
        )
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


def eval_shape_summary(fn: Any, *example_specs: Any) -> dict[str, Any]:
    """Evaluate output shape and dtype via jax.eval_shape without device memory allocation."""
    abstract_out = jax.eval_shape(fn, *example_specs)
    leaves = jax.tree_util.tree_leaves(abstract_out)
    shapes = [
        {"shape": list(leaf.shape), "dtype": str(leaf.dtype)}
        for leaf in leaves
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype")
    ]
    return {"outputs": shapes, "leaf_count": len(leaves)}
