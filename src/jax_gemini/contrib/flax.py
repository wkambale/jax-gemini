"""Backward-compatibility shim for jax_gemini.contrib.flax."""

from jaxforge.contrib.flax import count_parameters, eval_shape_summary, summarize_module

__all__ = ["count_parameters", "eval_shape_summary", "summarize_module"]
