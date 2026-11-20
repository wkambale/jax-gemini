"""Optional Flax and Optax utilities built for jax-gemini workflows."""

from jax_gemini.contrib.flax import count_parameters, summarize_module
from jax_gemini.contrib.optax import available_optimizers, count_opt_steps, create_optimizer

__all__ = [
    "available_optimizers",
    "count_opt_steps",
    "count_parameters",
    "create_optimizer",
    "summarize_module",
]
