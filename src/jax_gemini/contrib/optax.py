"""Backward-compatibility shim for jax_gemini.contrib.optax."""

from jaxforge.contrib.optax import (
    available_optimizers,
    count_opt_steps,
    create_cosine_schedule,
    create_optimizer,
)

__all__ = [
    "available_optimizers",
    "count_opt_steps",
    "create_cosine_schedule",
    "create_optimizer",
]
