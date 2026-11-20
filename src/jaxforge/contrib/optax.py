"""Optax optimizer and schedule convenience helpers for jaxforge workflows."""

from collections.abc import Callable
from typing import Any

import optax

_OPTIMIZER_FACTORIES: dict[str, Callable[[float], optax.GradientTransformation]] = {
    "adam": optax.adam,
    "adamw": optax.adamw,
    "sgd": optax.sgd,
    "rmsprop": optax.rmsprop,
}


def available_optimizers() -> tuple[str, ...]:
    """Return the supported optimizer names in sorted order."""
    return tuple(sorted(_OPTIMIZER_FACTORIES))


def create_optimizer(
    name: str = "adam", learning_rate: float = 1e-3
) -> optax.GradientTransformation:
    """Construct an Optax gradient transformation by name with input validation."""
    if not isinstance(name, str):
        raise ValueError("Optimizer name must be a string.")
    key = name.strip().lower()
    factory = _OPTIMIZER_FACTORIES.get(key)
    if factory is None:
        supported = ", ".join(available_optimizers())
        raise ValueError(f"Unsupported optimizer '{name}'. Choose from: {supported}.")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, int | float)
        or not (learning_rate > 0)
    ):
        raise ValueError(f"learning_rate must be a positive number, got {learning_rate!r}.")
    return factory(float(learning_rate))


def create_cosine_schedule(
    peak_lr: float = 1e-3,
    warmup_steps: int = 100,
    total_steps: int = 1000,
) -> optax.Schedule:
    """Build a linear warmup + cosine decay learning-rate schedule."""
    if warmup_steps < 0 or total_steps <= warmup_steps:
        raise ValueError("Require 0 <= warmup_steps < total_steps.")
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0 if warmup_steps > 0 else peak_lr,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
    )


def count_opt_steps(opt_state: Any) -> int:
    """Extract the step count from an Optax state pytree, returning 0 if unavailable."""
    count = getattr(opt_state, "count", None)
    if count is not None and not callable(count):
        return int(count)
    if isinstance(opt_state, tuple | list):
        for item in opt_state:
            steps = count_opt_steps(item)
            if steps > 0:
                return steps
    return 0
