"""Backward-compatibility shim for jax_gemini.contrib."""

from jaxforge.contrib import (
    InMemoryTracker,
    JsonlTracker,
    MetricsTracker,
    available_optimizers,
    count_opt_steps,
    count_parameters,
    create_cosine_schedule,
    create_device_mesh,
    create_optimizer,
    create_tracker,
    device_summary,
    eval_shape_summary,
    make_data_sharding,
    make_replicated_sharding,
    summarize_module,
)

__all__ = [
    "InMemoryTracker",
    "JsonlTracker",
    "MetricsTracker",
    "available_optimizers",
    "count_opt_steps",
    "count_parameters",
    "create_cosine_schedule",
    "create_device_mesh",
    "create_optimizer",
    "create_tracker",
    "device_summary",
    "eval_shape_summary",
    "make_data_sharding",
    "make_replicated_sharding",
    "summarize_module",
]
