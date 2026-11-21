"""Backward-compatibility shim for jax_gemini.contrib.sharding."""

from jaxforge.contrib.sharding import (
    create_device_mesh,
    device_summary,
    make_data_sharding,
    make_replicated_sharding,
)

__all__ = [
    "create_device_mesh",
    "device_summary",
    "make_data_sharding",
    "make_replicated_sharding",
]
