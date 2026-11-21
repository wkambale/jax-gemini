"""JAX distributed device mesh and NamedSharding helpers."""

from __future__ import annotations

import math
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def device_summary() -> dict[str, Any]:
    """Return discovered JAX accelerators, count, and default backend platform."""
    devices = jax.devices()
    return {
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "devices": [
            {"id": int(d.id), "platform": str(d.platform), "device_kind": str(d.device_kind)}
            for d in devices
        ],
    }


def create_device_mesh(
    axis_names: tuple[str, ...] = ("data",),
    mesh_shape: tuple[int, ...] | None = None,
) -> Mesh:
    """Create a JAX Mesh across available accelerators."""
    if not isinstance(axis_names, tuple | list) or len(axis_names) < 1:
        raise ValueError("axis_names must be a non-empty sequence of axis names.")
    axis_tuple = tuple(axis_names)
    if any(not isinstance(a, str) or not a.strip() for a in axis_tuple):
        raise ValueError("All axis_names must be non-empty strings.")
    if len(set(axis_tuple)) != len(axis_tuple):
        raise ValueError(f"axis_names must be unique, got {axis_tuple!r}.")

    raw_devices = jax.devices()
    num_devices = len(raw_devices)
    devices = np.array(raw_devices)
    if mesh_shape is None:
        if len(axis_tuple) == 1:
            devices = devices.reshape((num_devices,))
        else:
            shape = (num_devices,) + (1,) * (len(axis_tuple) - 1)
            devices = devices.reshape(shape)
    else:
        if not isinstance(mesh_shape, tuple | list) or len(mesh_shape) != len(axis_tuple):
            raise ValueError(
                f"mesh_shape length must match axis_names ({len(axis_tuple)}), "
                f"got {mesh_shape!r}."
            )
        if any(isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in mesh_shape):
            raise ValueError(f"mesh_shape must contain positive integers, got {mesh_shape!r}.")
        if math.prod(mesh_shape) != num_devices:
            raise ValueError(
                f"prod(mesh_shape)={math.prod(mesh_shape)} must equal available device count "
                f"({num_devices})."
            )
        devices = devices.reshape(tuple(mesh_shape))
    return Mesh(devices, axis_tuple)


def make_replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Create a fully replicated NamedSharding across ``mesh``."""
    spec_cls: Any = PartitionSpec
    return NamedSharding(mesh, spec_cls())


def make_data_sharding(mesh: Mesh, axis: str = "data") -> NamedSharding:
    """Create a batch-axis NamedSharding along ``axis`` on ``mesh``."""
    spec_cls: Any = PartitionSpec
    return NamedSharding(mesh, spec_cls(axis))
