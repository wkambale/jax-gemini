"""JAX distributed device mesh and NamedSharding helpers."""

from __future__ import annotations

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
    devices = np.array(jax.devices())
    if mesh_shape is None:
        if len(axis_names) == 1:
            devices = devices.reshape((len(devices),))
        else:
            shape = (len(devices),) + (1,) * (len(axis_names) - 1)
            devices = devices.reshape(shape)
    else:
        devices = devices.reshape(mesh_shape)
    return Mesh(devices, axis_names)


def make_replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Create a fully replicated NamedSharding across ``mesh``."""
    spec_cls: Any = PartitionSpec
    return NamedSharding(mesh, spec_cls())


def make_data_sharding(mesh: Mesh, axis: str = "data") -> NamedSharding:
    """Create a batch-axis NamedSharding along ``axis`` on ``mesh``."""
    spec_cls: Any = PartitionSpec
    return NamedSharding(mesh, spec_cls(axis))
