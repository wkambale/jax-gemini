# Distributed sharding

Jax Forge provides lightweight wrappers over JAX's `Mesh`, `NamedSharding`, and `PartitionSpec` APIs under `jaxforge.contrib.sharding` to inspect accelerators and prepare batch-parallel or replicated placements.

## Inspecting discovered accelerators

```python
from jaxforge.contrib.sharding import device_summary

info = device_summary()
print(info)
# {'backend': 'cpu', 'device_count': 1, 'devices': [{'id': 0, 'platform': 'cpu', 'device_kind': 'cpu'}]}
```

From a terminal, run `jaxforge devices` to print the same accelerator inventory as formatted JSON.

## Creating meshes and NamedSharding rules

```python
from jaxforge.contrib.sharding import (
    create_device_mesh,
    make_data_sharding,
    make_replicated_sharding,
)

mesh = create_device_mesh(axis_names=("data",))
replicated = make_replicated_sharding(mesh)
data_sharding = make_data_sharding(mesh, axis="data")
```

For single-axis meshes (`("data",)`), `create_device_mesh()` arranges all discovered JAX devices across the data axis automatically. Multi-axis meshes (`("data", "model")`) accept an explicit `mesh_shape` tuple matching total device count.
