import builtins
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import flax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_gemini.codegen.validator import INTENT_ALLOWED_IMPORTS, WHITELIST_IMPORTS

try:
    import orbax
    import orbax.checkpoint as ocp

    _orbax_available = True
except ImportError:
    _orbax_available = False

try:
    import matplotlib
    import matplotlib.pyplot as plt

    _matplotlib_available = True
except ImportError:
    _matplotlib_available = False

_BLOCKED_NUMPY_ATTRS: frozenset[str] = frozenset(
    {"fromfile", "memmap", "DataSource", "ctypeslib", "f2py", "distutils", "testing", "lib"}
)
_BLOCKED_PYPLOT_ATTRS: frozenset[str] = frozenset(
    {"switch_backend", "rc_file", "rc_file_defaults", "matplotlib_fname"}
)


def _resolve_confined_path(
    raw_path: str | os.PathLike[str], allowed_dirs: tuple[Path, ...]
) -> str:
    """Resolve raw_path and ensure it stays inside one of allowed_dirs."""
    if not isinstance(raw_path, (str, os.PathLike)):
        raise PermissionError("Sandbox I/O requires a filesystem path string, not a file handle.")
    path_str = os.fspath(raw_path)
    if not path_str or "\x00" in path_str or "://" in path_str:
        raise PermissionError(f"Disallowed sandbox path: '{path_str}'")
    base = allowed_dirs[0] if allowed_dirs else Path.cwd().resolve()
    raw = Path(path_str)
    candidate = (raw if raw.is_absolute() else base / raw).resolve()
    for root in allowed_dirs:
        resolved_root = root.resolve()
        if candidate == resolved_root or candidate.is_relative_to(resolved_root):
            return str(candidate)
    raise PermissionError(f"Sandbox path '{raw_path}' escapes allowed directories.")


class _SafeModuleProxy(ModuleType):
    """Read-only proxy around a whitelisted module overriding unsafe attributes."""

    def __init__(
        self,
        target: ModuleType,
        overrides: dict[str, Any] | None = None,
        blocked: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(target.__name__)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_overrides", overrides or {})
        object.__setattr__(self, "_blocked", blocked)

    def __getattr__(self, name: str) -> Any:
        blocked: frozenset[str] = object.__getattribute__(self, "_blocked")
        if name in blocked:
            raise AttributeError(f"Attribute '{name}' is blocked in sandbox.")
        overrides: dict[str, Any] = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        target: ModuleType = object.__getattribute__(self, "_target")
        return getattr(target, name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Sandbox module proxies are read-only.")


def _wrap_path_io(
    fn: Callable[..., Any], allowed_dirs: tuple[Path, ...]
) -> Callable[..., Any]:
    def _wrapped(path_arg: Any, *args: Any, **kwargs: Any) -> Any:
        safe_path = _resolve_confined_path(path_arg, allowed_dirs)
        return fn(safe_path, *args, **kwargs)

    return _wrapped


def _create_safe_numpy(allowed_dirs: tuple[Path, ...]) -> ModuleType:
    def _safe_load(
        file: Any, *args: Any, allow_pickle: bool = False, **kwargs: Any
    ) -> Any:
        if allow_pickle or (len(args) >= 2 and bool(args[1])):
            raise PermissionError("Pickle deserialization (allow_pickle=True) is blocked.")
        safe_path = _resolve_confined_path(file, allowed_dirs)
        if len(args) < 2:
            kwargs["allow_pickle"] = False
        return np.load(safe_path, *args, **kwargs)

    overrides: dict[str, Any] = {
        "load": _safe_load,
        "loadtxt": _wrap_path_io(np.loadtxt, allowed_dirs),
        "genfromtxt": _wrap_path_io(np.genfromtxt, allowed_dirs),
        "save": _wrap_path_io(np.save, allowed_dirs),
        "savez": _wrap_path_io(np.savez, allowed_dirs),
        "savez_compressed": _wrap_path_io(np.savez_compressed, allowed_dirs),
        "savetxt": _wrap_path_io(np.savetxt, allowed_dirs),
    }
    return _SafeModuleProxy(np, overrides=overrides, blocked=_BLOCKED_NUMPY_ATTRS)


def _create_safe_pyplot(allowed_dirs: tuple[Path, ...]) -> ModuleType:
    overrides: dict[str, Any] = {
        "savefig": _wrap_path_io(plt.savefig, allowed_dirs),
        "imread": _wrap_path_io(plt.imread, allowed_dirs),
        "imsave": _wrap_path_io(plt.imsave, allowed_dirs),
    }
    return _SafeModuleProxy(plt, overrides=overrides, blocked=_BLOCKED_PYPLOT_ATTRS)


def _create_safe_orbax_checkpoint(allowed_dirs: tuple[Path, ...]) -> ModuleType:
    def _wrap_checkpointer_cls(cls: type[Any]) -> Callable[..., Any]:
        class _ConfinedCheckpointer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._inner = cls(*args, **kwargs)

            def save(self, directory: Any, *args: Any, **kwargs: Any) -> Any:
                safe_dir = _resolve_confined_path(directory, allowed_dirs)
                return self._inner.save(safe_dir, *args, **kwargs)

            def restore(self, directory: Any, *args: Any, **kwargs: Any) -> Any:
                safe_dir = _resolve_confined_path(directory, allowed_dirs)
                return self._inner.restore(safe_dir, *args, **kwargs)

            def __getattr__(self, attr: str) -> Any:
                return getattr(self._inner, attr)

        return _ConfinedCheckpointer

    def _safe_checkpoint_manager(directory: Any, *args: Any, **kwargs: Any) -> Any:
        safe_dir = _resolve_confined_path(directory, allowed_dirs)
        return ocp.CheckpointManager(safe_dir, *args, **kwargs)

    overrides: dict[str, Any] = {
        "StandardCheckpointer": _wrap_checkpointer_cls(ocp.StandardCheckpointer),
        "Checkpointer": _wrap_checkpointer_cls(ocp.Checkpointer),
        "PyTreeCheckpointer": _wrap_checkpointer_cls(ocp.PyTreeCheckpointer),
        "AsyncCheckpointer": _wrap_checkpointer_cls(ocp.AsyncCheckpointer),
        "CheckpointManager": _safe_checkpoint_manager,
    }
    return _SafeModuleProxy(ocp, overrides=overrides, blocked=frozenset({"test_utils"}))


def _make_safe_import(
    allowed_imports: frozenset[str],
    safe_modules: dict[str, ModuleType],
) -> Callable[..., ModuleType]:
    def _safe_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        """Restricted __import__ enforcing allowed_imports and returning safe proxies."""
        if level != 0:
            raise ImportError("Relative imports are not allowed.")
        root = name.split(".")[0]
        if root not in allowed_imports:
            raise ImportError(f"Import '{name}' is not allowed.")
        if name in safe_modules:
            return safe_modules[name]
        mod = builtins.__import__(name, globals, locals, fromlist, level)
        if not fromlist and root in safe_modules:
            return safe_modules[root]
        return mod

    return _safe_import


def _safe_import(
    name: str,
    globals: Mapping[str, object] | None = None,
    locals: Mapping[str, object] | None = None,
    fromlist: Sequence[str] = (),
    level: int = 0,
) -> ModuleType:
    """Default safe __import__ enforcing WHITELIST_IMPORTS."""
    return _make_safe_import(WHITELIST_IMPORTS, {})(name, globals, locals, fromlist, level)


SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(builtins, name, None)
    for name in [
        "print",
        "range",
        "len",
        "enumerate",
        "zip",
        "map",
        "filter",
        "sorted",
        "reversed",
        "min",
        "max",
        "sum",
        "abs",
        "round",
        "int",
        "float",
        "str",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
        "isinstance",
        "hasattr",
        "repr",
        "True",
        "False",
        "None",
        "__build_class__",
        "Exception",
        "ValueError",
        "TypeError",
        "RuntimeError",
        "KeyError",
        "PermissionError",
    ]
    if getattr(builtins, name, None) is not None
}
SAFE_BUILTINS["__import__"] = _safe_import

BASE_NAMESPACE: dict[str, Any] = {
    "jax": jax,
    "jnp": jnp,
    "flax": flax,
    "nnx": nnx,
    "optax": optax,
    "np": np,
    "__name__": "jax_gemini_sandbox",
    "__builtins__": SAFE_BUILTINS,
}

if _orbax_available:
    BASE_NAMESPACE["ocp"] = ocp
    BASE_NAMESPACE["orbax"] = orbax

if _matplotlib_available:
    BASE_NAMESPACE["plt"] = plt
    BASE_NAMESPACE["matplotlib"] = matplotlib


def get_execution_namespace(
    intent: str = "build",
    allowed_dirs: tuple[Path | str, ...] | None = None,
) -> dict[str, Any]:
    """Return a fresh sandbox namespace scoped to intent and allowed directories."""
    resolved_dirs = (
        tuple(Path(d).resolve() for d in allowed_dirs)
        if allowed_dirs
        else (Path.cwd().resolve(),)
    )
    allowed_imports = INTENT_ALLOWED_IMPORTS.get(intent, WHITELIST_IMPORTS)

    safe_np = _create_safe_numpy(resolved_dirs)
    safe_modules: dict[str, ModuleType] = {"numpy": safe_np}

    namespace: dict[str, Any] = {
        "jax": jax,
        "jnp": jnp,
        "flax": flax,
        "nnx": nnx,
        "optax": optax,
        "np": safe_np,
        "__name__": "jax_gemini_sandbox",
    }

    if _orbax_available and "orbax" in allowed_imports:
        safe_ocp = _create_safe_orbax_checkpoint(resolved_dirs)
        safe_orbax = _SafeModuleProxy(orbax, overrides={"checkpoint": safe_ocp})
        safe_modules["orbax"] = safe_orbax
        safe_modules["orbax.checkpoint"] = safe_ocp
        namespace["ocp"] = safe_ocp
        namespace["orbax"] = safe_orbax

    if _matplotlib_available and "matplotlib" in allowed_imports:
        safe_plt = _create_safe_pyplot(resolved_dirs)
        safe_mpl = _SafeModuleProxy(matplotlib, overrides={"pyplot": safe_plt})
        safe_modules["matplotlib"] = safe_mpl
        safe_modules["matplotlib.pyplot"] = safe_plt
        namespace["plt"] = safe_plt
        namespace["matplotlib"] = safe_mpl

    builtins_copy = dict(SAFE_BUILTINS)
    builtins_copy["__import__"] = _make_safe_import(allowed_imports, safe_modules)
    namespace["__builtins__"] = builtins_copy
    return namespace
