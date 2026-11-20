import builtins
import os
import weakref
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

import flax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_gemini.codegen.validator import (
    BANNED_ATTRIBUTES,
    BANNED_NUMPY_ATTRIBUTES,
    INTENT_ALLOWED_IMPORTS,
    WHITELIST_IMPORTS,
    is_banned_module_path,
)

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

_BLOCKED_NUMPY_ATTRS: frozenset[str] = BANNED_NUMPY_ATTRIBUTES
_BLOCKED_PYPLOT_ATTRS: frozenset[str] = frozenset(
    {"switch_backend", "rc_file", "rc_file_defaults", "matplotlib_fname"}
)
_BLOCKED_MATPLOTLIB_ATTRS: frozenset[str] = frozenset(
    {"use", "rc_file", "rc_file_defaults", "matplotlib_fname", "backends", "backend_bases"}
)

_PROXY_STATE: weakref.WeakKeyDictionary[
    ModuleType, tuple[ModuleType, dict[str, Any], frozenset[str]]
] = weakref.WeakKeyDictionary()
_PROXY_SELF_GUARD = frozenset({"_target", "_overrides", "_blocked"})


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
        _PROXY_STATE[self] = (target, dict(overrides or {}), frozenset(blocked))

    def __getattr__(self, name: str) -> Any:
        if name in _PROXY_SELF_GUARD:
            raise AttributeError(f"Attribute '{name}' is blocked in sandbox.")
        target, overrides, blocked = _PROXY_STATE[self]
        if name in blocked or is_banned_module_path(f"{target.__name__}.{name}"):
            raise AttributeError(f"Attribute '{name}' is blocked in sandbox.")
        if name in overrides:
            return overrides[name]
        attr = getattr(target, name)
        if isinstance(attr, ModuleType) and not isinstance(attr, _SafeModuleProxy):
            return _SafeModuleProxy(attr)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Sandbox module proxies are read-only.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Sandbox module proxies are read-only.")


def _wrap_path_io(
    fn: Callable[..., Any],
    allowed_dirs: tuple[Path, ...],
    kw_names: tuple[str, ...] = ("fname", "file", "directory", "path"),
) -> Callable[..., Any]:
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        if args:
            safe_path = _resolve_confined_path(args[0], allowed_dirs)
            return fn(safe_path, *args[1:], **kwargs)
        for key in kw_names:
            if key in kwargs:
                new_kwargs = dict(kwargs)
                new_kwargs[key] = _resolve_confined_path(kwargs[key], allowed_dirs)
                return fn(*args, **new_kwargs)
        raise PermissionError("Sandbox I/O requires a confined path argument.")

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


def _confine_figure(fig: Any, allowed_dirs: tuple[Path, ...]) -> Any:
    """Confine Figure.savefig on pyplot-created Figure objects to allowed_dirs."""
    raw_savefig = getattr(fig, "savefig", None)
    if callable(raw_savefig):
        object.__setattr__(fig, "savefig", _wrap_path_io(raw_savefig, allowed_dirs, ("fname",)))
    return fig


def _create_safe_pyplot(allowed_dirs: tuple[Path, ...]) -> ModuleType:
    def _safe_figure(*args: Any, **kwargs: Any) -> Any:
        return _confine_figure(plt.figure(*args, **kwargs), allowed_dirs)

    def _safe_subplots(*args: Any, **kwargs: Any) -> Any:
        res = plt.subplots(*args, **kwargs)
        if isinstance(res, tuple) and len(res) >= 1:
            _confine_figure(res[0], allowed_dirs)
        return res

    overrides: dict[str, Any] = {
        "savefig": _wrap_path_io(plt.savefig, allowed_dirs, ("fname",)),
        "imread": _wrap_path_io(plt.imread, allowed_dirs, ("fname",)),
        "imsave": _wrap_path_io(plt.imsave, allowed_dirs, ("fname",)),
        "figure": _safe_figure,
        "subplots": _safe_subplots,
    }
    return _SafeModuleProxy(plt, overrides=overrides, blocked=_BLOCKED_PYPLOT_ATTRS)


def _create_safe_orbax_checkpoint(allowed_dirs: tuple[Path, ...]) -> ModuleType:
    checkpointers: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()

    def _wrap_checkpointer_cls(cls: type[Any]) -> Callable[..., Any]:
        class _ConfinedCheckpointer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                checkpointers[self] = cls(*args, **kwargs)

            def save(self, directory: Any, *args: Any, **kwargs: Any) -> Any:
                safe_dir = _resolve_confined_path(directory, allowed_dirs)
                return checkpointers[self].save(safe_dir, *args, **kwargs)

            def restore(self, directory: Any, *args: Any, **kwargs: Any) -> Any:
                safe_dir = _resolve_confined_path(directory, allowed_dirs)
                return checkpointers[self].restore(safe_dir, *args, **kwargs)

            def __getattr__(self, attr: str) -> Any:
                if attr.startswith("_"):
                    raise AttributeError(f"Attribute '{attr}' is blocked in sandbox.")
                return getattr(checkpointers[self], attr)

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
        fromlist: Sequence[str] | None = None,
        level: int = 0,
    ) -> ModuleType:
        """Restricted __import__ enforcing allowed_imports and returning safe proxies."""
        if level != 0:
            raise ImportError("Relative imports are not allowed.")
        root = name.split(".")[0]
        if root not in allowed_imports or is_banned_module_path(name):
            raise ImportError(f"Import '{name}' is not allowed.")
        for part in name.split(".")[1:]:
            if part in BANNED_ATTRIBUTES or (root == "numpy" and part in BANNED_NUMPY_ATTRIBUTES):
                raise ImportError(f"Import '{name}' is not allowed.")
        for sym in fromlist or ():
            if (
                sym == "*"
                or is_banned_module_path(f"{name}.{sym}")
                or sym in BANNED_ATTRIBUTES
                or (root == "numpy" and sym in BANNED_NUMPY_ATTRIBUTES)
            ):
                raise ImportError(f"Import of '{sym}' from '{name}' is not allowed.")

        if root in safe_modules:
            if "." in name and name not in safe_modules:
                # Ensure intermediate submodules exist before resolving via root proxy.
                builtins.__import__(name, globals, locals, (), level)
            if not fromlist:
                return safe_modules[root]
            if name in safe_modules:
                return safe_modules[name]
            cur: Any = safe_modules[root]
            for part in name.split(".")[1:]:
                cur = getattr(cur, part)
            if isinstance(cur, ModuleType) and not isinstance(cur, _SafeModuleProxy):
                return _SafeModuleProxy(cur)
            if isinstance(cur, ModuleType):
                return cur
            raise ImportError(f"Import '{name}' is not allowed.")

        mod = builtins.__import__(name, globals, locals, fromlist, level)
        if isinstance(mod, ModuleType) and not isinstance(mod, _SafeModuleProxy):
            return _SafeModuleProxy(mod)
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


_RAW_SAFE_BUILTINS: dict[str, Any] = {
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
_RAW_SAFE_BUILTINS["__import__"] = _safe_import
SAFE_BUILTINS: Mapping[str, Any] = MappingProxyType(_RAW_SAFE_BUILTINS)

BASE_NAMESPACE: Mapping[str, Any] = MappingProxyType(
    {
        "jax": _SafeModuleProxy(jax),
        "jnp": _SafeModuleProxy(jnp),
        "flax": _SafeModuleProxy(flax),
        "nnx": _SafeModuleProxy(nnx),
        "optax": _SafeModuleProxy(optax),
        "np": _SafeModuleProxy(np),
        "__name__": "jax_gemini_sandbox",
        "__builtins__": SAFE_BUILTINS,
    }
)


def get_execution_namespace(
    intent: str = "build",
    allowed_dirs: tuple[Path | str, ...] | None = None,
) -> dict[str, Any]:
    """Return a fresh, isolated sandbox namespace scoped to intent and allowed directories."""
    resolved_dirs = (
        tuple(Path(d).resolve() for d in allowed_dirs)
        if allowed_dirs
        else (Path.cwd().resolve(),)
    )
    allowed_imports = INTENT_ALLOWED_IMPORTS.get(intent, WHITELIST_IMPORTS)

    safe_jnp = _SafeModuleProxy(jnp)
    safe_jax = _SafeModuleProxy(jax, overrides={"numpy": safe_jnp})
    safe_nnx = _SafeModuleProxy(nnx)
    safe_flax = _SafeModuleProxy(flax, overrides={"nnx": safe_nnx})
    safe_optax = _SafeModuleProxy(optax)
    safe_np = _create_safe_numpy(resolved_dirs)

    safe_modules: dict[str, ModuleType] = {
        "jax": safe_jax,
        "jax.numpy": safe_jnp,
        "flax": safe_flax,
        "flax.nnx": safe_nnx,
        "optax": safe_optax,
        "numpy": safe_np,
    }

    namespace: dict[str, Any] = {
        "jax": safe_jax,
        "jnp": safe_jnp,
        "flax": safe_flax,
        "nnx": safe_nnx,
        "optax": safe_optax,
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
        safe_mpl = _SafeModuleProxy(
            matplotlib,
            overrides={"pyplot": safe_plt},
            blocked=_BLOCKED_MATPLOTLIB_ATTRS,
        )
        safe_modules["matplotlib"] = safe_mpl
        safe_modules["matplotlib.pyplot"] = safe_plt
        namespace["plt"] = safe_plt
        namespace["matplotlib"] = safe_mpl

    builtins_copy = dict(SAFE_BUILTINS)
    builtins_copy["__import__"] = _make_safe_import(allowed_imports, safe_modules)
    namespace["__builtins__"] = builtins_copy
    return namespace
