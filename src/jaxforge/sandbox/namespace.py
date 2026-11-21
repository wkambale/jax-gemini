"""Isolated read-only sandbox namespaces and confined file I/O proxies."""

from __future__ import annotations

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

from jaxforge.codegen.validator import (
    BANNED_ATTRIBUTES,
    BANNED_NUMPY_ATTRIBUTES,
    INTENT_ALLOWED_IMPORTS,
    WHITELIST_IMPORTS,
    is_banned_module_path,
)

orbax: Any = None
ocp: Any = None
try:
    import orbax as _orbax_mod
    import orbax.checkpoint as _ocp_mod

    orbax = _orbax_mod
    ocp = _ocp_mod
    _orbax_available = True
except ImportError:
    _orbax_available = False

matplotlib: Any = None
plt: Any = None
try:
    import matplotlib as _mpl_mod
    import matplotlib.pyplot as _plt_mod

    matplotlib = _mpl_mod
    plt = _plt_mod
    _matplotlib_available = True
except ImportError:
    _matplotlib_available = False

eqx: Any = None
try:
    import equinox as _eqx_mod

    eqx = _eqx_mod
    _equinox_available = True
except ImportError:
    _equinox_available = False

grain: Any = None
try:
    import grain as _grain_mod

    grain = _grain_mod
    _grain_available = True
except ImportError:
    _grain_available = False

safetensors: Any = None
st_flax: Any = None
st_np: Any = None
try:
    import safetensors as _st_mod
    import safetensors.flax as _st_flax_mod
    import safetensors.numpy as _st_np_mod

    safetensors = _st_mod
    st_flax = _st_flax_mod
    st_np = _st_np_mod
    _safetensors_available = True
except ImportError:
    _safetensors_available = False

_BLOCKED_NUMPY_ATTRS: frozenset[str] = BANNED_NUMPY_ATTRIBUTES
_BLOCKED_PYPLOT_ATTRS: frozenset[str] = frozenset(
    {"switch_backend", "rc_file", "rc_file_defaults", "matplotlib_fname"}
)
_BLOCKED_MATPLOTLIB_ATTRS: frozenset[str] = frozenset(
    {"use", "rc_file", "rc_file_defaults", "matplotlib_fname", "backends", "backend_bases"}
)
_GLOBAL_BLOCKED_MODULE_ATTRS: frozenset[str] = frozenset(
    {
        "sys",
        "os",
        "io",
        "builtins",
        "subprocess",
        "pathlib",
        "shutil",
        "socket",
        "ctypes",
        "importlib",
        "pickle",
        "marshal",
        "tempfile",
        "glob",
        "multiprocessing",
    }
)

_ProxyEntry = tuple[
    ModuleType,
    dict[str, Any],
    frozenset[str],
    frozenset[str],
    dict[str, ModuleType],
]
_PROXY_STATE: weakref.WeakKeyDictionary[ModuleType, _ProxyEntry] = weakref.WeakKeyDictionary()
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
    """Read-only proxy around an allowed module enforcing submodule allowlists and overrides."""

    def __init__(
        self,
        target: ModuleType,
        overrides: dict[str, Any] | None = None,
        blocked: frozenset[str] = frozenset(),
        allowed_imports: frozenset[str] = WHITELIST_IMPORTS,
        safe_modules: dict[str, ModuleType] | None = None,
    ) -> None:
        super().__init__(target.__name__)
        _PROXY_STATE[self] = (
            target,
            dict(overrides or {}),
            frozenset(blocked),
            frozenset(allowed_imports),
            safe_modules if safe_modules is not None else {},
        )

    def __getattr__(self, name: str) -> Any:
        if (
            name in _PROXY_SELF_GUARD
            or name in _GLOBAL_BLOCKED_MODULE_ATTRS
            or name in BANNED_ATTRIBUTES
        ):
            raise AttributeError(f"Attribute '{name}' is blocked in sandbox.")
        target, overrides, blocked, allowed_imports, safe_modules = _PROXY_STATE[self]
        dotted_attr = f"{target.__name__}.{name}"
        if name in blocked or is_banned_module_path(dotted_attr):
            raise AttributeError(f"Attribute '{name}' is blocked in sandbox.")
        if name in overrides:
            return overrides[name]
        if dotted_attr in safe_modules:
            return safe_modules[dotted_attr]
        attr = getattr(target, name)
        if isinstance(attr, ModuleType):
            if isinstance(attr, _SafeModuleProxy):
                return attr
            attr_name = attr.__name__
            attr_root = attr_name.split(".")[0]
            if attr_root not in allowed_imports or is_banned_module_path(attr_name):
                raise AttributeError(
                    f"Module attribute '{name}' ('{attr_name}') is blocked in sandbox."
                )
            if attr_name in safe_modules:
                return safe_modules[attr_name]
            return _SafeModuleProxy(
                attr,
                allowed_imports=allowed_imports,
                safe_modules=safe_modules,
            )
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Sandbox module proxies are read-only.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Sandbox module proxies are read-only.")


def _wrap_path_io(
    fn: Callable[..., Any],
    allowed_dirs: tuple[Path, ...],
    kw_names: tuple[str, ...] = ("fname", "file", "directory", "path", "filename"),
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


def _wrap_multi_or_single_path_io(
    fn: Callable[..., Any],
    allowed_dirs: tuple[Path, ...],
    kw_names: tuple[str, ...] = ("paths", "path", "filenames", "filename"),
) -> Callable[..., Any]:
    def _confine_arg(val: Any) -> Any:
        if isinstance(val, (str, os.PathLike)):
            return _resolve_confined_path(val, allowed_dirs)
        if isinstance(val, (list, tuple)):
            return type(val)(_resolve_confined_path(item, allowed_dirs) for item in val)
        raise PermissionError("Sandbox data source requires a path string or sequence of paths.")

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        if args:
            return fn(_confine_arg(args[0]), *args[1:], **kwargs)
        for key in kw_names:
            if key in kwargs:
                new_kwargs = dict(kwargs)
                new_kwargs[key] = _confine_arg(kwargs[key])
                return fn(*args, **new_kwargs)
        raise PermissionError("Sandbox data source requires a confined path argument.")

    return _wrapped


def _create_safe_numpy(
    allowed_dirs: tuple[Path, ...],
    allowed_imports: frozenset[str],
    safe_modules: dict[str, ModuleType],
) -> ModuleType:
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
    return _SafeModuleProxy(
        np,
        overrides=overrides,
        blocked=_BLOCKED_NUMPY_ATTRS,
        allowed_imports=allowed_imports,
        safe_modules=safe_modules,
    )


def _confine_figure(fig: Any, allowed_dirs: tuple[Path, ...]) -> Any:
    """Confine Figure.savefig on pyplot-created Figure objects to allowed_dirs."""
    raw_savefig = getattr(fig, "savefig", None)
    if callable(raw_savefig):
        object.__setattr__(fig, "savefig", _wrap_path_io(raw_savefig, allowed_dirs, ("fname",)))
    return fig


def _create_safe_pyplot(
    allowed_dirs: tuple[Path, ...],
    allowed_imports: frozenset[str],
    safe_modules: dict[str, ModuleType],
) -> ModuleType:
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
    return _SafeModuleProxy(
        plt,
        overrides=overrides,
        blocked=_BLOCKED_PYPLOT_ATTRS,
        allowed_imports=allowed_imports,
        safe_modules=safe_modules,
    )


def _create_safe_orbax_checkpoint(
    allowed_dirs: tuple[Path, ...],
    allowed_imports: frozenset[str],
    safe_modules: dict[str, ModuleType],
) -> ModuleType:
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
    return _SafeModuleProxy(
        ocp,
        overrides=overrides,
        blocked=frozenset({"test_utils"}),
        allowed_imports=allowed_imports,
        safe_modules=safe_modules,
    )


def _create_safe_equinox(
    eqx_mod: ModuleType,
    allowed_dirs: tuple[Path, ...],
    intent: str,
    allowed_imports: frozenset[str],
    safe_modules: dict[str, ModuleType],
) -> ModuleType:
    overrides: dict[str, Any] = {}
    blocked: set[str] = set()
    if intent == "save" and hasattr(eqx_mod, "tree_serialise_leaves"):
        overrides["tree_serialise_leaves"] = _wrap_path_io(
            eqx_mod.tree_serialise_leaves, allowed_dirs, ("path_or_file",)
        )
        blocked.add("tree_deserialise_leaves")
    elif intent == "load" and hasattr(eqx_mod, "tree_deserialise_leaves"):
        overrides["tree_deserialise_leaves"] = _wrap_path_io(
            eqx_mod.tree_deserialise_leaves, allowed_dirs, ("path_or_file",)
        )
        blocked.add("tree_serialise_leaves")
    else:
        blocked.update({"tree_serialise_leaves", "tree_deserialise_leaves"})
    return _SafeModuleProxy(
        eqx_mod,
        overrides=overrides,
        blocked=frozenset(blocked),
        allowed_imports=allowed_imports,
        safe_modules=safe_modules,
    )


def _create_safe_grain(
    grain_mod: ModuleType,
    allowed_dirs: tuple[Path, ...],
    allowed_imports: frozenset[str],
    safe_modules: dict[str, ModuleType],
) -> ModuleType:
    source_names = ("ArrayRecordDataSource", "ParquetDataSource", "TfRecordDataSource")
    overrides: dict[str, Any] = {}
    for cls_name in source_names:
        raw_cls = getattr(grain_mod, cls_name, None)
        if callable(raw_cls):
            overrides[cls_name] = _wrap_multi_or_single_path_io(raw_cls, allowed_dirs)
    py_sub = getattr(grain_mod, "python", None)
    if isinstance(py_sub, ModuleType):
        py_overrides: dict[str, Any] = {}
        for cls_name in source_names:
            raw_cls = getattr(py_sub, cls_name, None)
            if callable(raw_cls):
                py_overrides[cls_name] = _wrap_multi_or_single_path_io(raw_cls, allowed_dirs)
        safe_py = _SafeModuleProxy(
            py_sub,
            overrides=py_overrides,
            blocked=frozenset({"multiprocessing"}),
            allowed_imports=allowed_imports,
            safe_modules=safe_modules,
        )
        overrides["python"] = safe_py
        safe_modules["grain.python"] = safe_py
    return _SafeModuleProxy(
        grain_mod,
        overrides=overrides,
        blocked=frozenset({"multiprocessing"}),
        allowed_imports=allowed_imports,
        safe_modules=safe_modules,
    )


def _create_safe_safetensors(
    st_mod: ModuleType,
    st_np_mod: ModuleType,
    st_flax_mod: ModuleType,
    allowed_dirs: tuple[Path, ...],
    intent: str,
    allowed_imports: frozenset[str],
    safe_modules: dict[str, ModuleType],
) -> ModuleType:
    sub_blocked = frozenset({"save_file", "save"}) if intent == "load" else frozenset()
    np_overrides: dict[str, Any] = {
        "load_file": _wrap_path_io(st_np_mod.load_file, allowed_dirs, ("filename",)),
    }
    flax_overrides: dict[str, Any] = {
        "load_file": _wrap_path_io(st_flax_mod.load_file, allowed_dirs, ("filename",)),
    }
    if intent == "save":
        np_overrides["save_file"] = _wrap_path_io(
            st_np_mod.save_file, allowed_dirs, ("filename",)
        )
        flax_overrides["save_file"] = _wrap_path_io(
            st_flax_mod.save_file, allowed_dirs, ("filename",)
        )

    safe_st_np = _SafeModuleProxy(
        st_np_mod,
        overrides=np_overrides,
        blocked=sub_blocked,
        allowed_imports=allowed_imports,
        safe_modules=safe_modules,
    )
    safe_st_flax = _SafeModuleProxy(
        st_flax_mod,
        overrides=flax_overrides,
        blocked=sub_blocked,
        allowed_imports=allowed_imports,
        safe_modules=safe_modules,
    )
    safe_modules["safetensors.numpy"] = safe_st_np
    safe_modules["safetensors.flax"] = safe_st_flax

    root_overrides: dict[str, Any] = {"numpy": safe_st_np, "flax": safe_st_flax}
    if hasattr(st_mod, "safe_open"):
        raw_safe_open = st_mod.safe_open

        def _confined_safe_open(filename: Any, *args: Any, **kwargs: Any) -> Any:
            safe_file = _resolve_confined_path(filename, allowed_dirs)
            return raw_safe_open(safe_file, *args, **kwargs)

        root_overrides["safe_open"] = _confined_safe_open

    root_blocked = sub_blocked | frozenset(
        {"_safetensors_rust", "torch", "tensorflow", "paddle"}
    )
    return _SafeModuleProxy(
        st_mod,
        overrides=root_overrides,
        blocked=root_blocked,
        allowed_imports=allowed_imports,
        safe_modules=safe_modules,
    )


def _make_safe_import(
    allowed_imports: frozenset[str],
    safe_modules: dict[str, ModuleType],
) -> Callable[..., ModuleType]:
    def _safe_import_impl(
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
            if (
                part in BANNED_ATTRIBUTES
                or part in _GLOBAL_BLOCKED_MODULE_ATTRS
                or (root == "numpy" and part in BANNED_NUMPY_ATTRIBUTES)
            ):
                raise ImportError(f"Import '{name}' is not allowed.")
        for sym in fromlist or ():
            if (
                sym == "*"
                or is_banned_module_path(f"{name}.{sym}")
                or sym in BANNED_ATTRIBUTES
                or sym in _GLOBAL_BLOCKED_MODULE_ATTRS
                or (root == "numpy" and sym in BANNED_NUMPY_ATTRIBUTES)
            ):
                raise ImportError(f"Import of '{sym}' from '{name}' is not allowed.")

        if root in safe_modules:
            if "." in name and name not in safe_modules:
                builtins.__import__(name, globals, locals, (), level)
            if not fromlist:
                return safe_modules[root]
            if name in safe_modules:
                return safe_modules[name]
            cur: Any = safe_modules[root]
            for part in name.split(".")[1:]:
                cur = getattr(cur, part)
            if isinstance(cur, ModuleType) and not isinstance(cur, _SafeModuleProxy):
                return _SafeModuleProxy(
                    cur, allowed_imports=allowed_imports, safe_modules=safe_modules
                )
            if isinstance(cur, ModuleType):
                return cur
            raise ImportError(f"Import '{name}' is not allowed.")

        mod = builtins.__import__(name, globals, locals, fromlist, level)
        if isinstance(mod, ModuleType) and not isinstance(mod, _SafeModuleProxy):
            return _SafeModuleProxy(
                mod, allowed_imports=allowed_imports, safe_modules=safe_modules
            )
        return mod

    return _safe_import_impl


def _safe_import(
    name: str,
    globals: Mapping[str, object] | None = None,
    locals: Mapping[str, object] | None = None,
    fromlist: Sequence[str] | None = None,
    level: int = 0,
) -> ModuleType:
    """Default safe __import__ scoped to the 'build' execution namespace."""
    importer = get_execution_namespace(intent="build")["__builtins__"]["__import__"]
    res: ModuleType = importer(name, globals, locals, fromlist, level)
    return res


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
    safe_modules: dict[str, ModuleType] = {}

    safe_jnp = _SafeModuleProxy(jnp, allowed_imports=allowed_imports, safe_modules=safe_modules)
    safe_jax = _SafeModuleProxy(
        jax,
        overrides={"numpy": safe_jnp},
        blocked=frozenset({"profiler"}),
        allowed_imports=allowed_imports,
        safe_modules=safe_modules,
    )
    safe_nnx = _SafeModuleProxy(nnx, allowed_imports=allowed_imports, safe_modules=safe_modules)
    safe_flax = _SafeModuleProxy(
        flax,
        overrides={"nnx": safe_nnx},
        allowed_imports=allowed_imports,
        safe_modules=safe_modules,
    )
    safe_optax = _SafeModuleProxy(optax, allowed_imports=allowed_imports, safe_modules=safe_modules)
    safe_np = _create_safe_numpy(resolved_dirs, allowed_imports, safe_modules)

    safe_modules.update(
        {
            "jax": safe_jax,
            "jax.numpy": safe_jnp,
            "flax": safe_flax,
            "flax.nnx": safe_nnx,
            "optax": safe_optax,
            "numpy": safe_np,
        }
    )

    namespace: dict[str, Any] = {
        "jax": safe_jax,
        "jnp": safe_jnp,
        "flax": safe_flax,
        "nnx": safe_nnx,
        "optax": safe_optax,
        "np": safe_np,
        "__name__": "jaxforge_sandbox",
    }

    if _equinox_available and eqx is not None and "equinox" in allowed_imports:
        safe_eqx = _create_safe_equinox(eqx, resolved_dirs, intent, allowed_imports, safe_modules)
        safe_modules["equinox"] = safe_eqx
        namespace["eqx"] = safe_eqx
        namespace["equinox"] = safe_eqx

    if _grain_available and grain is not None and "grain" in allowed_imports:
        safe_grain = _create_safe_grain(grain, resolved_dirs, allowed_imports, safe_modules)
        safe_modules["grain"] = safe_grain
        namespace["grain"] = safe_grain

    if _orbax_available and orbax is not None and "orbax" in allowed_imports:
        safe_ocp = _create_safe_orbax_checkpoint(resolved_dirs, allowed_imports, safe_modules)
        safe_orbax = _SafeModuleProxy(
            orbax,
            overrides={"checkpoint": safe_ocp},
            allowed_imports=allowed_imports,
            safe_modules=safe_modules,
        )
        safe_modules["orbax"] = safe_orbax
        safe_modules["orbax.checkpoint"] = safe_ocp
        namespace["ocp"] = safe_ocp
        namespace["orbax"] = safe_orbax

    if (
        _safetensors_available
        and safetensors is not None
        and st_np is not None
        and st_flax is not None
        and "safetensors" in allowed_imports
    ):
        safe_st = _create_safe_safetensors(
            safetensors, st_np, st_flax, resolved_dirs, intent, allowed_imports, safe_modules
        )
        safe_modules["safetensors"] = safe_st
        namespace["safetensors"] = safe_st

    if _matplotlib_available and matplotlib is not None and "matplotlib" in allowed_imports:
        safe_plt = _create_safe_pyplot(resolved_dirs, allowed_imports, safe_modules)
        safe_mpl = _SafeModuleProxy(
            matplotlib,
            overrides={"pyplot": safe_plt},
            blocked=_BLOCKED_MATPLOTLIB_ATTRS,
            allowed_imports=allowed_imports,
            safe_modules=safe_modules,
        )
        safe_modules["matplotlib"] = safe_mpl
        safe_modules["matplotlib.pyplot"] = safe_plt
        namespace["plt"] = safe_plt
        namespace["matplotlib"] = safe_mpl

    builtins_copy = dict(SAFE_BUILTINS)
    builtins_copy["__import__"] = _make_safe_import(allowed_imports, safe_modules)
    namespace["__builtins__"] = builtins_copy
    return namespace
