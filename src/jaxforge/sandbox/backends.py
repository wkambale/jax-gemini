"""Pluggable execution backends: in-process thread, OS subprocess (rlimit), Docker, E2B, Modal."""

from __future__ import annotations

import base64
import io
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any

from jaxforge.codegen.validator import (
    BANNED_ATTRIBUTES,
    BANNED_MODULE_PREFIXES,
    BANNED_NUMPY_ATTRIBUTES,
    INTENT_ALLOWED_IMPORTS,
    WHITELIST_IMPORTS,
    CodeValidator,
)
from jaxforge.exceptions import JaxForgeConfigError, JaxForgeExecutionError
from jaxforge.sandbox.namespace import get_execution_namespace

_RESULT_MARKER = "__JAXFORGE_RESULT_B64__:"

_SAFE_UNPICKLE_BUILTINS = frozenset(
    {
        "dict",
        "list",
        "tuple",
        "set",
        "frozenset",
        "slice",
        "complex",
        "bytes",
        "bytearray",
        "str",
        "int",
        "float",
        "bool",
        "range",
    }
)
_SAFE_UNPICKLE_ROOTS = frozenset(
    {
        "numpy",
        "jax",
        "jaxlib",
        "flax",
        "optax",
        "equinox",
        "collections",
        "types",
        "functools",
        "jaxforge_sandbox",
    }
)


class _RestrictedUnpickler(pickle.Unpickler):
    """Confined unpickler rejecting arbitrary host callables such as os.system or eval."""

    def find_class(self, module: str, name: str) -> Any:
        root = module.split(".")[0]
        if module == "builtins":
            if name in _SAFE_UNPICKLE_BUILTINS:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"Blocked unpickling of builtin '{name}'.")
        if root == "_codecs" and name == "encode":
            return super().find_class(module, name)
        if root in _SAFE_UNPICKLE_ROOTS and not any(
            module == b or module.startswith(f"{b}.") for b in BANNED_MODULE_PREFIXES
        ):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Blocked unpickling of untrusted class '{module}.{name}'."
        )


def _reconstruct_nnx_envelope(obj: Any, mod_dict: dict[str, Any]) -> Any:
    """Rehydrate ('__jaxforge_nnx_leaves__', leaves) envelopes produced on closure-heavy Flax."""
    if (
        isinstance(obj, tuple)
        and len(obj) == 2
        and obj[0] == "__jaxforge_nnx_leaves__"
        and isinstance(obj[1], list)
    ):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp

        build_fn = mod_dict.get("build_model")
        if not callable(build_fn):
            raise RuntimeError("Cannot reconstruct nnx.Module without build_model in code.")
        host_m = build_fn(nnx.Rngs(0))
        graphdef, state = nnx.split(host_m)
        treedef = jax.tree_util.tree_structure(state)
        restored = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(x) for x in obj[1]])
        return nnx.merge(graphdef, restored)
    if isinstance(obj, tuple):
        return tuple(_reconstruct_nnx_envelope(item, mod_dict) for item in obj)
    return obj


def _safe_pickle_loads(data: bytes, code: str = "", intent: str = "build") -> Any:
    """Deserialize sandbox output safely, registering custom sandbox classes when needed."""
    needs_sandbox_mod = (b"jaxforge_sandbox" in data) or (b"__jaxforge_nnx_leaves__" in data)
    prev_mod = sys.modules.get("jaxforge_sandbox")
    mod_dict: dict[str, Any] = {}
    try:
        if needs_sandbox_mod and code:
            ns = get_execution_namespace(intent=intent)
            mod = ModuleType("jaxforge_sandbox")
            mod.__dict__.update(ns)
            sys.modules["jaxforge_sandbox"] = mod
            exec(code, mod.__dict__)
            mod_dict = mod.__dict__
        raw_obj = _RestrictedUnpickler(io.BytesIO(data)).load()
        return _reconstruct_nnx_envelope(raw_obj, mod_dict) if mod_dict else raw_obj
    except Exception as e:
        raise JaxForgeExecutionError(
            f"Failed to deserialize sandbox result safely: {e}",
            code=code,
        ) from e
    finally:
        if needs_sandbox_mod:
            if prev_mod is None:
                sys.modules.pop("jaxforge_sandbox", None)
            else:
                sys.modules["jaxforge_sandbox"] = prev_mod


def _encode_remote_payload(
    code: str,
    intent: str,
    fn_name: str,
    call_args: dict[str, Any],
) -> str:
    allowed = sorted(INTENT_ALLOWED_IMPORTS.get(intent, WHITELIST_IMPORTS))
    try:
        raw = pickle.dumps(
            {
                "code": code,
                "intent": intent,
                "fn_name": fn_name,
                "call_args": call_args,
                "allowed_imports": allowed,
                "banned_prefixes": sorted(BANNED_MODULE_PREFIXES),
                "banned_attrs": sorted(BANNED_ATTRIBUTES | BANNED_NUMPY_ATTRIBUTES),
            }
        )
    except Exception as e:
        raise JaxForgeExecutionError(
            f"Runtime arguments cannot be serialized for isolated backend: {e}",
            code=code,
        ) from e
    return base64.b64encode(raw).decode("ascii")


def _build_self_contained_runner(payload_b64: str, out_file: str | None = None) -> str:
    """Build a self-contained Python script requiring no preinstalled jaxforge package."""
    sink_stmt = (
        f"open({out_file!r}, 'wb').write(_out_bytes)\n"
        if out_file
        else f"print({_RESULT_MARKER!r} + base64.b64encode(_out_bytes).decode('ascii'))\n"
    )
    return (
        "import base64, builtins, pickle, sys, types\n"
        f"_payload = pickle.loads(base64.b64decode({payload_b64!r}))\n"
        "_allowed = set(_payload['allowed_imports'])\n"
        "_banned_prefixes = tuple(_payload['banned_prefixes'])\n"
        "_banned_attrs = set(_payload['banned_attrs'])\n"
        "def _safe_imp(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    if level != 0:\n"
        "        raise ImportError('Relative imports are not allowed.')\n"
        "    root = name.split('.')[0]\n"
        "    _bad = any(name == p or name.startswith(p + '.') for p in _banned_prefixes)\n"
        "    if root not in _allowed or _bad:\n"
        "        raise ImportError(f'Import {name!r} is not allowed.')\n"
        "    for sym in (fromlist or ()):\n"
        "        if sym == '*' or sym in _banned_attrs:\n"
        "            raise ImportError(f'Import {sym!r} is not allowed.')\n"
        "    return builtins.__import__(name, globals, locals, fromlist, level)\n"
        "_safe_builtins = {\n"
        "    k: getattr(builtins, k) for k in (\n"
        "        'print','range','len','enumerate','zip','map','filter','sorted','reversed',\n"
        "        'min','max','sum','abs','round','int','float','str','bool','list','dict',\n"
        "        'tuple','set','isinstance','hasattr','repr','True','False','None',\n"
        "        '__build_class__','Exception','ValueError','TypeError','RuntimeError',\n"
        "        'KeyError','PermissionError'\n"
        "    ) if hasattr(builtins, k)\n"
        "}\n"
        "_safe_builtins['__import__'] = _safe_imp\n"
        "_mod = types.ModuleType('jaxforge_sandbox')\n"
        "_mod.__dict__['__builtins__'] = _safe_builtins\n"
        "for _alias, _pkg in (('jax','jax'),('jnp','jax.numpy'),('flax','flax'),"
        "('nnx','flax.nnx'),('optax','optax'),('np','numpy')):\n"
        "    try:\n"
        "        _m = __import__(_pkg, fromlist=['*'])\n"
        "        _mod.__dict__[_alias] = _m\n"
        "    except ImportError:\n"
        "        pass\n"
        "sys.modules['jaxforge_sandbox'] = _mod\n"
        "exec(_payload['code'], _mod.__dict__)\n"
        "_res = _mod.__dict__[_payload['fn_name']](**_payload['call_args'])\n"
        "def _pack(v):\n"
        "    try:\n"
        "        pickle.dumps(v)\n"
        "        return v\n"
        "    except Exception:\n"
        "        if 'nnx' in _mod.__dict__ and isinstance(v, _mod.__dict__['nnx'].Module):\n"
        "            _jax = _mod.__dict__['jax']\n"
        "            _np = _mod.__dict__['np']\n"
        "            _nnx = _mod.__dict__['nnx']\n"
        "            _lvs = [_np.asarray(x) for x in _jax.tree_util.tree_leaves(_nnx.state(v))]\n"
        "            return ('__jaxforge_nnx_leaves__', _lvs)\n"
        "        if isinstance(v, tuple):\n"
        "            return tuple(_pack(i) for i in v)\n"
        "        raise\n"
        "_out_bytes = pickle.dumps(_pack(_res))\n"
        + sink_stmt
    )


def _decode_stdout_result(stdout_text: str, code: str, intent: str) -> Any:
    for line in reversed(stdout_text.splitlines()):
        if line.startswith(_RESULT_MARKER):
            b64_part = line[len(_RESULT_MARKER) :].strip()
            return _safe_pickle_loads(base64.b64decode(b64_part), code=code, intent=intent)
    raise JaxForgeExecutionError(
        "Isolated sandbox completed without emitting a valid result payload.",
        code=code,
    )


class _SandboxTimeoutError(BaseException):
    """Unrecoverable timeout raised inside the sandbox so except Exception cannot swallow it."""


def _run_with_timeout(fn: Callable[[], Any], timeout_seconds: float) -> Any:
    """Execute ``fn`` in a daemon worker thread enforced by trace deadlines and join timeout."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    result_box: list[Any] = []
    error_box: list[BaseException] = []
    deadline = time.monotonic() + timeout_seconds

    def _trace_deadline(frame: FrameType, event: str, arg: Any) -> Any:
        if time.monotonic() > deadline:
            raise _SandboxTimeoutError(f"Execution timed out after {timeout_seconds}s")
        return _trace_deadline

    def _target() -> None:
        sys.settrace(_trace_deadline)
        try:
            result_box.append(fn())
        except BaseException as exc:
            error_box.append(exc)
        finally:
            sys.settrace(None)

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=timeout_seconds + 0.5)
    if worker.is_alive():
        raise _SandboxTimeoutError(f"Execution timed out after {timeout_seconds}s")
    if error_box:
        raise error_box[0]
    return result_box[0]


class ExecutionBackend(ABC):
    """Interface for sandboxed JAX code execution environments."""

    @abstractmethod
    def execute(
        self,
        code: str,
        intent: str,
        fn_name: str,
        call_args: dict[str, Any],
        allowed_dirs: tuple[Path | str, ...] | None,
        timeout_seconds: float,
        max_memory_mb: int | None = None,
    ) -> Any:
        """Validate and run ``fn_name(**call_args)`` inside this backend."""
        ...


class ThreadSandboxBackend(ExecutionBackend):
    """Low-latency isolated namespace executed in a daemon thread with trace deadline."""

    def execute(
        self,
        code: str,
        intent: str,
        fn_name: str,
        call_args: dict[str, Any],
        allowed_dirs: tuple[Path | str, ...] | None,
        timeout_seconds: float,
        max_memory_mb: int | None = None,
    ) -> Any:
        namespace = get_execution_namespace(intent=intent, allowed_dirs=allowed_dirs)
        try:
            _run_with_timeout(lambda: exec(code, namespace), timeout_seconds)
        except (_SandboxTimeoutError, TimeoutError, Exception) as e:
            tb = traceback.format_exc()
            raise JaxForgeExecutionError(
                f"Code failed during definition phase: {e}\n\n{tb}",
                code=code,
                attempts=1,
            ) from e

        if fn_name not in namespace:
            raise JaxForgeExecutionError(
                f"Function '{fn_name}' not found in namespace after exec. "
                "This is a validator bug; please report it.",
                code=code,
            )

        try:
            return _run_with_timeout(lambda: namespace[fn_name](**call_args), timeout_seconds)
        except (_SandboxTimeoutError, TimeoutError, Exception) as e:
            tb = traceback.format_exc()
            raise JaxForgeExecutionError(
                f"Function '{fn_name}' raised an error during execution:\n"
                f"{type(e).__name__}: {e}\n\nTraceback:\n{tb}",
                code=code,
                attempts=1,
            ) from e


class SubprocessSandboxBackend(ExecutionBackend):
    """OS-process isolation enforcing CPU/memory rlimits and hard subprocess timeout."""

    def execute(
        self,
        code: str,
        intent: str,
        fn_name: str,
        call_args: dict[str, Any],
        allowed_dirs: tuple[Path | str, ...] | None,
        timeout_seconds: float,
        max_memory_mb: int | None = None,
    ) -> Any:
        CodeValidator.validate(code, intent)
        resolved_dirs = [str(Path(d).resolve()) for d in (allowed_dirs or (Path.cwd(),))]
        runner = (
            "import pickle, resource, sys, types\n"
            "from pathlib import Path\n"
            "from jaxforge.sandbox.namespace import get_execution_namespace\n"
            "payload = pickle.loads(Path(sys.argv[1]).read_bytes())\n"
            "mem_mb = payload.get('max_memory_mb')\n"
            "if mem_mb:\n"
            "    limit = int(mem_mb) * 1024 * 1024\n"
            "    applied = False\n"
            "    for rlim_name in ('RLIMIT_DATA', 'RLIMIT_RSS', 'RLIMIT_AS'):\n"
            "        rlim = getattr(resource, rlim_name, None)\n"
            "        if rlim is None:\n"
            "            continue\n"
            "        try:\n"
            "            _, hard = resource.getrlimit(rlim)\n"
            "            eff = limit if hard == resource.RLIM_INFINITY else min(limit, hard)\n"
            "            resource.setrlimit(rlim, (eff, hard))\n"
            "            applied = True\n"
            "            break\n"
            "        except (ValueError, OSError):\n"
            "            continue\n"
            "    if not applied:\n"
            "        sys.stderr.write('JAXFORGE_RLIMIT_UNSUPPORTED\\n')\n"
            "ns = get_execution_namespace(intent=payload['intent'], "
            "allowed_dirs=tuple(payload['allowed_dirs']))\n"
            "mod = types.ModuleType('jaxforge_sandbox')\n"
            "mod.__dict__.update(ns)\n"
            "sys.modules['jaxforge_sandbox'] = mod\n"
            "exec(payload['code'], mod.__dict__)\n"
            "res = mod.__dict__[payload['fn_name']](**payload['call_args'])\n"
            "def _pack(v):\n"
            "    try:\n"
            "        pickle.dumps(v)\n"
            "        return v\n"
            "    except Exception:\n"
            "        import flax.nnx as _nnx, jax as _jax, numpy as _np\n"
            "        if isinstance(v, _nnx.Module):\n"
            "            _st = _nnx.state(v)\n"
            "            leaves = [_np.asarray(x) for x in _jax.tree_util.tree_leaves(_st)]\n"
            "            return ('__jaxforge_nnx_leaves__', leaves)\n"
            "        if isinstance(v, tuple):\n"
            "            return tuple(_pack(i) for i in v)\n"
            "        raise\n"
            "Path(sys.argv[2]).write_bytes(pickle.dumps(_pack(res)))\n"
        )
        with tempfile.TemporaryDirectory(prefix="jaxforge_subproc_") as tmp:
            in_path = Path(tmp) / "in.pkl"
            out_path = Path(tmp) / "out.pkl"
            try:
                in_path.write_bytes(
                    pickle.dumps(
                        {
                            "code": code,
                            "intent": intent,
                            "fn_name": fn_name,
                            "call_args": call_args,
                            "allowed_dirs": resolved_dirs,
                            "max_memory_mb": max_memory_mb,
                        }
                    )
                )
            except Exception as e:
                raise JaxForgeExecutionError(
                    f"Runtime arguments cannot be serialized for subprocess backend: {e}",
                    code=code,
                ) from e

            try:
                proc = subprocess.run(  # noqa: S603
                    [sys.executable, "-c", runner, str(in_path), str(out_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env={
                        "PATH": os.environ.get("PATH", ""),
                        "HOME": tmp,
                        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    },
                )
            except subprocess.TimeoutExpired as e:
                raise JaxForgeExecutionError(
                    f"Subprocess execution timed out after {timeout_seconds}s",
                    code=code,
                    attempts=1,
                ) from e

            if "JAXFORGE_RLIMIT_UNSUPPORTED" in proc.stderr:
                warnings.warn(
                    "OS memory rlimit could not be enforced on this platform.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if proc.returncode != 0:
                raise JaxForgeExecutionError(
                    f"Subprocess execution failed (exit {proc.returncode}):\n{proc.stderr}",
                    code=code,
                    attempts=1,
                )
            if not out_path.exists():
                raise JaxForgeExecutionError(
                    "Subprocess exited cleanly without writing output payload.",
                    code=code,
                    attempts=1,
                )
            return _safe_pickle_loads(out_path.read_bytes(), code=code, intent=intent)


class DockerSandboxBackend(ExecutionBackend):
    """Isolated container backend running validated code in a hardened Python container."""

    def __init__(self, image: str = "jaxforge-sandbox:latest"):
        self._image = image

    def build_command(
        self,
        work_dir: Path,
        timeout_seconds: float,
        max_memory_mb: int | None,
        allowed_dirs: tuple[Path | str, ...] | None = None,
    ) -> list[str]:
        mem_flag = f"{max_memory_mb or 2048}m"
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",  # noqa: S108
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--memory={mem_flag}",
            f"--stop-timeout={max(1, int(timeout_seconds))}",
            "-e",
            "HOME=/tmp",
            "-e",
            "XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "-v",
            f"{work_dir}:/workspace:rw",
        ]
        for extra_dir in allowed_dirs or ():
            resolved = Path(extra_dir).resolve()
            if resolved.exists():
                cmd.extend(["-v", f"{resolved}:{resolved}:rw"])
        cmd.extend(
            [
                "-w",
                "/workspace",
                self._image,
                "python",
                "/workspace/run_sandbox.py",
            ]
        )
        return cmd

    def execute(
        self,
        code: str,
        intent: str,
        fn_name: str,
        call_args: dict[str, Any],
        allowed_dirs: tuple[Path | str, ...] | None,
        timeout_seconds: float,
        max_memory_mb: int | None = None,
    ) -> Any:
        CodeValidator.validate(code, intent)
        payload_b64 = _encode_remote_payload(code, intent, fn_name, call_args)
        with tempfile.TemporaryDirectory(prefix="jaxforge_docker_") as tmp:
            work_dir = Path(tmp)
            out_file = "/workspace/out.pkl"
            script = _build_self_contained_runner(payload_b64, out_file=out_file)
            (work_dir / "run_sandbox.py").write_text(script, encoding="utf-8")
            cmd = self.build_command(
                work_dir, timeout_seconds, max_memory_mb, allowed_dirs=allowed_dirs
            )
            try:
                proc = subprocess.run(  # noqa: S603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds + 10,
                )
            except FileNotFoundError as e:
                raise JaxForgeConfigError(
                    "Docker backend selected, but 'docker' executable was not found on PATH."
                ) from e
            except subprocess.TimeoutExpired as e:
                raise JaxForgeExecutionError(
                    f"Docker execution timed out after {timeout_seconds}s",
                    code=code,
                    attempts=1,
                ) from e
            if proc.returncode != 0:
                raise JaxForgeExecutionError(
                    f"Docker execution failed:\n{proc.stderr}",
                    code=code,
                    attempts=1,
                )
            host_out = work_dir / "out.pkl"
            if not host_out.exists():
                raise JaxForgeExecutionError(
                    "Docker container exited without producing output payload.",
                    code=code,
                    attempts=1,
                )
            return _safe_pickle_loads(host_out.read_bytes(), code=code, intent=intent)


class E2BSandboxBackend(ExecutionBackend):
    """Firecracker microVM backend delegating execution to e2b_code_interpreter."""

    def execute(
        self,
        code: str,
        intent: str,
        fn_name: str,
        call_args: dict[str, Any],
        allowed_dirs: tuple[Path | str, ...] | None,
        timeout_seconds: float,
        max_memory_mb: int | None = None,
    ) -> Any:
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError as e:
            raise JaxForgeConfigError(
                "E2B backend selected, but e2b-code-interpreter is not installed. "
                "Install with: pip install 'jaxforge[e2b]'."
            ) from e
        CodeValidator.validate(code, intent)
        payload_b64 = _encode_remote_payload(code, intent, fn_name, call_args)
        runner_script = _build_self_contained_runner(payload_b64, out_file=None)
        with Sandbox(timeout=int(timeout_seconds)) as sandbox:
            execution = sandbox.run_code(runner_script)
            if getattr(execution, "error", None):
                raise JaxForgeExecutionError(
                    f"E2B sandbox error: {execution.error.name}: {execution.error.value}",
                    code=code,
                    attempts=1,
                )
            logs = getattr(execution, "logs", None)
            stdout_chunks = getattr(logs, "stdout", None) if logs is not None else None
            stdout_text = (
                "\n".join(stdout_chunks)
                if isinstance(stdout_chunks, list)
                else str(getattr(execution, "text", "") or "")
            )
            return _decode_stdout_result(stdout_text, code=code, intent=intent)


class ModalSandboxBackend(ExecutionBackend):
    """Serverless remote GPU/TPU sandbox backend delegating execution to Modal."""

    _cached_image: Any = None

    @classmethod
    def _get_image(cls, modal_mod: Any) -> Any:
        if cls._cached_image is None:
            cls._cached_image = modal_mod.Image.debian_slim().pip_install(
                "jax", "flax", "optax", "numpy"
            )
        return cls._cached_image

    def execute(
        self,
        code: str,
        intent: str,
        fn_name: str,
        call_args: dict[str, Any],
        allowed_dirs: tuple[Path | str, ...] | None,
        timeout_seconds: float,
        max_memory_mb: int | None = None,
    ) -> Any:
        try:
            import modal
        except ImportError as e:
            raise JaxForgeConfigError(
                "Modal backend selected, but modal is not installed. "
                "Install with: pip install 'jaxforge[modal]'."
            ) from e
        CodeValidator.validate(code, intent)
        payload_b64 = _encode_remote_payload(code, intent, fn_name, call_args)
        runner_script = _build_self_contained_runner(payload_b64, out_file=None)
        app = modal.App.lookup("jaxforge-sandbox", create_if_missing=True)
        image = self._get_image(modal)
        sb = modal.Sandbox.create(
            "python",
            "-c",
            runner_script,
            app=app,
            image=image,
            timeout=int(timeout_seconds),
            block_network=True,
        )
        sb.wait()
        if sb.returncode != 0:
            raise JaxForgeExecutionError(
                f"Modal sandbox error:\n{sb.stderr.read()}",
                code=code,
                attempts=1,
            )
        return _decode_stdout_result(sb.stdout.read(), code=code, intent=intent)


def create_backend(
    name: str = "thread",
    docker_image: str = "jaxforge-sandbox:latest",
) -> ExecutionBackend:
    """Instantiate an ExecutionBackend by configured name."""
    if name == "docker":
        return DockerSandboxBackend(image=docker_image)
    mapping: dict[str, Callable[[], ExecutionBackend]] = {
        "thread": ThreadSandboxBackend,
        "subprocess": SubprocessSandboxBackend,
        "e2b": E2BSandboxBackend,
        "modal": ModalSandboxBackend,
    }
    factory = mapping.get(name)
    if factory is None:
        all_names = sorted([*mapping.keys(), "docker"])
        raise JaxForgeConfigError(
            f"Unknown execution_backend '{name}'. Expected one of: {all_names}."
        )
    return factory()
