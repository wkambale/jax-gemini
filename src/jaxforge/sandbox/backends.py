"""Pluggable execution backends: in-process thread, OS subprocess (rlimit), Docker, E2B, Modal."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

from jaxforge.codegen.validator import CodeValidator
from jaxforge.exceptions import JaxForgeConfigError, JaxForgeExecutionError
from jaxforge.sandbox.namespace import get_execution_namespace


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
            "import json, pickle, resource, sys\n"
            "from pathlib import Path\n"
            "from jaxforge.sandbox.namespace import get_execution_namespace\n"
            "payload = pickle.loads(Path(sys.argv[1]).read_bytes())\n"
            "mem_mb = payload.get('max_memory_mb')\n"
            "if mem_mb:\n"
            "    limit = int(mem_mb) * 1024 * 1024\n"
            "    try:\n"
            "        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))\n"
            "    except (ValueError, OSError):\n"
            "        pass\n"
            "ns = get_execution_namespace(intent=payload['intent'], "
            "allowed_dirs=tuple(payload['allowed_dirs']))\n"
            "exec(payload['code'], ns)\n"
            "res = ns[payload['fn_name']](**payload['call_args'])\n"
            "Path(sys.argv[2]).write_bytes(pickle.dumps(res))\n"
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
                    },
                )
            except subprocess.TimeoutExpired as e:
                raise JaxForgeExecutionError(
                    f"Subprocess execution timed out after {timeout_seconds}s",
                    code=code,
                    attempts=1,
                ) from e

            if proc.returncode != 0:
                raise JaxForgeExecutionError(
                    f"Subprocess execution failed (exit {proc.returncode}):\n{proc.stderr}",
                    code=code,
                    attempts=1,
                )
            return pickle.loads(out_path.read_bytes())  # noqa: S301


class DockerSandboxBackend(ExecutionBackend):
    """Isolated container backend running validated code in a hardened Python container."""

    def __init__(self, image: str = "python:3.12-slim"):
        self._image = image

    def build_command(
        self,
        work_dir: Path,
        timeout_seconds: float,
        max_memory_mb: int | None,
    ) -> list[str]:
        mem_flag = f"{max_memory_mb or 2048}m"
        return [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--memory={mem_flag}",
            f"--stop-timeout={max(1, int(timeout_seconds))}",
            "-v",
            f"{work_dir}:/workspace:rw",
            "-w",
            "/workspace",
            self._image,
            "python",
            "/workspace/run_sandbox.py",
        ]

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
        with tempfile.TemporaryDirectory(prefix="jaxforge_docker_") as tmp:
            work_dir = Path(tmp)
            script = (
                "import json, pickle\n"
                "from jaxforge.sandbox.namespace import get_execution_namespace\n"
                "args = pickle.loads(open('/workspace/args.pkl', 'rb').read())\n"
                "ns = get_execution_namespace(intent="
                f"{intent!r}"
                ", allowed_dirs=('/workspace',))\n"
                f"code = {code!r}\n"
                "exec(code, ns)\n"
                f"res = ns[{fn_name!r}](**args)\n"
                "with open('/workspace/out.pkl', 'wb') as f:\n"
                "    f.write(pickle.dumps(res))\n"
            )
            (work_dir / "args.pkl").write_bytes(pickle.dumps(call_args))
            (work_dir / "run_sandbox.py").write_text(script, encoding="utf-8")
            cmd = self.build_command(work_dir, timeout_seconds, max_memory_mb)
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
            return pickle.loads((work_dir / "out.pkl").read_bytes())  # noqa: S301


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
        with Sandbox(timeout=int(timeout_seconds)) as sandbox:
            payload = (
                f"{code}\n"
                f"_res = {fn_name}(**{json.dumps(call_args, default=str)})\n"
                "print(repr(_res))\n"
            )
            execution = sandbox.run_code(payload)
            if execution.error:
                raise JaxForgeExecutionError(
                    f"E2B sandbox error: {execution.error.name}: {execution.error.value}",
                    code=code,
                    attempts=1,
                )
            return execution.text


class ModalSandboxBackend(ExecutionBackend):
    """Serverless remote GPU/TPU sandbox backend delegating execution to Modal."""

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
        app = modal.App.lookup("jaxforge-sandbox", create_if_missing=True)
        image = modal.Image.debian_slim().pip_install("jax", "flax", "optax", "numpy")
        sb = modal.Sandbox.create(
            "python",
            "-c",
            f"{code}\nprint(repr({fn_name}(**{json.dumps(call_args, default=str)})))",
            app=app,
            image=image,
            timeout=int(timeout_seconds),
        )
        sb.wait()
        if sb.returncode != 0:
            raise JaxForgeExecutionError(
                f"Modal sandbox error:\n{sb.stderr.read()}",
                code=code,
                attempts=1,
            )
        return sb.stdout.read().strip()


def create_backend(name: str = "thread") -> ExecutionBackend:
    """Instantiate an ExecutionBackend by configured name."""
    mapping: dict[str, Callable[[], ExecutionBackend]] = {
        "thread": ThreadSandboxBackend,
        "subprocess": SubprocessSandboxBackend,
        "docker": DockerSandboxBackend,
        "e2b": E2BSandboxBackend,
        "modal": ModalSandboxBackend,
    }
    factory = mapping.get(name)
    if factory is None:
        raise JaxForgeConfigError(
            f"Unknown execution_backend '{name}'. Expected one of: {sorted(mapping)}."
        )
    return factory()
