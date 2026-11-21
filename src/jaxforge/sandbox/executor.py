"""Validated sandbox executor delegating to a configured ExecutionBackend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jaxforge.codegen.validator import CodeValidator
from jaxforge.exceptions import JaxForgeExecutionError
from jaxforge.sandbox.backends import (
    ExecutionBackend,
    ThreadSandboxBackend,
    _run_with_timeout,
    _SandboxTimeoutError,
    create_backend,
)

INTENT_FUNCTION_MAP: dict[str, str] = {
    "build": "build_model",
    "train": "train_model",
    "evaluate": "evaluate_model",
    "save": "save_model",
    "load": "load_model",
    "load_data": "load_data",
    "preprocess_data": "preprocess_data",
    "analyze_data": "analyze_data",
    "visualize_data": "visualize_data",
}

INTENT_DEFAULT_ARGS: dict[str, dict[str, Any]] = {
    "build": {},
    "train": {"dataset": None, "epochs": 5, "learning_rate": 1e-3},
    "evaluate": {"dataset": None},
    "save": {"path": "./checkpoint"},
    "load": {"path": "./checkpoint"},
}

__all__ = [
    "INTENT_DEFAULT_ARGS",
    "INTENT_FUNCTION_MAP",
    "SandboxExecutor",
    "_SandboxTimeoutError",
    "_run_with_timeout",
]


class SandboxExecutor:
    """Validates code via AST rules and delegates execution to a sandbox backend."""

    def __init__(self, backend: ExecutionBackend | None = None):
        self._backend = backend

    def run(
        self,
        code: str,
        intent: str,
        args: dict[str, Any] | None = None,
        allowed_dirs: tuple[Path | str, ...] | None = None,
        timeout_seconds: float = 60.0,
        backend_name: str = "thread",
        max_memory_mb: int | None = None,
        docker_image: str = "jaxforge-sandbox:latest",
    ) -> Any:
        """Validate ``code`` and run the intent handler function safely."""
        CodeValidator.validate(code, intent)
        fn_name = INTENT_FUNCTION_MAP.get(intent, "")
        if not fn_name:
            raise JaxForgeExecutionError(f"Unknown intent: {intent}")
        call_args = {**INTENT_DEFAULT_ARGS.get(intent, {}), **(args or {})}
        backend = self._backend or (
            ThreadSandboxBackend()
            if backend_name == "thread"
            else create_backend(backend_name, docker_image=docker_image)
        )
        return backend.execute(
            code=code,
            intent=intent,
            fn_name=fn_name,
            call_args=call_args,
            allowed_dirs=allowed_dirs,
            timeout_seconds=timeout_seconds,
            max_memory_mb=max_memory_mb,
        )
