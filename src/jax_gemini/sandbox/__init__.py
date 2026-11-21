"""Backward-compatibility shim for jax_gemini.sandbox."""

from jaxforge.sandbox import (
    DockerSandboxBackend,
    E2BSandboxBackend,
    ExecutionBackend,
    ModalSandboxBackend,
    SandboxExecutor,
    SubprocessSandboxBackend,
    ThreadSandboxBackend,
    create_backend,
    get_execution_namespace,
)

__all__ = [
    "DockerSandboxBackend",
    "E2BSandboxBackend",
    "ExecutionBackend",
    "ModalSandboxBackend",
    "SandboxExecutor",
    "SubprocessSandboxBackend",
    "ThreadSandboxBackend",
    "create_backend",
    "get_execution_namespace",
]
