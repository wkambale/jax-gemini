"""Model Context Protocol (MCP) tool definitions and stdio server for IDE agents."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import flax.nnx as nnx

import jaxforge
from jaxforge.codegen.validator import CodeValidator
from jaxforge.contrib.sharding import device_summary
from jaxforge.exceptions import JaxForgeError
from jaxforge.sandbox.executor import SandboxExecutor

MAX_MCP_TIMEOUT_SECONDS = 30.0
MAX_MCP_RESULT_CHARS = 4096
ALLOWED_MCP_EXEC_INTENTS = frozenset(
    {"build", "analyze_data", "preprocess_data", "load_data", "evaluate"}
)


def tool_validate_jax_code(code: str, intent: str = "build") -> dict[str, Any]:
    """Validate JAX Python source using CodeValidator without executing it."""
    try:
        CodeValidator.validate(code, intent)
        return {"valid": True, "intent": intent, "message": "Validation passed."}
    except JaxForgeError as exc:
        return {"valid": False, "intent": intent, "error": str(exc)}


def tool_device_summary() -> dict[str, Any]:
    """Return available JAX accelerators and default platform metadata."""
    return device_summary()


def tool_run_sandboxed_jax(
    code: str,
    intent: str = "analyze_data",
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    """Execute validated JAX code inside an isolated temporary directory and return status."""
    if not jaxforge.config.auto_approve:
        return {
            "ok": False,
            "error": "Sandbox execution via MCP is disabled while config.auto_approve is False.",
        }
    if intent not in ALLOWED_MCP_EXEC_INTENTS:
        return {
            "ok": False,
            "error": (
                f"Intent '{intent}' is not permitted via MCP tool_run_sandboxed_jax. "
                f"Allowed intents: {sorted(ALLOWED_MCP_EXEC_INTENTS)}."
            ),
        }
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or timeout_seconds > MAX_MCP_TIMEOUT_SECONDS
    ):
        return {
            "ok": False,
            "error": (
                f"timeout_seconds must be a finite number between 0 and "
                f"{MAX_MCP_TIMEOUT_SECONDS}."
            ),
        }

    default_args: dict[str, Any] = {}
    if intent == "build":
        default_args = {"rngs": nnx.Rngs(0)}
    elif intent in ("analyze_data", "preprocess_data", "evaluate"):
        default_args = {"dataset": None}

    try:
        with tempfile.TemporaryDirectory(prefix="jaxforge_mcp_") as tmp_dir:
            result = SandboxExecutor().run(
                code,
                intent=intent,
                args=default_args,
                allowed_dirs=(Path(tmp_dir),),
                timeout_seconds=float(timeout_seconds),
                backend_name=jaxforge.config.execution_backend,
                max_memory_mb=jaxforge.config.max_memory_mb,
                docker_image=jaxforge.config.docker_image,
            )
        rendered = repr(result)
        if len(rendered) > MAX_MCP_RESULT_CHARS:
            rendered = rendered[:MAX_MCP_RESULT_CHARS] + "... [truncated]"
        return {"ok": True, "result_repr": rendered}
    except (JaxForgeError, ValueError, TypeError) as exc:
        return {"ok": False, "error": str(exc)}


def serve_stdio() -> None:
    """Launch an MCP stdio server if the optional 'mcp' package is installed."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The optional 'mcp' package is required to start the MCP server. "
            "Install with: pip install 'jaxforge[mcp]'."
        ) from exc

    server = FastMCP("jaxforge")
    server.tool()(tool_validate_jax_code)
    server.tool()(tool_device_summary)
    server.tool()(tool_run_sandboxed_jax)
    server.run()
