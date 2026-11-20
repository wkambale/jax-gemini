"""Model Context Protocol (MCP) tool definitions and stdio server for IDE agents."""

from __future__ import annotations

from typing import Any

from jaxforge.codegen.validator import CodeValidator
from jaxforge.contrib.sharding import device_summary
from jaxforge.exceptions import JaxForgeError
from jaxforge.sandbox.executor import SandboxExecutor


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
    """Execute validated JAX code inside SandboxExecutor and return repr/status."""
    try:
        result = SandboxExecutor().run(
            code,
            intent=intent,
            args={"dataset": None} if intent in ("analyze_data", "preprocess_data") else {},
            timeout_seconds=timeout_seconds,
        )
        return {"ok": True, "result_repr": repr(result)}
    except JaxForgeError as exc:
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
