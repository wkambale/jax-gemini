"""Backward-compatibility shim for jax_gemini.sandbox.namespace."""

from jaxforge.sandbox.namespace import get_execution_namespace

__all__ = ["get_execution_namespace"]
