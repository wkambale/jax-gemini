"""Backward-compatibility shim for jax_gemini.codegen.fixer."""

from jaxforge.codegen.fixer import CodeFixer, sanitize_error_for_llm

__all__ = ["CodeFixer", "sanitize_error_for_llm"]
