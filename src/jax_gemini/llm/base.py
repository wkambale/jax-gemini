"""Backward-compatibility shim for jax_gemini.llm.base."""

from jaxforge.llm.base import AbstractLLM, LLMUsage

__all__ = ["AbstractLLM", "LLMUsage"]
