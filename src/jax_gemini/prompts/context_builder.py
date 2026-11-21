"""Backward-compatibility shim for jax_gemini.prompts.context_builder."""

from jaxforge.prompts.context_builder import (
    MAX_CONTEXT_PROMPT_CHARS,
    MAX_PROMPT_CHARS,
    ContextBuilder,
)

__all__ = [
    "MAX_CONTEXT_PROMPT_CHARS",
    "MAX_PROMPT_CHARS",
    "ContextBuilder",
]
