"""Backward-compatibility shim for jax_gemini.prompts."""

from jaxforge.prompts import (
    MAX_CONTEXT_PROMPT_CHARS,
    MAX_PROMPT_CHARS,
    ContextBuilder,
    IntentClassifier,
)

__all__ = [
    "MAX_CONTEXT_PROMPT_CHARS",
    "MAX_PROMPT_CHARS",
    "ContextBuilder",
    "IntentClassifier",
]
