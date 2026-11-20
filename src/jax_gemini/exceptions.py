"""Backward-compatibility shim for jax_gemini.exceptions."""

from jaxforge.exceptions import (
    JaxForgeApprovalError,
    JaxGeminiCheckpointError,
    JaxGeminiConfigError,
    JaxGeminiError,
    JaxGeminiExecutionError,
    JaxGeminiIntentError,
    JaxGeminiLLMError,
    JaxGeminiValidationError,
)

__all__ = [
    "JaxForgeApprovalError",
    "JaxGeminiCheckpointError",
    "JaxGeminiConfigError",
    "JaxGeminiError",
    "JaxGeminiExecutionError",
    "JaxGeminiIntentError",
    "JaxGeminiLLMError",
    "JaxGeminiValidationError",
]
