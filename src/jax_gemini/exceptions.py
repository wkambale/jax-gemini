"""Backward-compatibility shim for jax_gemini.exceptions."""

from jaxforge.exceptions import (
    JaxForgeApprovalError,
    JaxGeminiApprovalError,
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
    "JaxGeminiApprovalError",
    "JaxGeminiCheckpointError",
    "JaxGeminiConfigError",
    "JaxGeminiError",
    "JaxGeminiExecutionError",
    "JaxGeminiIntentError",
    "JaxGeminiLLMError",
    "JaxGeminiValidationError",
]
