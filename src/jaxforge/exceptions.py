"""Exception hierarchy for jaxforge."""


class JaxForgeError(Exception):
    """Base exception for all jaxforge errors."""


class JaxForgeConfigError(JaxForgeError):
    """Raised when configuration is invalid or missing."""


class JaxForgeLLMError(JaxForgeError):
    """Raised when an LLM provider call fails or returns malformed output."""

    def __init__(self, message: str, raw_response: str = ""):
        super().__init__(message)
        self.raw_response = raw_response


class JaxForgeValidationError(JaxForgeError):
    """Raised when generated code fails AST validation or policy checks."""

    def __init__(self, message: str, code: str = ""):
        super().__init__(message)
        self.code = code


class JaxForgeExecutionError(JaxForgeError):
    """Raised when generated code fails to execute after configured retries."""

    def __init__(self, message: str, code: str = "", attempts: int = 0):
        super().__init__(message)
        self.code = code
        self.attempts = attempts


class JaxForgeIntentError(JaxForgeError):
    """Raised when the intent classifier cannot determine an operation."""


class JaxForgeCheckpointError(JaxForgeError):
    """Raised when checkpoint save or restore fails."""


class JaxForgeApprovalError(JaxForgeError):
    """Raised when a previewed plan is rejected before execution."""

    def __init__(self, message: str, code: str = ""):
        super().__init__(message)
        self.code = code


# Backward-compatible aliases for jax-gemini users
JaxGeminiError = JaxForgeError
JaxGeminiConfigError = JaxForgeConfigError
JaxGeminiLLMError = JaxForgeLLMError
JaxGeminiValidationError = JaxForgeValidationError
JaxGeminiExecutionError = JaxForgeExecutionError
JaxGeminiIntentError = JaxForgeIntentError
JaxGeminiCheckpointError = JaxForgeCheckpointError
JaxGeminiApprovalError = JaxForgeApprovalError
