class JaxGeminiError(Exception):
    """Base exception for all jax-gemini errors."""
    pass


class JaxGeminiConfigError(JaxGeminiError):
    """Raised when configuration is invalid or missing."""
    pass


class JaxGeminiLLMError(JaxGeminiError):
    """Raised when the Gemini API call fails."""
    def __init__(self, message: str, raw_response: str = ""):
        super().__init__(message)
        self.raw_response = raw_response


class JaxGeminiValidationError(JaxGeminiError):
    """Raised when generated code fails AST validation."""
    def __init__(self, message: str, code: str = ""):
        super().__init__(message)
        self.code = code


class JaxGeminiExecutionError(JaxGeminiError):
    """Raised when generated code fails to execute after all retries."""
    def __init__(self, message: str, code: str = "", attempts: int = 0):
        super().__init__(message)
        self.code = code
        self.attempts = attempts


class JaxGeminiIntentError(JaxGeminiError):
    """Raised when the intent classifier cannot determine the operation."""
    pass


class JaxGeminiCheckpointError(JaxGeminiError):
    """Raised when model save or load fails."""
    pass
