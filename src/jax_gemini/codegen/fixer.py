from jax_gemini.exceptions import JaxGeminiExecutionError, JaxGeminiValidationError

class CodeFixer:

    @staticmethod
    def build_fix_prompt(
        original_prompt: str,
        failed_code: str,
        error_message: str,
        attempt: int,
    ) -> str:
        """
        Build a correction prompt to send back to Gemini when code fails.

        The error message must be specific enough that Gemini can fix it.
        """
        return (
            f"The following code was generated for the request: '{original_prompt}'\n\n"
            f"```python\n{failed_code}\n```\n\n"
            f"It failed with this error (attempt {attempt}):\n"
            f"```\n{error_message}\n```\n\n"
            f"Please fix the code. Return ONLY corrected code in the same JSON format. "
            f"Do not change the function name or signature. "
            f"Only import from: jax, flax, optax, orbax, numpy."
        )
