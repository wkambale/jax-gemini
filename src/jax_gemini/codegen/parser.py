import json
from jax_gemini.exceptions import JaxGeminiLLMError


class CodeParser:

    @staticmethod
    def extract(response: dict, intent: str) -> str:
        """
        Extract the 'code' field from a structured Gemini response.

        Args:
            response: Parsed JSON dict from GeminiLLM.generate()
            intent: One of 'build', 'train', 'evaluate', 'save', 'load'

        Returns:
            Python code string

        Raises:
            JaxGeminiLLMError: If 'code' field is missing or empty
        """
        if "code" not in response:
            raise JaxGeminiLLMError(
                "Gemini response missing 'code' field. "
                f"Got fields: {list(response.keys())}",
                raw_response=str(response)
            )
        code = response["code"].strip()
        if not code:
            raise JaxGeminiLLMError("Gemini returned an empty code block.")
        return code
