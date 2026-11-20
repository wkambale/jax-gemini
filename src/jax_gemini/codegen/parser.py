from typing import Any

from jax_gemini.exceptions import JaxGeminiLLMError


class CodeParser:
    @staticmethod
    def extract(response: dict[str, Any], intent: str) -> str:
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
        if not isinstance(response, dict):
            raise JaxGeminiLLMError(
                "Gemini response must be a JSON object.",
                raw_response=str(response),
            )
        if "code" not in response:
            raise JaxGeminiLLMError(
                f"Gemini response missing 'code' field. Got fields: {list(response.keys())}",
                raw_response=str(response),
            )
        response_intent = response.get("intent")
        if isinstance(response_intent, str) and response_intent and response_intent != intent:
            raise JaxGeminiLLMError(
                f"Gemini response intent '{response_intent}' does not match '{intent}'.",
                raw_response=str(response),
            )
        raw_code = response["code"]
        if not isinstance(raw_code, str):
            raise JaxGeminiLLMError(
                "Gemini response 'code' field must be a string.",
                raw_response=str(response),
            )
        code = raw_code.strip()
        if not code:
            raise JaxGeminiLLMError("Gemini returned an empty code block.")
        return code
