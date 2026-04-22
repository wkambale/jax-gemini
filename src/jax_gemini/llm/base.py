from abc import ABC, abstractmethod
from typing import Any

class AbstractLLM(ABC):

    @abstractmethod
    def generate(self, messages: list[dict[str, Any]], schema: dict) -> dict:
        """
        Send messages to the LLM and return a parsed JSON response.

        Args:
            messages: List of {"role": "user"|"model", "parts": [str]} dicts
            schema: JSON schema the response must conform to

        Returns:
            Parsed dict matching the provided schema

        Raises:
            JaxGeminiLLMError: If the API call fails or response is malformed
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM is reachable and the API key is valid."""
        ...
