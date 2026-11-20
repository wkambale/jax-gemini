import json
import os
import threading
from typing import Any

import google.generativeai as genai

from jax_gemini.config import JaxGeminiConfig
from jax_gemini.exceptions import JaxGeminiConfigError, JaxGeminiLLMError
from jax_gemini.llm.base import AbstractLLM

_GENAI_LOCK = threading.Lock()


class GeminiLLM(AbstractLLM):
    def __init__(self, config: JaxGeminiConfig):
        self._config = config
        self._api_key = self._resolve_api_key()
        with _GENAI_LOCK:
            genai.configure(api_key=self._api_key)  # type: ignore[attr-defined]

    def _resolve_api_key(self) -> str:
        api_key = self._config.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise JaxGeminiConfigError(
                "No Gemini API key found. Set gemini_api_key in config "
                "or export GEMINI_API_KEY as an environment variable."
            )
        return api_key

    def generate(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        response = None
        try:
            api_key = self._resolve_api_key()
            with _GENAI_LOCK:
                genai.configure(api_key=api_key)  # type: ignore[attr-defined]
                model = genai.GenerativeModel(  # type: ignore[attr-defined]
                    model_name=self._config.model_name,
                    generation_config=genai.GenerationConfig(  # type: ignore[attr-defined]
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=self._config.temperature,
                        max_output_tokens=self._config.max_output_tokens,
                    ),
                )
                response = model.generate_content(messages)
            parsed = json.loads(response.text)
            if not isinstance(parsed, dict):
                raise JaxGeminiLLMError(
                    "Gemini returned a non-object JSON payload.",
                    raw_response=response.text if hasattr(response, "text") else "",
                )
            return parsed
        except JaxGeminiLLMError:
            raise
        except json.JSONDecodeError as e:
            raise JaxGeminiLLMError(
                f"Gemini returned invalid JSON: {e}",
                raw_response=response.text if (response and hasattr(response, "text")) else "",
            ) from e
        except Exception as e:
            raise JaxGeminiLLMError(f"Gemini API error: {e}") from e

    def is_available(self) -> bool:
        try:
            api_key = self._resolve_api_key()
            with _GENAI_LOCK:
                genai.configure(api_key=api_key)  # type: ignore[attr-defined]
                model = genai.GenerativeModel(self._config.model_name)  # type: ignore[attr-defined]
                model.generate_content("ping")
            return True
        except Exception:
            return False
