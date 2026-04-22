import json
import os
import google.generativeai as genai
from jax_gemini.llm.base import AbstractLLM
from jax_gemini.exceptions import JaxGeminiLLMError, JaxGeminiConfigError
from jax_gemini.config import JaxGeminiConfig


class GeminiLLM(AbstractLLM):

    def __init__(self, config: JaxGeminiConfig):
        api_key = config.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise JaxGeminiConfigError(
                "No Gemini API key found. Set gemini_api_key in config "
                "or export GEMINI_API_KEY as an environment variable."
            )
        genai.configure(api_key=api_key)
        self._config = config

    def generate(self, messages: list[dict], schema: dict) -> dict:
        response = None
        try:
            model = genai.GenerativeModel(
                model_name=self._config.model_name,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=self._config.temperature,
                    max_output_tokens=self._config.max_output_tokens,
                ),
            )
            response = model.generate_content(messages)
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            raise JaxGeminiLLMError(
                f"Gemini returned invalid JSON: {e}",
                raw_response=response.text if (response and hasattr(response, 'text')) else ""
            )
        except Exception as e:
            raise JaxGeminiLLMError(f"Gemini API error: {e}")

    def is_available(self) -> bool:
        try:
            model = genai.GenerativeModel(self._config.model_name)
            model.generate_content("ping")
            return True
        except Exception:
            return False
