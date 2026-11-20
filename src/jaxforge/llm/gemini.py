"""Google Gemini LLM adapter supporting google-genai, legacy SDK mocks, and REST."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from jaxforge.config import JaxForgeConfig
from jaxforge.exceptions import JaxForgeConfigError, JaxForgeLLMError
from jaxforge.llm.base import AbstractLLM, LLMUsage

_GENAI_LOCK = threading.Lock()

# Kept as an optional module attribute so tests can inject a mock client cleanly without
# eagerly importing the deprecated google.generativeai package at startup.
genai: Any = None


def _extract_response_text(response: Any) -> str:
    """Read response.text safely once so blocked or empty candidates fail cleanly."""
    try:
        text = response.text
    except Exception as e:
        raise JaxForgeLLMError(f"Gemini returned no readable text payload: {e}") from e
    if not isinstance(text, str) or not text.strip():
        raise JaxForgeLLMError("Gemini returned an empty text payload.")
    return text


def _messages_to_gemini_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        gemini_role = "model" if role in ("model", "assistant") else "user"
        parts = [{"text": str(p)} for p in msg.get("parts", [])]
        contents.append({"role": gemini_role, "parts": parts})
    return contents


class GeminiLLM(AbstractLLM):
    """Google Gemini backend using mock/installed google-genai or zero-dep HTTPS REST."""

    def __init__(self, config: JaxForgeConfig):
        self._config = config
        self.last_usage: LLMUsage | None = None
        self._api_key = self._resolve_api_key()
        if genai is not None and hasattr(genai, "configure"):
            with _GENAI_LOCK:
                genai.configure(api_key=self._api_key)

    def _resolve_api_key(self) -> str:
        api_key = self._config.get_effective_api_key()
        if not api_key:
            raise JaxForgeConfigError(
                "No Gemini API key found. Set api_key/gemini_api_key in config "
                "or export GEMINI_API_KEY as an environment variable."
            )
        return api_key

    def _generate_via_rest(
        self,
        api_key: str,
        model_id: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
    ) -> tuple[str, int, int]:
        base = (
            self._config.base_url.rstrip("/")
            if self._config.base_url
            else "https://generativelanguage.googleapis.com/v1beta"
        )
        url = f"{base}/models/{model_id}:generateContent?key={api_key}"
        payload = {
            "contents": _messages_to_gemini_contents(messages),
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema,
                "temperature": self._config.temperature,
                "maxOutputTokens": self._config.max_output_tokens,
            },
        }
        req = urllib.request.Request(  # noqa: S310
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._config.timeout_seconds) as resp:  # noqa: S310
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            raise JaxForgeLLMError(f"Gemini HTTP {e.code}: {detail}") from e
        except Exception as e:
            raise JaxForgeLLMError(f"Gemini REST error: {e}") from e

        candidates = body.get("candidates") or []
        if not candidates:
            raise JaxForgeLLMError("Gemini returned no candidates.")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(str(p.get("text", "")) for p in parts).strip()
        if not text:
            raise JaxForgeLLMError("Gemini returned an empty text payload.")
        meta = body.get("usageMetadata") or {}
        return (
            text,
            int(meta.get("promptTokenCount", 0)),
            int(meta.get("candidatesTokenCount", 0)),
        )

    def _generate_via_google_genai(
        self,
        api_key: str,
        model_id: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
    ) -> str | None:
        try:
            from google import genai as new_genai
            from google.genai import types as genai_types
        except ImportError:
            return None

        client = new_genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_id,
            contents=_messages_to_gemini_contents(messages),
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_output_tokens,
            ),
        )
        return _extract_response_text(response)

    def generate(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        raw_text = ""
        model_id = self._config.resolve_model_id()
        start = time.monotonic()
        prompt_tok = 0
        comp_tok = 0
        try:
            api_key = self._resolve_api_key()
            if genai is not None:
                with _GENAI_LOCK:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(
                        model_name=model_id,
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            response_schema=schema,
                            temperature=self._config.temperature,
                            max_output_tokens=self._config.max_output_tokens,
                        ),
                    )
                    response = model.generate_content(
                        messages,
                        request_options={"timeout": self._config.timeout_seconds},
                    )
                raw_text = _extract_response_text(response)
            else:
                sdk_text = self._generate_via_google_genai(api_key, model_id, messages, schema)
                if sdk_text is not None:
                    raw_text = sdk_text
                else:
                    raw_text, prompt_tok, comp_tok = self._generate_via_rest(
                        api_key, model_id, messages, schema
                    )

            parsed = json.loads(raw_text)
            if not isinstance(parsed, dict):
                raise JaxForgeLLMError(
                    "Gemini returned a non-object JSON payload.",
                    raw_response=raw_text,
                )
            self.last_usage = LLMUsage(
                provider="google",
                model=model_id,
                prompt_tokens=prompt_tok,
                completion_tokens=comp_tok,
                total_tokens=prompt_tok + comp_tok,
                latency_ms=(time.monotonic() - start) * 1000.0,
            )
            return parsed
        except JaxForgeLLMError:
            raise
        except json.JSONDecodeError as e:
            raise JaxForgeLLMError(
                f"Gemini returned invalid JSON: {e}",
                raw_response=raw_text,
            ) from e
        except Exception as e:
            raise JaxForgeLLMError(f"Gemini API error: {e}") from e

    def is_available(self) -> bool:
        try:
            api_key = self._resolve_api_key()
            model_id = self._config.resolve_model_id()
            if genai is not None:
                with _GENAI_LOCK:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(model_id)
                    model.generate_content(
                        "ping",
                        request_options={"timeout": self._config.timeout_seconds},
                    )
                return True
            self._generate_via_rest(
                api_key,
                model_id,
                [{"role": "user", "parts": ["ping"]}],
                {"type": "object", "properties": {"ok": {"type": "boolean"}}},
            )
            return True
        except Exception:
            return False
