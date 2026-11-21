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
_STDLIB_URLOPEN = urllib.request.urlopen


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Block automatic HTTP redirects so API credentials are never forwarded cross-origin."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        raise urllib.error.HTTPError(
            req.full_url,
            code,
            f"HTTP redirect ({code}) blocked to protect API credentials.",
            headers,
            fp,
        )


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler)

# Kept as an optional module attribute so tests can inject a mock client cleanly without
# eagerly importing the deprecated google.generativeai package at startup.
genai: Any = None


def _open_url_safely(req: urllib.request.Request, timeout: float) -> Any:
    if urllib.request.urlopen is not _STDLIB_URLOPEN:
        return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310
    return _NO_REDIRECT_OPENER.open(req, timeout=timeout)


def _extract_response_text(response: Any) -> str:
    """Read response.text safely once so blocked or empty candidates fail cleanly."""
    try:
        text = response.text
    except Exception as e:
        raise JaxForgeLLMError(f"Gemini returned no readable text payload: {e}") from e
    if not isinstance(text, str) or not text.strip():
        raise JaxForgeLLMError("Gemini returned an empty text payload.")
    return text


def _extract_sdk_token_counts(response: Any) -> tuple[int, int]:
    """Extract integer token counts from SDK usage_metadata when present."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return 0, 0
    p_tok = getattr(meta, "prompt_token_count", 0)
    c_tok = getattr(meta, "candidates_token_count", 0)
    prompt_tokens = p_tok if type(p_tok) is int else 0
    comp_tokens = c_tok if type(c_tok) is int else 0
    return prompt_tokens, comp_tokens


def _split_system_and_gemini_contents(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    sys_chunks: list[str] = []
    contents: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("system"):
            sys_chunks.extend(str(p) for p in msg.get("parts", []))
            continue
        if msg.get("system_ack"):
            continue
        role = msg.get("role", "user")
        gemini_role = "model" if role in ("model", "assistant") else "user"
        parts = [{"text": str(p)} for p in msg.get("parts", [])]
        contents.append({"role": gemini_role, "parts": parts})
    sys_text = "\n\n".join(c for c in sys_chunks if c).strip() or None
    if not contents:
        contents = [{"role": "user", "parts": [{"text": sys_text or ""}]}]
        sys_text = None
    return sys_text, contents


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
        url = f"{base}/models/{model_id}:generateContent"
        sys_text, contents = _split_system_and_gemini_contents(messages)
        gen_cfg: dict[str, Any] = {
            "responseMimeType": "application/json",
            "temperature": self._config.temperature,
            "maxOutputTokens": self._config.max_output_tokens,
        }
        if schema:
            gen_cfg["responseSchema"] = schema
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": gen_cfg,
        }
        if sys_text:
            payload["systemInstruction"] = {"parts": [{"text": sys_text}]}
        req = urllib.request.Request(  # noqa: S310
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        try:
            with _open_url_safely(req, timeout=self._config.timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300] if e.fp else str(e)
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
    ) -> tuple[str, int, int] | None:
        try:
            from google import genai as new_genai
            from google.genai import types as genai_types
        except ImportError:
            return None

        sys_text, contents = _split_system_and_gemini_contents(messages)
        cfg_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "temperature": self._config.temperature,
            "max_output_tokens": self._config.max_output_tokens,
        }
        if schema:
            cfg_kwargs["response_schema"] = schema
        if sys_text:
            cfg_kwargs["system_instruction"] = sys_text
        client = new_genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=genai_types.GenerateContentConfig(**cfg_kwargs),
        )
        text = _extract_response_text(response)
        p_tok, c_tok = _extract_sdk_token_counts(response)
        return text, p_tok, c_tok

    def generate(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        raw_text = ""
        model_id = self._config.resolve_model_id()
        start = time.monotonic()
        prompt_tok = 0
        comp_tok = 0
        try:
            api_key = self._resolve_api_key()
            if genai is not None:
                clean_msgs = [
                    {"role": m["role"], "parts": m["parts"]} for m in messages
                ]
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
                        clean_msgs,
                        request_options={"timeout": self._config.timeout_seconds},
                    )
                raw_text = _extract_response_text(response)
                prompt_tok, comp_tok = _extract_sdk_token_counts(response)
            else:
                sdk_out = self._generate_via_google_genai(api_key, model_id, messages, schema)
                if sdk_out is not None:
                    raw_text, prompt_tok, comp_tok = sdk_out
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
