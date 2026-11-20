"""Multi-provider LLM adapters (OpenAI, OpenRouter, Ollama/vLLM, Anthropic, LiteLLM)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from copy import deepcopy
from typing import Any, ClassVar

from jaxforge.config import JaxForgeConfig
from jaxforge.exceptions import JaxForgeConfigError, JaxForgeLLMError
from jaxforge.llm.base import AbstractLLM, LLMUsage
from jaxforge.llm.gemini import GeminiLLM


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _normalize_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "user")
        oai_role = "assistant" if role in ("model", "assistant") else "user"
        content = "\n".join(str(p) for p in msg.get("parts", []))
        out.append({"role": oai_role, "content": content})
    return out


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int,
    provider_label: str,
) -> dict[str, Any]:
    if not url.startswith(("http://", "https://")):
        raise JaxForgeConfigError(f"Unsupported URL scheme for {provider_label}: {url!r}")
    req = urllib.request.Request(  # noqa: S310
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
            if not isinstance(data, dict):
                raise JaxForgeLLMError(f"{provider_label} returned non-object response.")
            return data
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        raise JaxForgeLLMError(f"{provider_label} HTTP {e.code}: {detail}") from e
    except JaxForgeLLMError:
        raise
    except Exception as e:
        raise JaxForgeLLMError(f"{provider_label} request failed: {e}") from e


class OpenAICompatibleLLM(AbstractLLM):
    """Adapter for OpenAI, OpenRouter, Ollama, vLLM, and OpenAI-compatible endpoints."""

    DEFAULT_BASE_URLS: ClassVar[dict[str, str]] = {
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "ollama": "http://localhost:11434/v1",
    }

    def __init__(self, config: JaxForgeConfig, provider: str = "openai"):
        self._config = config
        self._provider = provider
        self.last_usage: LLMUsage | None = None

    def _endpoint(self) -> str:
        base = self._config.base_url or self.DEFAULT_BASE_URLS[self._provider]
        return f"{base.rstrip('/')}/chat/completions"

    def _headers(self) -> dict[str, str]:
        key = self._config.get_effective_api_key()
        if not key and self._provider != "ollama":
            raise JaxForgeConfigError(
                f"No API key found for provider '{self._provider}'. "
                "Set api_key in config or export the provider environment variable."
            )
        headers: dict[str, str] = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def generate(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        model_id = self._config.resolve_model_id()
        start = time.monotonic()
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": _normalize_openai_messages(messages),
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if schema:
            payload["messages"].append(
                {
                    "role": "user",
                    "content": (
                        "Respond ONLY with a valid JSON object adhering to this JSON schema: "
                        f"{json.dumps(schema)}"
                    ),
                }
            )
        body = _post_json(
            self._endpoint(),
            payload,
            self._headers(),
            self._config.timeout_seconds,
            self._provider,
        )
        choices = body.get("choices") or []
        if not choices:
            raise JaxForgeLLMError(f"{self._provider} returned no choices.")
        raw_content = str((choices[0].get("message") or {}).get("content") or "")
        cleaned = _strip_json_fences(raw_content)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise JaxForgeLLMError(
                f"{self._provider} returned invalid JSON: {e}",
                raw_response=raw_content,
            ) from e
        if not isinstance(parsed, dict):
            raise JaxForgeLLMError(
                f"{self._provider} returned a non-object JSON payload.",
                raw_response=raw_content,
            )
        usage = body.get("usage") or {}
        prompt_tok = int(usage.get("prompt_tokens", 0))
        comp_tok = int(usage.get("completion_tokens", 0))
        self.last_usage = LLMUsage(
            provider=self._provider,
            model=model_id,
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            total_tokens=int(usage.get("total_tokens", prompt_tok + comp_tok)),
            latency_ms=(time.monotonic() - start) * 1000.0,
        )
        return parsed

    def is_available(self) -> bool:
        try:
            self.generate(
                [{"role": "user", "parts": ['Return {"ok": true}']}],
                {"type": "object", "properties": {"ok": {"type": "boolean"}}},
            )
            return True
        except Exception:
            return False


class AnthropicLLM(AbstractLLM):
    """Adapter for Anthropic Messages API."""

    def __init__(self, config: JaxForgeConfig):
        self._config = config
        self.last_usage: LLMUsage | None = None

    def _headers(self) -> dict[str, str]:
        key = self._config.get_effective_api_key()
        if not key:
            raise JaxForgeConfigError(
                "No Anthropic API key found. Set api_key in config or export ANTHROPIC_API_KEY."
            )
        return {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }

    def generate(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        model_id = self._config.resolve_model_id()
        start = time.monotonic()
        base = (self._config.base_url or "https://api.anthropic.com/v1").rstrip("/")
        norm = _normalize_openai_messages(messages)
        if schema:
            norm.append(
                {
                    "role": "user",
                    "content": (
                        "Respond strictly with raw JSON (no markdown fences) matching schema: "
                        f"{json.dumps(schema)}"
                    ),
                }
            )
        payload = {
            "model": model_id,
            "max_tokens": self._config.max_output_tokens,
            "temperature": self._config.temperature,
            "messages": norm,
        }
        body = _post_json(
            f"{base}/messages",
            payload,
            self._headers(),
            self._config.timeout_seconds,
            "anthropic",
        )
        content_blocks = body.get("content") or []
        raw_text = "".join(
            str(b.get("text", "")) for b in content_blocks if isinstance(b, dict)
        )
        cleaned = _strip_json_fences(raw_text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise JaxForgeLLMError(
                f"anthropic returned invalid JSON: {e}",
                raw_response=raw_text,
            ) from e
        if not isinstance(parsed, dict):
            raise JaxForgeLLMError(
                "anthropic returned a non-object JSON payload.",
                raw_response=raw_text,
            )
        usage = body.get("usage") or {}
        prompt_tok = int(usage.get("input_tokens", 0))
        comp_tok = int(usage.get("output_tokens", 0))
        self.last_usage = LLMUsage(
            provider="anthropic",
            model=model_id,
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            total_tokens=prompt_tok + comp_tok,
            latency_ms=(time.monotonic() - start) * 1000.0,
        )
        return parsed

    def is_available(self) -> bool:
        try:
            self.generate(
                [{"role": "user", "parts": ['Return {"ok": true}']}],
                {"type": "object", "properties": {"ok": {"type": "boolean"}}},
            )
            return True
        except Exception:
            return False


class LiteLLMBackend(AbstractLLM):
    """Optional adapter delegating to litellm.completion when litellm is installed."""

    def __init__(self, config: JaxForgeConfig):
        self._config = config
        self.last_usage: LLMUsage | None = None

    def generate(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        try:
            import litellm
        except ImportError as e:
            raise JaxForgeConfigError(
                "litellm provider selected, but litellm is not installed. "
                "Install with: pip install 'jaxforge[litellm]'."
            ) from e

        model_id = self._config.resolve_model_id()
        start = time.monotonic()
        resp = litellm.completion(
            model=model_id,
            messages=_normalize_openai_messages(messages),
            temperature=self._config.temperature,
            max_tokens=self._config.max_output_tokens,
            timeout=self._config.timeout_seconds,
            api_key=self._config.get_effective_api_key(),
            api_base=self._config.base_url,
            response_format={"type": "json_object"},
        )
        raw_text = str(resp.choices[0].message.content or "")
        cleaned = _strip_json_fences(raw_text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise JaxForgeLLMError(
                f"litellm returned invalid JSON: {e}",
                raw_response=raw_text,
            ) from e
        if not isinstance(parsed, dict):
            raise JaxForgeLLMError(
                "litellm returned a non-object JSON payload.",
                raw_response=raw_text,
            )
        usage = getattr(resp, "usage", None)
        prompt_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        comp_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        self.last_usage = LLMUsage(
            provider="litellm",
            model=model_id,
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            total_tokens=prompt_tok + comp_tok,
            latency_ms=(time.monotonic() - start) * 1000.0,
        )
        return parsed

    def is_available(self) -> bool:
        try:
            self.generate(
                [{"role": "user", "parts": ['Return {"ok": true}']}],
                {"type": "object", "properties": {"ok": {"type": "boolean"}}},
            )
            return True
        except Exception:
            return False


class RouterLLM(AbstractLLM):
    """Routes initial heavy planning to primary LLM and quick fixes to optional fast LLM."""

    def __init__(self, primary: AbstractLLM, fast: AbstractLLM | None = None):
        self._primary = primary
        self._fast = fast
        self.usage_history: list[LLMUsage] = []
        self.last_usage: LLMUsage | None = None

    def generate(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        use_fast: bool = False,
    ) -> dict[str, Any]:
        backend = self._fast if (use_fast and self._fast is not None) else self._primary
        result = backend.generate(messages, schema)
        if backend.last_usage is not None:
            self.last_usage = backend.last_usage
            self.usage_history.append(backend.last_usage)
        return result

    def is_available(self) -> bool:
        return self._primary.is_available()


def create_llm(config: JaxForgeConfig, model_override: str | None = None) -> AbstractLLM:
    """Instantiate the configured LLM backend from JaxForgeConfig."""
    cfg = deepcopy(config)
    if model_override:
        cfg.model_name = model_override
    prov = cfg.resolve_provider()
    if prov == "google":
        return GeminiLLM(cfg)
    if prov in ("openai", "ollama", "openrouter"):
        return OpenAICompatibleLLM(cfg, provider=prov)
    if prov == "anthropic":
        return AnthropicLLM(cfg)
    if prov == "litellm":
        return LiteLLMBackend(cfg)
    raise JaxForgeConfigError(f"Unsupported LLM provider: '{prov}'")
