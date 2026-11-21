import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from jaxforge.config import JaxForgeConfig
from jaxforge.exceptions import JaxForgeConfigError, JaxForgeLLMError
from jaxforge.llm.gemini import GeminiLLM
from jaxforge.llm.providers import (
    AnthropicLLM,
    LiteLLMBackend,
    OpenAICompatibleLLM,
    RouterLLM,
    create_llm,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._bytes = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._bytes

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class TestProvidersAndRouting:
    def test_config_provider_prefix_inference_and_validation(self, monkeypatch):
        cfg = JaxForgeConfig(model_name="anthropic:claude-sonnet-4-5")
        assert cfg.resolve_provider() == "anthropic"
        assert cfg.resolve_model_id() == "claude-sonnet-4-5"

        assert JaxForgeConfig(model_name="gpt-5").resolve_provider() == "openai"
        assert JaxForgeConfig(model_name="claude-3-7-sonnet").resolve_provider() == "anthropic"
        assert JaxForgeConfig(model_name="llama3.2").resolve_provider() == "ollama"
        assert JaxForgeConfig(model_name="gemini-3.1-pro").resolve_provider() == "google"

        with pytest.raises(JaxForgeConfigError, match="provider"):
            cfg.set({"provider": "unknown"})
        with pytest.raises(JaxForgeConfigError, match="base_url"):
            cfg.set({"base_url": "ftp://invalid"})
        with pytest.raises(JaxForgeConfigError, match="Plaintext http://"):
            cfg.set({"base_url": "http://api.remote-llm.example.com/v1"})
        cfg.set({"base_url": "https://api.remote-llm.example.com/v1"})
        cfg.set({"base_url": "http://127.0.0.1:11434/v1"})
        with pytest.raises(JaxForgeConfigError, match="execution_backend"):
            cfg.set({"execution_backend": "kubernetes"})
        with pytest.raises(JaxForgeConfigError, match="max_memory_mb"):
            cfg.set({"max_memory_mb": 16})

        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        oai_cfg = JaxForgeConfig(model_name="gpt-5")
        oai_cfg.validate()
        assert oai_cfg.get_effective_api_key() == "sk-openai-test"

        # Ollama works without API key
        ollama_cfg = JaxForgeConfig(model_name="ollama:qwen2.5-coder:7b")
        ollama_cfg.validate()

    def test_openai_compatible_and_ollama_rest(self):
        cfg = JaxForgeConfig(model_name="ollama:llama3", base_url="http://localhost:11434/v1")
        llm = create_llm(cfg)
        assert isinstance(llm, OpenAICompatibleLLM)

        fake_body = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '```json\n{"intent":"build","code":"x=1","description":"ok"}\n```'
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        }
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(fake_body)):
            out = llm.generate([{"role": "user", "parts": ["Build MLP"]}], {"type": "object"})
            assert out["intent"] == "build"
            assert llm.last_usage is not None
            assert llm.last_usage.total_tokens == 20
            assert llm.is_available() is True

        with pytest.raises(JaxForgeConfigError, match="No API key"):
            create_llm(JaxForgeConfig(model_name="gpt-5")).generate([], {})

    def test_anthropic_rest_adapter(self):
        cfg = JaxForgeConfig(model_name="claude-sonnet-4-5", api_key="ant-key")
        llm = create_llm(cfg)
        assert isinstance(llm, AnthropicLLM)

        fake_body = {
            "content": [
                {
                    "type": "text",
                    "text": '{"intent":"build","code":"y=2","description":"ok"}',
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(fake_body)):
            out = llm.generate([{"role": "user", "parts": ["Build CNN"]}], {})
            assert out["code"] == "y=2"
            assert llm.last_usage is not None
            assert llm.last_usage.provider == "anthropic"
            assert llm.last_usage.total_tokens == 15

    def test_gemini_rest_fallback_without_legacy_sdk(self, monkeypatch):
        cfg = JaxForgeConfig(gemini_api_key="gem-key", model_name="gemini-3.1-pro")
        llm = GeminiLLM(cfg)
        fake_body = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    '{"intent":"build","code":"def build_model(rngs): return 1",'
                                    '"description":"ok"}'
                                )
                            }
                        ]
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 9},
        }
        monkeypatch.setattr(llm, "_generate_via_google_genai", lambda *_a, **_kw: None)
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(fake_body)):
            out = llm.generate([{"role": "user", "parts": ["build"]}], {})
            assert out["intent"] == "build"
            assert llm.last_usage is not None
            assert llm.last_usage.total_tokens == 16

    def test_litellm_and_router_fast_model(self):
        cfg = JaxForgeConfig(provider="litellm", model_name="openai/gpt-5", api_key="k")
        backend = LiteLLMBackend(cfg)
        fake_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"intent":"build","code":"pass","description":"ok"}'
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6),
        )
        fake_litellm = MagicMock(completion=MagicMock(return_value=fake_resp))
        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            parsed = backend.generate(
                [{"role": "user", "parts": ["hi"]}],
                {"type": "object", "required": ["intent"]},
            )
            assert parsed["intent"] == "build"
            sent_msgs = fake_litellm.completion.call_args.kwargs["messages"]
            assert sent_msgs[0]["role"] == "system"
            assert "JSON schema" in sent_msgs[0]["content"]

        primary = MagicMock()
        primary.generate.return_value = {"p": 1}
        primary.last_usage = None
        fast = MagicMock()
        fast.generate.return_value = {"f": 1}
        fast.last_usage = None
        router = RouterLLM(primary=primary, fast=fast)
        assert router.generate([], {}, use_fast=False) == {"p": 1}
        assert router.generate([], {}, use_fast=True) == {"f": 1}

    def test_cross_provider_key_isolation_headers_alternation_and_redirect_block(
        self, monkeypatch
    ):
        import urllib.error
        import urllib.request

        from jaxforge.llm.gemini import _NoRedirectHandler
        from jaxforge.memory.conversation import ConversationMemory
        from jaxforge.prompts.context_builder import ContextBuilder

        # 1. Cross-provider fast_model_name must NOT leak primary api_key or OPENAI_API_KEY
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-do-not-leak")
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        cfg = JaxForgeConfig(
            api_key="sk-ant-primary-secret",
            model_name="anthropic:claude-sonnet-4-5",
            fast_model_name="ollama:qwen2.5-coder:7b",
            base_url="https://proxy.anthropic.example.com/v1",
        )
        fast_llm = create_llm(cfg, model_override=cfg.fast_model_name)
        assert isinstance(fast_llm, OpenAICompatibleLLM)
        assert fast_llm._headers() == {}
        assert fast_llm._endpoint() == "http://localhost:11434/v1/chat/completions"

        # 2. Anthropic uses top-level system and strictly alternating user/assistant messages
        ctx = ContextBuilder.build("build", "Build MLP", ConversationMemory(), None)
        ant = AnthropicLLM(cfg)
        captured_payloads: list[dict] = []

        ok_json = '{"intent":"build","code":"1","description":"d"}'

        def _fake_urlopen(req, timeout=60):
            captured_payloads.append(
                {
                    "url": req.full_url,
                    "headers": dict(req.header_items()),
                    "body": json.loads(req.data.decode("utf-8")),
                }
            )
            return _FakeHTTPResponse(
                {
                    "content": [{"type": "text", "text": ok_json}],
                    "usage": {"input_tokens": 5, "output_tokens": 5},
                }
            )

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            ant.generate(ctx, {"type": "object", "required": ["intent"]})
        ant_body = captured_payloads[0]["body"]
        assert "system" in ant_body and "matching schema" in ant_body["system"]
        ant_roles = [m["role"] for m in ant_body["messages"]]
        assert ant_roles == ["user"]

        # 3. Gemini REST sends x-goog-api-key header (never ?key= query string)
        gem_cfg = JaxForgeConfig(gemini_api_key="gem-secret-header", model_name="gemini-3.1-pro")
        gem = GeminiLLM(gem_cfg)
        monkeypatch.setattr(gem, "_generate_via_google_genai", lambda *_a, **_kw: None)
        captured_payloads.clear()
        with patch(
            "urllib.request.urlopen",
            side_effect=lambda req, timeout=60: (
                captured_payloads.append(
                    {
                        "url": req.full_url,
                        "headers": {k.lower(): v for k, v in req.header_items()},
                        "body": json.loads(req.data.decode("utf-8")),
                    }
                )
                or _FakeHTTPResponse(
                    {
                        "candidates": [{"content": {"parts": [{"text": ok_json}]}}],
                        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4},
                    }
                )
            ),
        ):
            gem.generate(ctx, {"type": "object"})
        assert "key=" not in captured_payloads[0]["url"]
        assert captured_payloads[0]["headers"]["x-goog-api-key"] == "gem-secret-header"
        assert "systemInstruction" in captured_payloads[0]["body"]

        # 4. Redirect handler blocks HTTP 302
        handler = _NoRedirectHandler()
        req = urllib.request.Request("https://api.example.com/v1/messages")
        with pytest.raises(urllib.error.HTTPError, match="redirect"):
            handler.redirect_request(req, None, 302, "Found", {}, "http://evil.example.com")

    def test_unsupported_provider_raises(self):
        bad = JaxForgeConfig()
        bad.provider = "unsupported"
        with pytest.raises(JaxForgeConfigError, match="provider"):
            bad.validate()
        with pytest.raises(JaxForgeLLMError):
            OpenAICompatibleLLM(
                JaxForgeConfig(model_name="ollama:m"), provider="ollama"
            )._headers() or None
            with patch(
                "urllib.request.urlopen",
                return_value=_FakeHTTPResponse({"choices": [{"message": {"content": "not json"}}]}),
            ):
                OpenAICompatibleLLM(
                    JaxForgeConfig(model_name="ollama:m"), provider="ollama"
                ).generate([{"role": "user", "parts": ["x"]}], {})
