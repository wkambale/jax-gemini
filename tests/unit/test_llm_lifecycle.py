import json
from unittest.mock import MagicMock, patch

import pytest

import jax_gemini as jg
from jax_gemini.exceptions import JaxGeminiConfigError


class TestLLMCredentialLifecycle:
    @patch("jax_gemini.llm.gemini.genai")
    def test_multi_instance_api_keys_do_not_bleed(self, mock_genai):
        resp = MagicMock()
        resp.text = json.dumps(
            {
                "intent": "build",
                "code": "import flax.nnx as nnx\ndef build_model(rngs):\n    return 1\n",
                "description": "ok",
                "warnings": [],
            }
        )
        mock_genai.GenerativeModel.return_value.generate_content.return_value = resp

        agent_a = jg.JaxGemini(jg.JaxGeminiConfig(gemini_api_key="key-a"))
        agent_b = jg.JaxGemini(jg.JaxGeminiConfig(gemini_api_key="key-b"))

        agent_a.build("m1")
        agent_b.build("m2")
        mock_genai.configure.reset_mock()
        agent_a.build("m3")
        mock_genai.configure.assert_called_with(api_key="key-a")

    @patch("jax_gemini.llm.gemini.genai")
    def test_key_rotation_and_reset_invalidate_cached_llm(self, mock_genai):
        resp = MagicMock()
        resp.text = json.dumps(
            {
                "intent": "build",
                "code": "import flax.nnx as nnx\ndef build_model(rngs):\n    return 1\n",
                "description": "ok",
                "warnings": [],
            }
        )
        mock_genai.GenerativeModel.return_value.generate_content.return_value = resp

        agent = jg.JaxGemini(jg.JaxGeminiConfig(gemini_api_key="key-1"))
        agent.build("m1")
        first_llm = agent._llm
        assert first_llm is not None

        agent.config.set({"gemini_api_key": "key-2"})
        agent.build("m2")
        assert agent._llm is not first_llm
        mock_genai.configure.assert_called_with(api_key="key-2")

        agent.reset()
        assert agent._llm is None

    @patch("jax_gemini.llm.gemini.genai")
    def test_post_init_invalid_config_is_revalidated(self, mock_genai):
        resp = MagicMock()
        resp.text = json.dumps(
            {
                "intent": "build",
                "code": "import flax.nnx as nnx\ndef build_model(rngs):\n    return 1\n",
                "description": "ok",
                "warnings": [],
            }
        )
        mock_genai.GenerativeModel.return_value.generate_content.return_value = resp

        agent = jg.JaxGemini(jg.JaxGeminiConfig(gemini_api_key="key-1"))
        agent.build("m1")
        with pytest.raises(JaxGeminiConfigError, match="max_retries"):
            agent.config.set({"max_retries": 99})
        agent.config.max_retries = 99
        with pytest.raises(JaxGeminiConfigError, match="max_retries"):
            agent.build("m2")

    @patch("jax_gemini.llm.gemini.genai")
    def test_generate_passes_timeout_request_option(self, mock_genai):
        resp = MagicMock()
        resp.text = json.dumps(
            {
                "intent": "build",
                "code": "import flax.nnx as nnx\ndef build_model(rngs):\n    return 1\n",
                "description": "ok",
                "warnings": [],
            }
        )
        mock_genai.GenerativeModel.return_value.generate_content.return_value = resp

        agent = jg.JaxGemini(jg.JaxGeminiConfig(gemini_api_key="key-1", timeout_seconds=15))
        agent.build("m1")
        call_kwargs = mock_genai.GenerativeModel.return_value.generate_content.call_args.kwargs
        assert call_kwargs["request_options"] == {"timeout": 15}

    @patch("jax_gemini.llm.gemini.genai")
    def test_env_api_key_not_pinned_and_blocked_response_handled(self, mock_genai, monkeypatch):
        from jax_gemini.exceptions import JaxGeminiLLMError

        monkeypatch.setenv("GEMINI_API_KEY", "env-key-1")
        resp1 = MagicMock()
        resp1.text = json.dumps(
            {
                "intent": "build",
                "code": "import flax.nnx as nnx\ndef build_model(rngs):\n    return 1\n",
                "description": "ok",
                "warnings": [],
            }
        )
        blocked_resp = MagicMock()
        type(blocked_resp).text = property(
            lambda _self: (_ for _ in ()).throw(ValueError("safety blocked"))
        )
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = [
            resp1,
            blocked_resp,
        ]

        agent = jg.JaxGemini(jg.JaxGeminiConfig(max_retries=1))
        assert agent.build("m1") == 1
        assert agent.config.gemini_api_key is None

        monkeypatch.setenv("GEMINI_API_KEY", "env-key-2")
        with pytest.raises(JaxGeminiLLMError, match="no readable text payload"):
            agent.build("m2")
        mock_genai.configure.assert_called_with(api_key="env-key-2")
