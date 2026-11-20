import json
from unittest.mock import MagicMock, patch

import pytest

from jaxforge.config import JaxForgeConfig
from jaxforge.exceptions import JaxForgeConfigError, JaxForgeLLMError
from jaxforge.llm.gemini import GeminiLLM


class TestGeminiLLM:
    def test_missing_api_key_raises_config_error(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(JaxForgeConfigError, match="No Gemini API key found"):
            GeminiLLM(JaxForgeConfig())

    @patch("jaxforge.llm.gemini.genai")
    def test_generate_error_paths(self, mock_genai):
        llm = GeminiLLM(JaxForgeConfig(gemini_api_key="test-key"))

        # Empty text payload
        empty_resp = MagicMock()
        empty_resp.text = "   "
        mock_genai.GenerativeModel.return_value.generate_content.return_value = empty_resp
        with pytest.raises(JaxForgeLLMError, match="empty text payload"):
            llm.generate([], {})

        # Non-object JSON payload (e.g. list)
        list_resp = MagicMock()
        list_resp.text = json.dumps([1, 2, 3])
        mock_genai.GenerativeModel.return_value.generate_content.return_value = list_resp
        with pytest.raises(JaxForgeLLMError, match="non-object JSON"):
            llm.generate([], {})

        # Invalid JSON syntax
        bad_json_resp = MagicMock()
        bad_json_resp.text = "{broken json"
        mock_genai.GenerativeModel.return_value.generate_content.return_value = bad_json_resp
        with pytest.raises(JaxForgeLLMError, match="invalid JSON") as exc_info:
            llm.generate([], {})
        assert exc_info.value.raw_response == "{broken json"

        # SDK exception
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = RuntimeError(
            "network down"
        )
        with pytest.raises(JaxForgeLLMError, match="Gemini API error"):
            llm.generate([], {})

    @patch("jaxforge.llm.gemini.genai")
    def test_is_available_true_and_false(self, mock_genai):
        llm = GeminiLLM(JaxForgeConfig(gemini_api_key="test-key"))
        mock_genai.GenerativeModel.return_value.generate_content.return_value = MagicMock()
        assert llm.is_available() is True

        mock_genai.GenerativeModel.return_value.generate_content.side_effect = RuntimeError("fail")
        assert llm.is_available() is False
