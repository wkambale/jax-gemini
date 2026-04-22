import pytest
import os
from jax_gemini.config import JaxGeminiConfig
from jax_gemini.exceptions import JaxGeminiConfigError

class TestConfig:

    def test_default_config(self):
        config = JaxGeminiConfig()
        assert config.model_name in ["gemini-1.5-pro", "gemini-2.5-pro"]
        assert config.temperature == 0.2

    def test_set_updates_values(self):
        config = JaxGeminiConfig()
        config.set({"temperature": 0.5})
        assert config.temperature == 0.5

    def test_set_unknown_key_raises(self):
        config = JaxGeminiConfig()
        with pytest.raises(JaxGeminiConfigError, match="Unknown config key"):
            config.set({"unknown_key": "val"})

    def test_validate_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        config = JaxGeminiConfig()
        with pytest.raises(JaxGeminiConfigError, match="gemini_api_key"):
            config.validate()

    def test_validate_with_api_key(self):
        config = JaxGeminiConfig(gemini_api_key="key")
        config.validate() # should not raise
