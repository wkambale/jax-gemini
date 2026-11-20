import pytest

from jax_gemini.config import JaxGeminiConfig
from jax_gemini.exceptions import JaxGeminiConfigError


class TestConfig:
    def test_default_config(self):
        config = JaxGeminiConfig()
        assert config.model_name == "gemini-3.1-pro"
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
        config.validate()  # should not raise

    def test_validate_invalid_checkpoint_dir_raises(self):
        config = JaxGeminiConfig(gemini_api_key="key", checkpoint_dir="   ")
        with pytest.raises(JaxGeminiConfigError, match="checkpoint_dir"):
            config.validate()

    def test_set_rejects_method_shadowing_and_dunders(self):
        config = JaxGeminiConfig(gemini_api_key="key")
        for key in ["validate", "set", "__dict__", "__class__"]:
            with pytest.raises(JaxGeminiConfigError, match="Unknown config key"):
                config.set({key: lambda: None})

    def test_set_rejects_nan_bool_and_out_of_range_and_rolls_back(self):
        config = JaxGeminiConfig(gemini_api_key="key", temperature=0.3)
        with pytest.raises(JaxGeminiConfigError, match="temperature"):
            config.set({"max_retries": 4, "temperature": float("nan")})
        assert config.max_retries == 3  # rolled back atomically

        with pytest.raises(JaxGeminiConfigError, match="max_retries"):
            config.set({"max_retries": True})
        with pytest.raises(JaxGeminiConfigError, match="max_output_tokens"):
            config.set({"max_output_tokens": 0})
        with pytest.raises(JaxGeminiConfigError, match="timeout_seconds"):
            config.set({"timeout_seconds": -5})
        with pytest.raises(JaxGeminiConfigError, match="gemini_api_key"):
            config.set({"gemini_api_key": "   "})

    def test_repr_hides_api_key(self):
        config = JaxGeminiConfig(gemini_api_key="super-secret-api-key")
        assert "super-secret-api-key" not in repr(config)
