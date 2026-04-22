from unittest.mock import patch, MagicMock
import pytest
import jax_gemini as jg

MOCK_BUILD_RESPONSE = {
    "intent": "build",
    "code": (
        "import flax.nnx as nnx\n\n"
        "def build_model(rngs):\n"
        "    model = nnx.Sequential(\n"
        "        nnx.Linear(784, 256, rngs=rngs),\n"
        "        nnx.relu,\n"
        "        nnx.Linear(256, 10, rngs=rngs),\n"
        "    )\n"
        "    return model\n"
    ),
    "description": "A 2-layer MLP for classification.",
    "warnings": [],
}

class TestBuildPipeline:

    @patch("jax_gemini.llm.gemini.genai")
    def test_build_returns_flax_module(self, mock_genai, config):
        import json
        mock_response = MagicMock()
        mock_response.text = json.dumps(MOCK_BUILD_RESPONSE)
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response

        agent = jg.JaxGemini(config=config)
        model = agent.build("Build a model")
        assert model is not None
        assert hasattr(model, "__call__")

    def test_build_without_api_key_raises_config_error(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        agent = jg.JaxGemini(jg.JaxGeminiConfig(gemini_api_key=None))
        with pytest.raises(jg.JaxGeminiConfigError):
            agent.build("Build a model")
