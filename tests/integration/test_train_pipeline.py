from unittest.mock import MagicMock, patch

import numpy as np

import jax_gemini as jg

MOCK_TRAIN_RESPONSE = {
    "intent": "train",
    "code": (
        "import jax.numpy as jnp\n"
        "def train_model(model, dataset, epochs=1, learning_rate=1e-3):\n"
        "    return model, {'accuracy': 0.95, 'loss': 0.1}\n"
    ),
    "description": "Trained",
    "warnings": [],
}


class TestTrainPipeline:
    @patch("jax_gemini.llm.gemini.genai")
    def test_train_pipeline(self, mock_genai, config):
        import json

        mock_response = MagicMock()
        mock_response.text = json.dumps(MOCK_TRAIN_RESPONSE)
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response

        agent = jg.JaxGemini(config=config)
        agent._current_model = "mock_model"

        X = np.zeros((10, 10))
        y = np.zeros((10,))

        _model, metrics = agent.train("Train it", dataset=(X, y))
        assert metrics["accuracy"] == 0.95
        assert metrics["loss"] == 0.1
