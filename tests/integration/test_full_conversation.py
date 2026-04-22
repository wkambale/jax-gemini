from unittest.mock import patch, MagicMock
import pytest
import jax_gemini as jg

MOCK_MODIFY_RESPONSE = {
    "intent": "build",
    "code": (
        "import flax.nnx as nnx\n\n"
        "def build_model(rngs):\n"
        "    model = nnx.Sequential(\n"
        "        nnx.Linear(784, 512, rngs=rngs),\n"
        "        nnx.relu,\n"
        "        nnx.Linear(512, 10, rngs=rngs),\n"
        "    )\n"
        "    return model\n"
    ),
    "description": "Modified.",
    "warnings": [],
}

class TestFullConversation:

    @patch("jax_gemini.llm.gemini.genai")
    def test_full_conversation(self, mock_genai, config):
        import json
        mock_response = MagicMock()
        mock_response.text = json.dumps(MOCK_MODIFY_RESPONSE)
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response

        agent = jg.JaxGemini(config=config)
        model = agent.build("Build a model")
        model = agent.modify("Change hidden to 512")
        
        assert agent._memory.turn_count == 4
        agent.reset()
        assert agent._memory.turn_count == 0
