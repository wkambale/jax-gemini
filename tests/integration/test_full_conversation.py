from unittest.mock import MagicMock, patch

import jaxforge as jf

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
    @patch("jaxforge.llm.gemini.genai")
    def test_full_conversation(self, mock_genai, config):
        import json

        mock_response = MagicMock()
        mock_response.text = json.dumps(MOCK_MODIFY_RESPONSE)
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response

        agent = jf.JaxForge(config=config)
        agent.build("Build a model")
        agent.modify("Change hidden to 512")

        assert agent._memory.turn_count == 4
        agent.reset()
        assert agent._memory.turn_count == 0
