import pytest
from unittest.mock import MagicMock
from jax_gemini.config import JaxGeminiConfig


@pytest.fixture
def config() -> JaxGeminiConfig:
    return JaxGeminiConfig(
        gemini_api_key="test-key-not-real",
        verbose=False,
        max_retries=2,
    )


@pytest.fixture
def mock_llm():
    """A mock LLM that returns a valid build response."""
    llm = MagicMock()
    llm.generate.return_value = {
        "intent": "build",
        "code": (
            "import flax.nnx as nnx\n"
            "def build_model(rngs):\n"
            "    return nnx.Linear(784, 10, rngs=rngs)\n"
        ),
        "description": "A simple linear model mapping 784 inputs to 10 outputs.",
        "warnings": [],
    }
    return llm
