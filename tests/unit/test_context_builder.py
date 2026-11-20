import pytest

from jax_gemini import JaxGemini, JaxGeminiConfig
from jax_gemini.exceptions import JaxGeminiValidationError
from jax_gemini.memory.conversation import ConversationMemory, ModelState
from jax_gemini.prompts.context_builder import MAX_PROMPT_CHARS, ContextBuilder


class TestContextBuilder:
    def test_includes_current_model_state_with_alternating_roles(self):
        memory = ConversationMemory()
        memory.add_turn("user", "Build MLP")
        memory.add_turn("model", "Built MLP")
        state = ModelState(
            code="def build_model(rngs): return nnx.Linear(4, 2, rngs=rngs)",
            description="2-feature linear model",
            intent="build",
        )

        messages = ContextBuilder.build("train", "Train on dataset", memory, state)
        roles = [m["role"] for m in messages]
        # Every consecutive pair must strictly alternate user/model ending with user
        assert roles[0] == "user"
        assert roles[-1] == "user"
        for i in range(len(roles) - 1):
            assert roles[i] != roles[i + 1]

        joined = "\n".join(str(m["parts"][0]) for m in messages)
        assert "2-feature linear model" in joined
        assert "def build_model(rngs)" in joined

    def test_oversized_or_empty_prompt_rejected(self):
        memory = ConversationMemory()
        with pytest.raises(JaxGeminiValidationError):
            ContextBuilder.build("build", "x" * (MAX_PROMPT_CHARS + 1), memory, None)
        with pytest.raises(JaxGeminiValidationError):
            ContextBuilder.build("build", "   ", memory, None)

    def test_agent_rejects_oversized_prompt_before_llm_init(self):
        # No API key needed: must fail fast on prompt validation before _get_llm()
        agent = JaxGemini(config=JaxGeminiConfig())
        with pytest.raises(JaxGeminiValidationError, match="exceeds maximum allowed"):
            agent.build("a" * (MAX_PROMPT_CHARS + 10))
