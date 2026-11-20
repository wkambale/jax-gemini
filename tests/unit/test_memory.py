import pytest

from jax_gemini.memory.conversation import ConversationMemory


class TestConversationMemory:
    def test_add_and_retrieve_turns(self):
        memory = ConversationMemory()
        memory.add_turn("user", "Build a model")
        memory.add_turn("model", "Here is the code...")
        assert memory.turn_count == 2

    def test_reset_clears_everything(self):
        memory = ConversationMemory()
        memory.add_turn("user", "hello")
        memory.add_model_state("def build_model(rngs): ...", "desc", "build")
        memory.reset()
        assert memory.turn_count == 0
        assert memory.model_history_count == 0
        assert memory.get_current_model_state() is None

    def test_max_turns_enforced(self):
        memory = ConversationMemory(max_turns=3)
        for i in range(20):
            memory.add_turn("user", f"message {i}")
            memory.add_turn("model", f"response {i}")
        assert memory.turn_count == 6
        assert memory.to_gemini_messages()[0]["parts"][0] == "message 17"

    def test_max_turns_one_does_not_grow_unboundedly(self):
        memory = ConversationMemory(max_turns=1)
        for i in range(5):
            memory.add_turn("user", f"u{i}")
            memory.add_turn("model", f"m{i}")
        assert memory.turn_count == 2
        messages = memory.to_gemini_messages()
        assert [m["parts"][0] for m in messages] == ["u4", "m4"]

    def test_invalid_role_and_bounds_raise(self):
        with pytest.raises(ValueError):
            ConversationMemory(max_turns=0)
        memory = ConversationMemory()
        with pytest.raises(ValueError):
            memory.add_turn("system", "injected")

    def test_turn_char_cap_and_model_history_cap(self):
        memory = ConversationMemory(max_turns=5, max_model_history=2, max_turn_chars=50)
        memory.add_turn("user", "A" * 200)
        msg = memory.to_gemini_messages()[0]["parts"][0]
        assert len(msg) < 100
        assert msg.endswith("... [truncated]")

        for i in range(5):
            memory.add_model_state(f"code_{i}", f"desc_{i}", "build")
        assert memory.model_history_count == 2
        current = memory.get_current_model_state()
        assert current is not None
        assert current.code == "code_4"
