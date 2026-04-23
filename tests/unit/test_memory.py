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
        memory.reset()
        assert memory.turn_count == 0
        assert memory.get_current_model_state() is None

    def test_max_turns_enforced(self):
        memory = ConversationMemory(max_turns=3)
        for i in range(20):
            memory.add_turn("user", f"message {i}")
            memory.add_turn("model", f"response {i}")
        # Should not exceed 2 * max_turns * 2 (with system context)
        # So wait, 3 * 2 = 6 turns ? The code says self._max_turns * 2.
        # Actually from conversation.py logic:
        # self._turns = self._turns[:2] + self._turns[-(self._max_turns * 2 - 2):]
        # max turns total = max_turns * 2
        assert memory.turn_count <= 6
