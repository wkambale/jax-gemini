from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationTurn:
    role: str          # "user" or "model"
    content: str       # The message text


@dataclass
class ModelState:
    code: str          # Python code that produced this model
    description: str   # Gemini's description of the model
    intent: str        # What operation produced this state


class ConversationMemory:

    def __init__(self, max_turns: int = 20):
        self._turns: list[ConversationTurn] = []
        self._model_history: list[ModelState] = []
        self._max_turns = max_turns

    def add_turn(self, role: str, content: str) -> None:
        self._turns.append(ConversationTurn(role=role, content=content))
        if len(self._turns) > self._max_turns * 2:
            # Keep first 2 turns (system context) + latest turns
            self._turns = self._turns[:2] + self._turns[-(self._max_turns * 2 - 2):]

    def add_model_state(self, code: str, description: str, intent: str) -> None:
        self._model_history.append(ModelState(code=code, description=description, intent=intent))

    def get_current_model_state(self) -> ModelState | None:
        return self._model_history[-1] if self._model_history else None

    def to_gemini_messages(self) -> list[dict]:
        """Convert to the format expected by google-generativeai."""
        return [
            {"role": turn.role, "parts": [turn.content]}
            for turn in self._turns
        ]

    def reset(self) -> None:
        self._turns.clear()
        self._model_history.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)
