"""Bounded conversation and model-state memory for Jax Forge sessions."""

from dataclasses import dataclass
from typing import Any

VALID_ROLES = frozenset({"user", "model"})
DEFAULT_MAX_TURN_CHARS = 16_000
DEFAULT_MAX_MODEL_HISTORY = 20


def _bound_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class ModelState:
    code: str
    description: str
    intent: str


class ConversationMemory:
    """Bounded conversation and model-state memory for a JaxForge session."""

    def __init__(
        self,
        max_turns: int = 20,
        max_model_history: int = DEFAULT_MAX_MODEL_HISTORY,
        max_turn_chars: int = DEFAULT_MAX_TURN_CHARS,
    ):
        for name, value in (
            ("max_turns", max_turns),
            ("max_model_history", max_model_history),
            ("max_turn_chars", max_turn_chars),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer >= 1, got {value!r}.")
        self._turns: list[ConversationTurn] = []
        self._model_history: list[ModelState] = []
        self._max_turns = max_turns
        self._max_model_history = max_model_history
        self._max_turn_chars = max_turn_chars

    def add_turn(self, role: str, content: str) -> None:
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role {role!r}: must be one of {sorted(VALID_ROLES)}.")
        if not isinstance(content, str):
            raise ValueError("Turn content must be a string.")
        self._turns.append(
            ConversationTurn(role=role, content=_bound_text(content, self._max_turn_chars))
        )
        max_messages = self._max_turns * 2
        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]

    def add_model_state(self, code: str, description: str, intent: str) -> None:
        self._model_history.append(
            ModelState(
                code=_bound_text(code, self._max_turn_chars),
                description=_bound_text(description, self._max_turn_chars),
                intent=intent,
            )
        )
        if len(self._model_history) > self._max_model_history:
            self._model_history = self._model_history[-self._max_model_history :]

    def get_current_model_state(self) -> ModelState | None:
        return self._model_history[-1] if self._model_history else None

    def to_gemini_messages(self) -> list[dict[str, Any]]:
        """Convert turns into normalized role/parts chat message dictionaries."""
        return [{"role": turn.role, "parts": [turn.content]} for turn in self._turns]

    def reset(self) -> None:
        self._turns.clear()
        self._model_history.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def model_history_count(self) -> int:
        return len(self._model_history)
