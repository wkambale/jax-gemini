"""Abstract LLM interface and token/latency usage tracking for Jax Forge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMUsage:
    """Telemetry captured for a single LLM generation call."""

    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


class AbstractLLM(ABC):
    """Provider-agnostic interface for structured JSON code generation backends."""

    last_usage: LLMUsage | None = None

    @abstractmethod
    def generate(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        """Send chat messages and return a parsed JSON dict matching ``schema``."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the configured backend and credentials respond cleanly."""
        ...
