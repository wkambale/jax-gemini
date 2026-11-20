"""Pipeline stage helper for building and iteratively refining models."""

from typing import Any

from jax_gemini.agent import JaxGemini


class ModelBuilder:
    """High-level wrapper around JaxGemini model construction and refinement."""

    def __init__(self, agent: JaxGemini | None = None):
        self._agent = agent or JaxGemini()

    def run(self, prompt: str, refinements: tuple[str, ...] = ()) -> Any:
        """Build a model from ``prompt`` and apply any sequential refinement prompts."""
        model = self._agent.build(prompt)
        for step in refinements:
            model = self._agent.modify(step)
        return model
