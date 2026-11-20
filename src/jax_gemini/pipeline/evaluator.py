"""Pipeline stage helper for model evaluation."""

from typing import Any

from jax_gemini.agent import JaxGemini


class ModelEvaluator:
    """High-level wrapper around JaxGemini evaluation."""

    def __init__(self, agent: JaxGemini | None = None):
        self._agent = agent or JaxGemini()

    def run(self, prompt: str, dataset: tuple[Any, ...] | None = None) -> dict[str, Any]:
        """Evaluate the active model and return metrics."""
        return self._agent.evaluate(prompt, dataset=dataset)
