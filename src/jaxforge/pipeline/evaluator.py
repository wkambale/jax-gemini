"""Pipeline stage helper for model evaluation."""

from typing import Any, cast

from jaxforge.agent import JaxForge


class ModelEvaluator:
    """High-level wrapper around JaxForge evaluation."""

    def __init__(self, agent: JaxForge | None = None):
        self._agent = agent or JaxForge()

    def run(self, prompt: str, dataset: tuple[Any, ...] | None = None) -> dict[str, Any]:
        """Evaluate the active model and return metrics."""
        return cast("dict[str, Any]", self._agent.evaluate(prompt, dataset=dataset))
