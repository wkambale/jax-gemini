"""Pipeline stage helper for training runs and metric tracking."""

from typing import Any

from jaxforge.agent import JaxForge


class ModelTrainer:
    """High-level wrapper around JaxForge training with metric history tracking."""

    def __init__(self, agent: JaxForge | None = None):
        self._agent = agent or JaxForge()
        self.history: list[dict[str, Any]] = []

    def run(
        self, prompt: str, dataset: tuple[Any, ...] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Execute a training run and record returned metrics."""
        model, metrics = self._agent.train(prompt, dataset=dataset)
        self.history.append(dict(metrics))
        return model, metrics
