"""Pipeline stage helper for building and iteratively refining models."""

from typing import Any

from jaxforge.agent import JaxForge


class ModelBuilder:
    """High-level wrapper around JaxForge model construction and refinement."""

    def __init__(self, agent: JaxForge | None = None):
        self._agent = agent or JaxForge()

    def run(self, prompt: str, refinements: tuple[str, ...] = (), dry_run: bool = False) -> Any:
        """Build a model from ``prompt`` and apply any sequential refinement prompts."""
        model = self._agent.build(prompt, dry_run=dry_run)
        for step in refinements:
            model = self._agent.modify(step, dry_run=dry_run)
        return model
