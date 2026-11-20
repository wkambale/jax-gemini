"""Pipeline stage helper for saving and restoring checkpoints."""

from typing import Any

from jax_gemini.agent import JaxGemini


class ModelExporter:
    """High-level wrapper around JaxGemini checkpoint persistence."""

    def __init__(self, agent: JaxGemini | None = None):
        self._agent = agent or JaxGemini()

    def export(self, name: str) -> str:
        """Save the active model checkpoint under ``name`` inside checkpoint_dir."""
        return self._agent.save(name)

    def restore(self, name: str) -> Any:
        """Load a model checkpoint from ``name`` inside checkpoint_dir."""
        return self._agent.load(name)
