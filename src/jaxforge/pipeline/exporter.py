"""Pipeline stage helper for saving and restoring checkpoints."""

from typing import Any

from jaxforge.agent import JaxForge


class ModelExporter:
    """High-level wrapper around JaxForge checkpoint persistence."""

    def __init__(self, agent: JaxForge | None = None):
        self._agent = agent or JaxForge()

    def export(self, name: str) -> str:
        """Save the active model checkpoint under ``name`` inside checkpoint_dir."""
        return str(self._agent.save(name))

    def restore(self, name: str) -> Any:
        """Load a model checkpoint from ``name`` inside checkpoint_dir."""
        return self._agent.load(name)
