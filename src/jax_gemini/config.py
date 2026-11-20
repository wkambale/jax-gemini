import math
import os
from dataclasses import dataclass, field, fields
from typing import Any

from jax_gemini.exceptions import JaxGeminiConfigError


@dataclass
class JaxGeminiConfig:
    gemini_api_key: str | None = field(default=None, repr=False)
    model_name: str = "gemini-3.1-pro"
    temperature: float = 0.2  # Low temp for deterministic code
    max_output_tokens: int = 4096
    max_retries: int = 3
    verbose: bool = False
    checkpoint_dir: str = "./jax_gemini_checkpoints"
    timeout_seconds: int = 60

    def _validate_field_values(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise JaxGeminiConfigError("model_name must be a non-empty string")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature < 0
            or self.temperature > 1
        ):
            raise JaxGeminiConfigError("temperature must be a finite number between 0.0 and 1.0")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens < 1
            or self.max_output_tokens > 65536
        ):
            raise JaxGeminiConfigError("max_output_tokens must be an integer between 1 and 65536")
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 1
            or self.max_retries > 5
        ):
            raise JaxGeminiConfigError("max_retries must be an integer between 1 and 5")
        if not isinstance(self.verbose, bool):
            raise JaxGeminiConfigError("verbose must be a boolean")
        if (
            not isinstance(self.checkpoint_dir, str)
            or not self.checkpoint_dir.strip()
            or "\x00" in self.checkpoint_dir
            or "://" in self.checkpoint_dir
        ):
            raise JaxGeminiConfigError("checkpoint_dir must be a non-empty local directory path")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or self.timeout_seconds < 1
            or self.timeout_seconds > 3600
        ):
            raise JaxGeminiConfigError("timeout_seconds must be an integer between 1 and 3600")
        if self.gemini_api_key is not None and (
            not isinstance(self.gemini_api_key, str) or not self.gemini_api_key.strip()
        ):
            raise JaxGeminiConfigError("gemini_api_key must be a non-empty string or None")

    def set(self, updates: dict[str, Any]) -> None:
        """Update declared config fields from a dict. Used as jg.config.set({...})."""
        if not isinstance(updates, dict):
            raise JaxGeminiConfigError("Config updates must be a dictionary")

        allowed_fields = {f.name for f in fields(self)}
        previous: dict[str, Any] = {}
        for key in updates:
            if key not in allowed_fields:
                raise JaxGeminiConfigError(f"Unknown config key: '{key}'")
            previous[key] = getattr(self, key)

        try:
            for key, value in updates.items():
                setattr(self, key, value)
            self._validate_field_values()
        except Exception:
            for key, old_val in previous.items():
                setattr(self, key, old_val)
            raise

    def validate(self) -> None:
        """Raise JaxGeminiConfigError if config is invalid."""
        effective_key = self.gemini_api_key
        if not effective_key or not str(effective_key).strip():
            effective_key = os.environ.get("GEMINI_API_KEY")
        if (
            not effective_key
            or not isinstance(effective_key, str)
            or not effective_key.strip()
        ):
            raise JaxGeminiConfigError(
                "gemini_api_key is not set. "
                "Call jg.config.set({'gemini_api_key': 'YOUR_KEY'}) first, "
                "or set the GEMINI_API_KEY environment variable."
            )
        self._validate_field_values()
