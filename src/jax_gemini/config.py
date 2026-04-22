from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JaxGeminiConfig:
    gemini_api_key: Optional[str] = None
    model_name: str = "gemini-3.1-pro"
    temperature: float = 0.2          # Low temp for deterministic code
    max_output_tokens: int = 4096
    max_retries: int = 3
    verbose: bool = False
    checkpoint_dir: str = "./jax_gemini_checkpoints"
    timeout_seconds: int = 60

    def set(self, updates: dict) -> None:
        """Update config from a dict. Used as jg.config.set({...})."""
        from jax_gemini.exceptions import JaxGeminiConfigError
        for key, value in updates.items():
            if not hasattr(self, key):
                raise JaxGeminiConfigError(f"Unknown config key: '{key}'")
            setattr(self, key, value)

    def validate(self) -> None:
        """Raise JaxGeminiConfigError if config is invalid."""
        from jax_gemini.exceptions import JaxGeminiConfigError
        import os
        if not self.gemini_api_key:
            self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise JaxGeminiConfigError(
                "gemini_api_key is not set. "
                "Call jg.config.set({'gemini_api_key': 'YOUR_KEY'}) first, "
                "or set the GEMINI_API_KEY environment variable."
            )
        if self.temperature < 0 or self.temperature > 1:
            raise JaxGeminiConfigError("temperature must be between 0.0 and 1.0")
        if self.max_retries < 1 or self.max_retries > 5:
            raise JaxGeminiConfigError("max_retries must be between 1 and 5")
