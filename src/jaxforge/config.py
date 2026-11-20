"""Configuration dataclass and atomic field validator for Jax Forge."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields
from typing import Any

from jaxforge.exceptions import JaxForgeConfigError

SUPPORTED_PROVIDERS = frozenset(
    {"auto", "google", "openai", "anthropic", "ollama", "openrouter", "litellm"}
)
SUPPORTED_BACKENDS = frozenset({"thread", "subprocess", "docker", "e2b", "modal"})
SUPPORTED_FRAMEWORKS = frozenset({"flax", "equinox", "auto"})


@dataclass
class JaxForgeConfig:
    """Runtime configuration for multi-provider LLM routing, validation, and sandboxing."""

    gemini_api_key: str | None = field(default=None, repr=False)
    api_key: str | None = field(default=None, repr=False)
    provider: str = "auto"
    model_name: str = "gemini-3.1-pro"
    fast_model_name: str | None = None
    base_url: str | None = None
    temperature: float = 0.2
    max_output_tokens: int = 4096
    max_retries: int = 3
    verbose: bool = False
    checkpoint_dir: str = "./jaxforge_checkpoints"
    timeout_seconds: int = 60
    execution_backend: str = "thread"
    max_memory_mb: int | None = None
    auto_approve: bool = True
    framework: str = "flax"

    def resolve_provider(self) -> str:
        """Determine the normalized provider name from config or model_name prefix."""
        if ":" in self.model_name:
            prefix = self.model_name.split(":", 1)[0].strip().lower()
            if prefix in SUPPORTED_PROVIDERS - {"auto"}:
                return prefix
        if self.provider != "auto":
            return self.provider.lower()
        lower = self.model_name.lower()
        if lower.startswith(("claude", "anthropic/")):
            return "anthropic"
        if lower.startswith(("gpt-", "o1", "o3", "o4", "openai/")):
            return "openai"
        if lower.startswith("openrouter/"):
            return "openrouter"
        if self.base_url and "11434" in self.base_url:
            return "ollama"
        if lower.startswith(("ollama/", "llama", "qwen", "deepseek", "mistral", "codellama")):
            return "ollama"
        return "google"

    def resolve_model_id(self) -> str:
        """Return model_name stripped of any provider: prefix."""
        if ":" in self.model_name:
            prefix, rest = self.model_name.split(":", 1)
            if prefix.strip().lower() in SUPPORTED_PROVIDERS - {"auto"} and rest.strip():
                return rest.strip()
        return self.model_name

    def get_effective_api_key(self) -> str | None:
        """Resolve an explicit key or matching environment variable without mutating state."""
        explicit = self.api_key or self.gemini_api_key
        if explicit and str(explicit).strip():
            return str(explicit).strip()
        prov = self.resolve_provider()
        env_candidates: dict[str, tuple[str, ...]] = {
            "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            "openai": ("OPENAI_API_KEY",),
            "anthropic": ("ANTHROPIC_API_KEY",),
            "openrouter": ("OPENROUTER_API_KEY",),
            "ollama": ("OLLAMA_API_KEY", "OPENAI_API_KEY"),
            "litellm": ("LITELLM_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"),
        }
        for env_var in env_candidates.get(prov, ("GEMINI_API_KEY",)):
            val = os.environ.get(env_var)
            if val and val.strip():
                return val.strip()
        return None

    def _validate_field_values(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise JaxForgeConfigError("model_name must be a non-empty string")
        if self.fast_model_name is not None and (
            not isinstance(self.fast_model_name, str) or not self.fast_model_name.strip()
        ):
            raise JaxForgeConfigError("fast_model_name must be a non-empty string or None")
        if not isinstance(self.provider, str) or self.provider.lower() not in SUPPORTED_PROVIDERS:
            raise JaxForgeConfigError(
                f"provider must be one of {sorted(SUPPORTED_PROVIDERS)}, got {self.provider!r}"
            )
        if (
            not isinstance(self.execution_backend, str)
            or self.execution_backend not in SUPPORTED_BACKENDS
        ):
            raise JaxForgeConfigError(
                f"execution_backend must be one of {sorted(SUPPORTED_BACKENDS)}, "
                f"got {self.execution_backend!r}"
            )
        if not isinstance(self.framework, str) or self.framework not in SUPPORTED_FRAMEWORKS:
            raise JaxForgeConfigError(
                f"framework must be one of {sorted(SUPPORTED_FRAMEWORKS)}, got {self.framework!r}"
            )
        if self.base_url is not None:
            if (
                not isinstance(self.base_url, str)
                or not self.base_url.strip()
                or not self.base_url.startswith(("http://", "https://"))
                or any(ch in self.base_url for ch in ("\n", "\r", "\x00"))
            ):
                raise JaxForgeConfigError("base_url must be a valid http(s) URL or None")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature < 0
            or self.temperature > 1
        ):
            raise JaxForgeConfigError("temperature must be a finite number between 0.0 and 1.0")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens < 1
            or self.max_output_tokens > 65536
        ):
            raise JaxForgeConfigError("max_output_tokens must be an integer between 1 and 65536")
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 1
            or self.max_retries > 5
        ):
            raise JaxForgeConfigError("max_retries must be an integer between 1 and 5")
        if not isinstance(self.verbose, bool):
            raise JaxForgeConfigError("verbose must be a boolean")
        if not isinstance(self.auto_approve, bool):
            raise JaxForgeConfigError("auto_approve must be a boolean")
        if (
            not isinstance(self.checkpoint_dir, str)
            or not self.checkpoint_dir.strip()
            or "\x00" in self.checkpoint_dir
            or "://" in self.checkpoint_dir
        ):
            raise JaxForgeConfigError("checkpoint_dir must be a non-empty local directory path")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or self.timeout_seconds < 1
            or self.timeout_seconds > 3600
        ):
            raise JaxForgeConfigError("timeout_seconds must be an integer between 1 and 3600")
        if self.max_memory_mb is not None and (
            isinstance(self.max_memory_mb, bool)
            or not isinstance(self.max_memory_mb, int)
            or self.max_memory_mb < 64
        ):
            raise JaxForgeConfigError("max_memory_mb must be an integer >= 64 or None")
        for key_name, key_val in (
            ("gemini_api_key", self.gemini_api_key),
            ("api_key", self.api_key),
        ):
            if key_val is not None and (
                not isinstance(key_val, str) or not key_val.strip()
            ):
                raise JaxForgeConfigError(f"{key_name} must be a non-empty string or None")

    def set(self, updates: dict[str, Any]) -> None:
        """Atomically validate and update declared configuration fields."""
        if not isinstance(updates, dict):
            raise JaxForgeConfigError("Config updates must be a dictionary")

        allowed_fields = {f.name for f in fields(self)}
        previous: dict[str, Any] = {}
        for key in updates:
            if key not in allowed_fields:
                raise JaxForgeConfigError(f"Unknown config key: '{key}'")
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
        """Validate all configuration fields and require credentials when needed."""
        self._validate_field_values()
        prov = self.resolve_provider()
        if prov == "ollama":
            return
        effective_key = self.get_effective_api_key()
        if not effective_key:
            raise JaxForgeConfigError(
                f"API key is not set for provider '{prov}'. "
                "Call jf.config.set({'api_key': 'YOUR_KEY'}) or set the corresponding "
                "environment variable (e.g., GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY)."
            )


JaxGeminiConfig = JaxForgeConfig
