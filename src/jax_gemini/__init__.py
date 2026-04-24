"""
jax-gemini: Natural language-driven JAX/Flax model building powered by Gemini.

Quick start:
    import jax_gemini as jg
    jg.config.set({"gemini_api_key": "YOUR_KEY"})
    model = jg.build("Build a 3-layer MLP for image classification")
"""

from jax_gemini.agent import JaxGemini
from jax_gemini.config import JaxGeminiConfig
from jax_gemini.exceptions import (
    JaxGeminiCheckpointError,
    JaxGeminiConfigError,
    JaxGeminiError,
    JaxGeminiExecutionError,
    JaxGeminiIntentError,
    JaxGeminiLLMError,
    JaxGeminiValidationError,
)

__version__ = "0.1.2"
__author__ = "Wesley Kambale"
__license__ = "MIT"

__all__ = [
    "JaxGemini",
    "JaxGeminiCheckpointError",
    "JaxGeminiConfig",
    "JaxGeminiConfigError",
    "JaxGeminiError",
    "JaxGeminiExecutionError",
    "JaxGeminiIntentError",
    "JaxGeminiLLMError",
    "JaxGeminiValidationError",
    "analyze_data",
    "build",
    "config",
    "evaluate",
    "explain",
    "load",
    "load_data",
    "modify",
    "preprocess_data",
    "reset",
    "save",
    "show_code",
    "train",
    "visualize_data",
]

# Module-level singleton for simple usage pattern (jg.build, jg.train, etc.)
_default_agent = JaxGemini()
config = _default_agent.config
build = _default_agent.build
modify = _default_agent.modify
train = _default_agent.train
evaluate = _default_agent.evaluate
save = _default_agent.save
load = _default_agent.load
explain = _default_agent.explain
show_code = _default_agent.show_code
reset = _default_agent.reset
load_data = _default_agent.load_data
preprocess_data = _default_agent.preprocess_data
analyze_data = _default_agent.analyze_data
visualize_data = _default_agent.visualize_data
