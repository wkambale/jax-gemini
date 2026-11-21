"""
jaxforge (Jax Forge): Model-agnostic natural language engineering workbench for JAX and Flax.
"""

from jaxforge.agent import JaxForge
from jaxforge.config import JaxForgeConfig
from jaxforge.exceptions import (
    JaxForgeApprovalError,
    JaxForgeCheckpointError,
    JaxForgeConfigError,
    JaxForgeError,
    JaxForgeExecutionError,
    JaxForgeIntentError,
    JaxForgeLLMError,
    JaxForgeValidationError,
)
from jaxforge.plan import DatasetProfile, ExperimentCandidate, ExperimentResult, GeneratedPlan

__version__ = "0.2.0"
__author__ = "Wesley Kambale"
__license__ = "MIT"

__all__ = [
    "DatasetProfile",
    "ExperimentCandidate",
    "ExperimentResult",
    "GeneratedPlan",
    "JaxForge",
    "JaxForgeApprovalError",
    "JaxForgeCheckpointError",
    "JaxForgeConfig",
    "JaxForgeConfigError",
    "JaxForgeError",
    "JaxForgeExecutionError",
    "JaxForgeIntentError",
    "JaxForgeLLMError",
    "JaxForgeValidationError",
    "analyze_data",
    "build",
    "config",
    "evaluate",
    "execute_plan",
    "explain",
    "explore",
    "export_project",
    "load",
    "load_data",
    "modify",
    "preprocess_data",
    "preview",
    "reset",
    "run",
    "save",
    "show_code",
    "train",
    "usage_summary",
    "visualize_data",
]

# Module-level singleton for single-session interactive usage.
# Multi-user or concurrent callers must instantiate separate JaxForge() instances.
_default_agent = JaxForge()
config = _default_agent.config
build = _default_agent.build
modify = _default_agent.modify
train = _default_agent.train
evaluate = _default_agent.evaluate
explore = _default_agent.explore
preview = _default_agent.preview
execute_plan = _default_agent.execute_plan
export_project = _default_agent.export_project
save = _default_agent.save
load = _default_agent.load
explain = _default_agent.explain
show_code = _default_agent.show_code
reset = _default_agent.reset
usage_summary = _default_agent.usage_summary
load_data = _default_agent.load_data
preprocess_data = _default_agent.preprocess_data
analyze_data = _default_agent.analyze_data
visualize_data = _default_agent.visualize_data
run = _default_agent.run
