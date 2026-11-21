"""Backward-compatibility shim redirecting jax_gemini imports to jaxforge."""

import warnings

from jaxforge import (
    JaxForge as JaxGemini,
)
from jaxforge import (
    JaxForgeApprovalError,
    __author__,
    __license__,
    __version__,
    analyze_data,
    build,
    config,
    evaluate,
    execute_plan,
    explain,
    explore,
    export_project,
    load,
    load_data,
    modify,
    preprocess_data,
    preview,
    reset,
    run,
    save,
    show_code,
    train,
    usage_summary,
    visualize_data,
)
from jaxforge import (
    JaxForgeApprovalError as JaxGeminiApprovalError,
)
from jaxforge import (
    JaxForgeCheckpointError as JaxGeminiCheckpointError,
)
from jaxforge import (
    JaxForgeConfig as JaxGeminiConfig,
)
from jaxforge import (
    JaxForgeConfigError as JaxGeminiConfigError,
)
from jaxforge import (
    JaxForgeError as JaxGeminiError,
)
from jaxforge import (
    JaxForgeExecutionError as JaxGeminiExecutionError,
)
from jaxforge import (
    JaxForgeIntentError as JaxGeminiIntentError,
)
from jaxforge import (
    JaxForgeLLMError as JaxGeminiLLMError,
)
from jaxforge import (
    JaxForgeValidationError as JaxGeminiValidationError,
)

warnings.warn(
    "jax_gemini is deprecated and has been renamed to jaxforge. "
    "Install 'jaxforge' and use 'import jaxforge as jf'.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "JaxForgeApprovalError",
    "JaxGemini",
    "JaxGeminiApprovalError",
    "JaxGeminiCheckpointError",
    "JaxGeminiConfig",
    "JaxGeminiConfigError",
    "JaxGeminiError",
    "JaxGeminiExecutionError",
    "JaxGeminiIntentError",
    "JaxGeminiLLMError",
    "JaxGeminiValidationError",
    "__author__",
    "__license__",
    "__version__",
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
