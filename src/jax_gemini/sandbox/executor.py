import traceback
from typing import Any

from jax_gemini.exceptions import JaxGeminiExecutionError
from jax_gemini.sandbox.namespace import get_execution_namespace

INTENT_FUNCTION_MAP: dict[str, str] = {
    "build": "build_model",
    "train": "train_model",
    "evaluate": "evaluate_model",
    "save": "save_model",
    "load": "load_model",
}

INTENT_DEFAULT_ARGS: dict[str, dict] = {
    "build": {},
    "train": {"dataset": None, "epochs": 5, "learning_rate": 1e-3},
    "evaluate": {"dataset": None},
    "save": {"path": "./checkpoint"},
    "load": {"path": "./checkpoint"},
}


class SandboxExecutor:
    def run(
        self,
        code: str,
        intent: str,
        args: dict | None = None,
    ) -> Any:
        """
        Execute validated code in a restricted namespace.

        Args:
            code: Python source code string (already AST-validated)
            intent: Operation intent for function name lookup
            args: Runtime arguments to pass to the generated function

        Returns:
            Whatever the generated function returns

        Raises:
            JaxGeminiExecutionError: If execution fails
        """
        namespace = get_execution_namespace()
        fn_name = INTENT_FUNCTION_MAP.get(intent, "")
        if not fn_name:
            raise JaxGeminiExecutionError(f"Unknown intent: {intent}")
        call_args = {**INTENT_DEFAULT_ARGS.get(intent, {}), **(args or {})}

        try:
            exec(code, namespace)
        except Exception as e:
            tb = traceback.format_exc()
            raise JaxGeminiExecutionError(
                f"Code failed during definition phase: {e}\n\n{tb}",
                code=code,
                attempts=1,
            )

        if fn_name not in namespace:
            raise JaxGeminiExecutionError(
                f"Function '{fn_name}' not found in namespace after exec. "
                "This is a validator bug — please report it.",
                code=code,
            )

        try:
            result = namespace[fn_name](**call_args)
            return result
        except Exception as e:
            tb = traceback.format_exc()
            raise JaxGeminiExecutionError(
                f"Function '{fn_name}' raised an error during execution:\n"
                f"{type(e).__name__}: {e}\n\nTraceback:\n{tb}",
                code=code,
                attempts=1,
            )
