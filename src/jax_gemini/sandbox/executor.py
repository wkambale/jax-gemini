import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

from jax_gemini.codegen.validator import CodeValidator
from jax_gemini.exceptions import JaxGeminiExecutionError
from jax_gemini.sandbox.namespace import get_execution_namespace

INTENT_FUNCTION_MAP: dict[str, str] = {
    "build": "build_model",
    "train": "train_model",
    "evaluate": "evaluate_model",
    "save": "save_model",
    "load": "load_model",
    "load_data": "load_data",
    "preprocess_data": "preprocess_data",
    "analyze_data": "analyze_data",
    "visualize_data": "visualize_data",
}

INTENT_DEFAULT_ARGS: dict[str, dict[str, Any]] = {
    "build": {},
    "train": {"dataset": None, "epochs": 5, "learning_rate": 1e-3},
    "evaluate": {"dataset": None},
    "save": {"path": "./checkpoint"},
    "load": {"path": "./checkpoint"},
}


class _SandboxTimeoutError(BaseException):
    """Unrecoverable timeout raised inside the sandbox so except Exception cannot swallow it."""


def _run_with_timeout(fn: Callable[[], Any], timeout_seconds: float) -> Any:
    """Execute ``fn`` in a daemon worker thread enforced by trace deadlines and join timeout."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    result_box: list[Any] = []
    error_box: list[BaseException] = []
    deadline = time.monotonic() + timeout_seconds

    def _trace_deadline(frame: FrameType, event: str, arg: Any) -> Any:
        if time.monotonic() > deadline:
            raise _SandboxTimeoutError(f"Execution timed out after {timeout_seconds}s")
        return _trace_deadline

    def _target() -> None:
        sys.settrace(_trace_deadline)
        try:
            result_box.append(fn())
        except BaseException as exc:
            error_box.append(exc)
        finally:
            sys.settrace(None)

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=timeout_seconds + 0.5)
    if worker.is_alive():
        raise _SandboxTimeoutError(f"Execution timed out after {timeout_seconds}s")
    if error_box:
        raise error_box[0]
    return result_box[0]


class SandboxExecutor:
    def run(
        self,
        code: str,
        intent: str,
        args: dict[str, Any] | None = None,
        allowed_dirs: tuple[Path | str, ...] | None = None,
        timeout_seconds: float = 60.0,
    ) -> Any:
        """
        Execute validated code in a restricted namespace.

        Args:
            code: Python source code string (already AST-validated)
            intent: Operation intent for function name lookup
            args: Runtime arguments to pass to the generated function
            allowed_dirs: Directories to which sandbox file I/O is confined
            timeout_seconds: Maximum wall-clock seconds allowed for execution

        Returns:
            Whatever the generated function returns

        Raises:
            JaxGeminiValidationError: If code fails AST validation
            JaxGeminiExecutionError: If execution fails or times out
        """
        CodeValidator.validate(code, intent)
        namespace = get_execution_namespace(intent=intent, allowed_dirs=allowed_dirs)
        fn_name = INTENT_FUNCTION_MAP.get(intent, "")
        if not fn_name:
            raise JaxGeminiExecutionError(f"Unknown intent: {intent}")
        call_args = {**INTENT_DEFAULT_ARGS.get(intent, {}), **(args or {})}

        try:
            _run_with_timeout(lambda: exec(code, namespace), timeout_seconds)
        except (_SandboxTimeoutError, TimeoutError, Exception) as e:
            tb = traceback.format_exc()
            raise JaxGeminiExecutionError(
                f"Code failed during definition phase: {e}\n\n{tb}",
                code=code,
                attempts=1,
            ) from e

        if fn_name not in namespace:
            raise JaxGeminiExecutionError(
                f"Function '{fn_name}' not found in namespace after exec. "
                "This is a validator bug; please report it.",
                code=code,
            )

        try:
            return _run_with_timeout(lambda: namespace[fn_name](**call_args), timeout_seconds)
        except (_SandboxTimeoutError, TimeoutError, Exception) as e:
            tb = traceback.format_exc()
            raise JaxGeminiExecutionError(
                f"Function '{fn_name}' raised an error during execution:\n"
                f"{type(e).__name__}: {e}\n\nTraceback:\n{tb}",
                code=code,
                attempts=1,
            ) from e
