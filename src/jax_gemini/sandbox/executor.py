import signal
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
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


@contextmanager
def _execution_timeout(timeout_seconds: float) -> Iterator[None]:
    """Enforce a wall-clock timeout via SIGALRM on the main thread or trace fallback."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    use_sigalrm = (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if use_sigalrm:
        def _handle_timeout(signum: int, frame: FrameType | None) -> None:
            raise _SandboxTimeoutError(f"Execution timed out after {timeout_seconds}s")

        prev_handler = signal.signal(signal.SIGALRM, _handle_timeout)
        prev_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prev_handler)
            if prev_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, prev_timer[0], prev_timer[1])
    else:
        deadline = time.monotonic() + timeout_seconds
        prev_trace = sys.gettrace()

        def _trace_deadline(frame: FrameType, event: str, arg: Any) -> Any:
            if time.monotonic() > deadline:
                raise _SandboxTimeoutError(f"Execution timed out after {timeout_seconds}s")
            return _trace_deadline

        sys.settrace(_trace_deadline)
        try:
            yield
        finally:
            sys.settrace(prev_trace)


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
            with _execution_timeout(timeout_seconds):
                exec(code, namespace)
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
                "This is a validator bug — please report it.",
                code=code,
            )

        try:
            with _execution_timeout(timeout_seconds):
                result = namespace[fn_name](**call_args)
            return result
        except (_SandboxTimeoutError, TimeoutError, Exception) as e:
            tb = traceback.format_exc()
            raise JaxGeminiExecutionError(
                f"Function '{fn_name}' raised an error during execution:\n"
                f"{type(e).__name__}: {e}\n\nTraceback:\n{tb}",
                code=code,
                attempts=1,
            ) from e
