import copy
import os
from pathlib import Path
from typing import Any, cast

import flax.nnx as nnx

from jax_gemini.codegen.fixer import CodeFixer
from jax_gemini.codegen.parser import CodeParser
from jax_gemini.codegen.validator import CodeValidator
from jax_gemini.config import JaxGeminiConfig
from jax_gemini.exceptions import (
    JaxGeminiCheckpointError,
    JaxGeminiExecutionError,
    JaxGeminiIntentError,
    JaxGeminiLLMError,
    JaxGeminiValidationError,
)
from jax_gemini.llm.gemini import GeminiLLM
from jax_gemini.memory.conversation import ConversationMemory
from jax_gemini.prompts.context_builder import MAX_CONTEXT_PROMPT_CHARS, ContextBuilder
from jax_gemini.prompts.intent import IntentClassifier
from jax_gemini.sandbox.executor import SandboxExecutor


def _snapshot_runtime_args(runtime_args: dict[str, Any] | None) -> dict[str, Any] | None:
    """Create an isolated copy of mutable runtime_args so failed retries do not mutate state."""
    if runtime_args is None:
        return None
    snapshot: dict[str, Any] = {}
    for key, val in runtime_args.items():
        try:
            snapshot[key] = copy.deepcopy(val)
        except Exception:
            snapshot[key] = val
    return snapshot


def _sanitize_validation_feedback(err: JaxGeminiValidationError) -> str:
    """Avoid exposing detailed AST security policy internals as an interactive bypass oracle."""
    msg = str(err)
    if "syntax error" in msg.lower() or "must define a top-level function" in msg.lower():
        return msg
    return "Generated code violated sandbox validation rules; use only allowed ML operations."


class JaxGemini:
    def __init__(self, config: JaxGeminiConfig | None = None):
        self.config = config or JaxGeminiConfig()
        self._llm: GeminiLLM | None = None
        self._llm_api_key: str | None = None
        self._memory = ConversationMemory()
        self._executor = SandboxExecutor()
        self._last_code: str | None = None
        self._last_response: dict[str, Any] | None = None
        self._current_model: Any = None

    def _get_llm(self) -> GeminiLLM:
        self.config.validate()
        effective_key = self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if self._llm is None or self._llm_api_key != effective_key:
            self._llm = GeminiLLM(self.config)
            self._llm_api_key = effective_key
        return self._llm

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[jax-gemini] {message}")

    def _generate_and_execute(
        self,
        user_prompt: str,
        intent: str,
        runtime_args: dict[str, Any] | None = None,
    ) -> Any:
        ContextBuilder.validate_prompt(user_prompt)
        llm = self._get_llm()
        context = ContextBuilder.build(
            intent=intent,
            user_prompt=user_prompt,
            memory=self._memory,
            current_model_state=self._memory.get_current_model_state(),
        )
        schema = ContextBuilder.get_schema(intent)
        last_error: Exception | None = None
        last_feedback: str = ""
        last_code: str = ""

        for attempt in range(1, self.config.max_retries + 1):
            self._log(f"Attempt {attempt}/{self.config.max_retries} for intent '{intent}'...")

            if attempt > 1 and last_error is not None:
                fix_prompt = CodeFixer.build_fix_prompt(
                    original_prompt=user_prompt,
                    failed_code=last_code,
                    error_message=last_feedback or str(last_error),
                    attempt=attempt,
                    intent=intent,
                )
                context = ContextBuilder.build(
                    intent=intent,
                    user_prompt=fix_prompt,
                    memory=self._memory,
                    current_model_state=self._memory.get_current_model_state(),
                    max_prompt_chars=MAX_CONTEXT_PROMPT_CHARS,
                )

            try:
                response = llm.generate(context, schema)
                code = CodeParser.extract(response, intent)
                CodeValidator.validate(code, intent)
                allowed_dirs = (
                    (Path(self.config.checkpoint_dir),)
                    if intent in ("save", "load")
                    else (Path.cwd(), Path(self.config.checkpoint_dir))
                )
                attempt_args = _snapshot_runtime_args(runtime_args)
                result = self._executor.run(
                    code,
                    intent,
                    attempt_args,
                    allowed_dirs=allowed_dirs,
                    timeout_seconds=self.config.timeout_seconds,
                )

                # Success — update state
                self._last_code = code
                self._last_response = response
                self._memory.add_turn("user", user_prompt)
                self._memory.add_turn("model", response.get("description", code))
                if intent in ("build", "train"):
                    self._memory.add_model_state(
                        code=code,
                        description=response.get("description", ""),
                        intent=intent,
                    )
                self._log(f"Attempt {attempt} succeeded.")
                return result

            except JaxGeminiValidationError as e:
                last_error = e
                last_feedback = _sanitize_validation_feedback(e)
                last_code = getattr(e, "code", "")
                self._log(f"Attempt {attempt} failed validation: {e}")
                if attempt < self.config.max_retries:
                    self._log("Sending sanitized validation context to Gemini for correction...")

            except JaxGeminiExecutionError as e:
                last_error = e
                last_feedback = str(e)
                last_code = getattr(e, "code", "")
                self._log(f"Attempt {attempt} failed execution: {e}")
                if attempt < self.config.max_retries:
                    self._log("Sending sanitized error context to Gemini for correction...")

            except JaxGeminiLLMError as e:
                last_error = e
                last_feedback = (
                    "Previous response was malformed or failed to parse as valid JSON; "
                    "respond strictly with the requested JSON schema."
                )
                last_code = ""
                self._log(f"Attempt {attempt} failed LLM response parsing: {e}")
                if attempt < self.config.max_retries:
                    self._log("Retrying Gemini generation...")

        if isinstance(last_error, JaxGeminiLLMError):
            raise last_error
        if isinstance(last_error, JaxGeminiValidationError):
            raise last_error

        raise JaxGeminiExecutionError(
            f"All {self.config.max_retries} attempts failed for '{user_prompt}'.\n"
            f"Last error: {last_error}",
            code=last_code,
            attempts=self.config.max_retries,
        )

    def build(self, prompt: str) -> Any:
        """Build a JAX/Flax model from a natural language description."""
        model = self._generate_and_execute(
            prompt, intent="build", runtime_args={"rngs": nnx.Rngs(0)}
        )
        self._current_model = model
        return model

    def modify(self, prompt: str) -> Any:
        """Refine the current model based on a natural language instruction."""
        model = self._generate_and_execute(
            prompt, intent="build", runtime_args={"rngs": nnx.Rngs(0)}
        )
        self._current_model = model
        return model

    def train(
        self, prompt: str, dataset: tuple[Any, ...] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Train the current model. Returns (model, metrics_dict)."""
        args: dict[str, Any] = {"dataset": dataset} if dataset else {}
        if self._current_model is not None:
            args["model"] = self._current_model
        result = self._generate_and_execute(prompt, intent="train", runtime_args=args)
        if isinstance(result, tuple):
            self._current_model = result[0]
            return cast("tuple[Any, dict[str, Any]]", result)
        return result, {}

    def evaluate(
        self, prompt: str, dataset: tuple[Any, ...] | None = None
    ) -> dict[str, Any]:
        """Evaluate the current model. Returns a metrics dict."""
        args: dict[str, Any] = {"dataset": dataset} if dataset else {}
        if self._current_model is not None:
            args["model"] = self._current_model
        return cast(
            "dict[str, Any]",
            self._generate_and_execute(prompt, intent="evaluate", runtime_args=args),
        )

    def _resolve_checkpoint_path(self, name: str) -> str:
        """Resolve a checkpoint name strictly inside self.config.checkpoint_dir."""
        if not isinstance(name, str):
            raise JaxGeminiCheckpointError("Checkpoint name must be a string.")
        cleaned = name.strip()
        if (
            not cleaned
            or "\x00" in cleaned
            or "\n" in cleaned
            or "\r" in cleaned
            or "://" in cleaned
        ):
            raise JaxGeminiCheckpointError(f"Invalid checkpoint name: '{name}'")
        raw = Path(cleaned)
        if raw.is_absolute() or os.path.isabs(cleaned) or ".." in raw.parts:
            raise JaxGeminiCheckpointError(
                f"Checkpoint name '{name}' must be a relative path within checkpoint_dir."
            )
        base_dir = Path(self.config.checkpoint_dir).resolve()
        candidate = (base_dir / raw).resolve()
        if candidate == base_dir or not candidate.is_relative_to(base_dir):
            raise JaxGeminiCheckpointError(
                f"Checkpoint path '{name}' escapes checkpoint_dir '{self.config.checkpoint_dir}'."
            )
        return str(candidate)

    def save(self, name: str) -> str:
        """Save the current model checkpoint. Returns the checkpoint path."""
        path = self._resolve_checkpoint_path(name)
        return str(
            self._generate_and_execute(
                "Save the model checkpoint to the provided path argument.",
                intent="save",
                runtime_args={"model": self._current_model, "path": path},
            )
        )

    def load(self, name: str) -> Any:
        """Load a model checkpoint by name."""
        path = self._resolve_checkpoint_path(name)
        model = self._generate_and_execute(
            "Load the model checkpoint from the provided path argument.",
            intent="load",
            runtime_args={"path": path},
        )
        self._current_model = model
        return model

    def load_data(self, prompt: str) -> Any:
        """Load a dataset based on natural language description."""
        return self._generate_and_execute(prompt, intent="load_data")

    def preprocess_data(self, prompt: str, dataset: Any) -> Any:
        """Preprocess a given dataset based on natural language instruction."""
        return self._generate_and_execute(
            prompt, intent="preprocess_data", runtime_args={"dataset": dataset}
        )

    def analyze_data(self, prompt: str, dataset: Any) -> Any:
        """Analyze a given dataset and return statistics or representations."""
        return self._generate_and_execute(
            prompt, intent="analyze_data", runtime_args={"dataset": dataset}
        )

    def visualize_data(self, prompt: str, dataset: Any) -> Any:
        """Visualize a given dataset by plotting."""
        return self._generate_and_execute(
            prompt, intent="visualize_data", runtime_args={"dataset": dataset}
        )

    def run(
        self,
        prompt: str,
        dataset: Any = None,
        name: str | None = None,
    ) -> Any:
        """Route a natural-language prompt automatically via IntentClassifier."""
        intent = IntentClassifier.classify(prompt)
        if intent == "build":
            return self.modify(prompt) if self._current_model is not None else self.build(prompt)
        if intent == "train":
            return self.train(prompt, dataset=dataset)
        if intent == "evaluate":
            return self.evaluate(prompt, dataset=dataset)
        if intent == "save":
            if not name:
                raise JaxGeminiCheckpointError("Saving a model requires a checkpoint 'name'.")
            return self.save(name)
        if intent == "load":
            if not name:
                raise JaxGeminiCheckpointError("Loading a model requires a checkpoint 'name'.")
            return self.load(name)
        if intent == "load_data":
            return self.load_data(prompt)
        if dataset is None:
            raise JaxGeminiIntentError(f"Intent '{intent}' requires a 'dataset' argument.")
        if intent == "preprocess_data":
            return self.preprocess_data(prompt, dataset)
        if intent == "analyze_data":
            return self.analyze_data(prompt, dataset)
        return self.visualize_data(prompt, dataset)

    def explain(self) -> str:
        """Return a plain-English explanation of the last generated code."""
        if not self._last_code:
            return "No code has been generated yet. Call jg.build() first."
        if self._last_response and "description" in self._last_response:
            return str(self._last_response["description"])
        return "No explanation available for the last operation."

    def show_code(self) -> str:
        """Return the raw generated Python code from the last operation."""
        if not self._last_code:
            return "No code has been generated yet."
        return self._last_code

    def reset(self) -> None:
        """Clear all conversation history, cached LLM client, and model state."""
        self._memory.reset()
        self._current_model = None
        self._last_code = None
        self._last_response = None
        self._llm = None
        self._llm_api_key = None
        self._log("Session reset. Memory and cached LLM client cleared.")
