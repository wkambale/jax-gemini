from typing import Any
import os
import jax.numpy as jnp
import numpy as np

from jax_gemini.config import JaxGeminiConfig
from jax_gemini.exceptions import JaxGeminiExecutionError, JaxGeminiValidationError
from jax_gemini.llm.gemini import GeminiLLM
from jax_gemini.prompts.context_builder import ContextBuilder
from jax_gemini.prompts.intent import IntentClassifier
from jax_gemini.codegen.parser import CodeParser
from jax_gemini.codegen.validator import CodeValidator
from jax_gemini.codegen.fixer import CodeFixer
from jax_gemini.sandbox.executor import SandboxExecutor
from jax_gemini.memory.conversation import ConversationMemory
from jax_gemini.pipeline import builder, trainer, evaluator, exporter


class JaxGemini:

    def __init__(self, config: JaxGeminiConfig | None = None):
        self.config = config or JaxGeminiConfig()
        self._llm: GeminiLLM | None = None
        self._memory = ConversationMemory()
        self._executor = SandboxExecutor()
        self._last_code: str | None = None
        self._last_response: dict | None = None
        self._current_model: Any = None

    def _get_llm(self) -> GeminiLLM:
        if self._llm is None:
            self.config.validate()
            self._llm = GeminiLLM(self.config)
        return self._llm

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[jax-gemini] {message}")

    def _generate_and_execute(
        self,
        user_prompt: str,
        intent: str,
        runtime_args: dict | None = None,
    ) -> Any:
        llm = self._get_llm()
        context = ContextBuilder.build(
            intent=intent,
            user_prompt=user_prompt,
            memory=self._memory,
            current_model_state=self._memory.get_current_model_state(),
        )
        schema = ContextBuilder.get_schema(intent)
        last_error: Exception | None = None
        last_code: str = ""

        for attempt in range(1, self.config.max_retries + 1):
            self._log(f"Attempt {attempt}/{self.config.max_retries} for intent '{intent}'...")

            if attempt > 1 and last_code and last_error:
                fix_prompt = CodeFixer.build_fix_prompt(
                    original_prompt=user_prompt,
                    failed_code=last_code,
                    error_message=str(last_error),
                    attempt=attempt,
                )
                context = ContextBuilder.build(
                    intent=intent,
                    user_prompt=fix_prompt,
                    memory=self._memory,
                    current_model_state=self._memory.get_current_model_state(),
                )

            try:
                response = llm.generate(context, schema)
                code = CodeParser.extract(response, intent)
                CodeValidator.validate(code, intent)
                result = self._executor.run(code, intent, runtime_args)

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

            except (JaxGeminiValidationError, JaxGeminiExecutionError) as e:
                last_error = e
                last_code = getattr(e, "code", "")
                self._log(f"Attempt {attempt} failed: {e}")
                if attempt < self.config.max_retries:
                    self._log("Sending error context to Gemini for correction...")

        raise JaxGeminiExecutionError(
            f"All {self.config.max_retries} attempts failed for '{user_prompt}'.\n"
            f"Last error: {last_error}",
            code=last_code,
            attempts=self.config.max_retries,
        )

    def build(self, prompt: str) -> Any:
        """Build a JAX/Flax model from a natural language description."""
        model = self._generate_and_execute(prompt, intent="build")
        self._current_model = model
        return model

    def modify(self, prompt: str) -> Any:
        """Refine the current model based on a natural language instruction."""
        model = self._generate_and_execute(prompt, intent="build")
        self._current_model = model
        return model

    def train(self, prompt: str, dataset: tuple | None = None) -> tuple[Any, dict]:
        """Train the current model. Returns (model, metrics_dict)."""
        args = {"dataset": dataset} if dataset else {}
        if self._current_model is not None:
             args["model"] = self._current_model
        result = self._generate_and_execute(prompt, intent="train", runtime_args=args)
        if isinstance(result, tuple):
            self._current_model = result[0]
            return result
        return result, {}

    def evaluate(self, prompt: str, dataset: tuple | None = None) -> dict:
        """Evaluate the current model. Returns a metrics dict."""
        args = {"dataset": dataset} if dataset else {}
        if self._current_model is not None:
             args["model"] = self._current_model
        return self._generate_and_execute(prompt, intent="evaluate", runtime_args=args)

    def save(self, name: str) -> str:
        """Save the current model checkpoint. Returns the checkpoint path."""
        path = os.path.join(self.config.checkpoint_dir, name)
        return self._generate_and_execute(
            f"Save the model to {path}", intent="save",
            runtime_args={"model": self._current_model, "path": path}
        )

    def load(self, name: str) -> Any:
        """Load a model checkpoint by name."""
        path = os.path.join(self.config.checkpoint_dir, name)
        model = self._generate_and_execute(
            f"Load the model from {path}", intent="load",
            runtime_args={"path": path}
        )
        self._current_model = model
        return model

    def explain(self) -> str:
        """Return a plain-English explanation of the last generated code."""
        if not self._last_code:
            return "No code has been generated yet. Call jg.build() first."
        if self._last_response and "description" in self._last_response:
            return self._last_response["description"]
        return "No explanation available for the last operation."

    def show_code(self) -> str:
        """Return the raw generated Python code from the last operation."""
        if not self._last_code:
            return "No code has been generated yet."
        return self._last_code

    def reset(self) -> None:
        """Clear all conversation history and model state."""
        self._memory.reset()
        self._current_model = None
        self._last_code = None
        self._last_response = None
        self._log("Session reset. Memory cleared.")
