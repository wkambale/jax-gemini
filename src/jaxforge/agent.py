"""Main JaxForge session orchestrator supporting preview, multi-provider routing, and explore."""

from __future__ import annotations

import asyncio
import copy
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import flax.nnx as nnx

from jaxforge.codegen.diff import build_unified_diff
from jaxforge.codegen.fixer import CodeFixer
from jaxforge.codegen.parser import CodeParser
from jaxforge.codegen.validator import CodeValidator
from jaxforge.config import JaxForgeConfig
from jaxforge.contrib.tracking import InMemoryTracker, MetricsTracker
from jaxforge.exceptions import (
    JaxForgeApprovalError,
    JaxForgeCheckpointError,
    JaxForgeError,
    JaxForgeExecutionError,
    JaxForgeIntentError,
    JaxForgeLLMError,
    JaxForgeValidationError,
)
from jaxforge.exporter import export_project as write_project_package
from jaxforge.llm.providers import RouterLLM, create_llm
from jaxforge.memory.conversation import ConversationMemory
from jaxforge.plan import (
    DatasetProfile,
    ExperimentCandidate,
    ExperimentResult,
    GeneratedPlan,
)
from jaxforge.prompts.context_builder import MAX_CONTEXT_PROMPT_CHARS, ContextBuilder
from jaxforge.prompts.intent import IntentClassifier
from jaxforge.sandbox.executor import SandboxExecutor

ApprovalCallback = Callable[[GeneratedPlan], bool]


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


def _sanitize_validation_feedback(err: JaxForgeValidationError) -> str:
    """Avoid exposing internal AST security policy details as an interactive bypass oracle."""
    msg = str(err)
    if "syntax error" in msg.lower() or "must define a top-level function" in msg.lower():
        return msg
    return "Generated code violated sandbox validation rules; use only allowed ML operations."


class JaxForge:
    """Natural language-driven, inspect-first JAX engineering workbench."""

    def __init__(
        self,
        config: JaxForgeConfig | None = None,
        tracker: MetricsTracker | None = None,
    ):
        self.config = config or JaxForgeConfig()
        self._llm: RouterLLM | None = None
        self._llm_cache_key: tuple[str, str, str | None, str | None, str | None] | None = None
        self._memory = ConversationMemory()
        self._executor = SandboxExecutor()
        self._tracker: MetricsTracker = tracker or InMemoryTracker()
        self._last_code: str | None = None
        self._last_build_code: str | None = None
        self._last_train_code: str | None = None
        self._last_eval_code: str | None = None
        self._last_response: dict[str, Any] | None = None
        self._current_model: Any = None

    def _get_llm(self) -> RouterLLM:
        self.config.validate()
        effective_key = self.config.get_effective_api_key()
        cache_key = (
            self.config.resolve_provider(),
            self.config.resolve_model_id(),
            self.config.fast_model_name,
            self.config.base_url,
            effective_key,
        )
        if self._llm is None or self._llm_cache_key != cache_key:
            primary = create_llm(self.config)
            fast = (
                create_llm(self.config, model_override=self.config.fast_model_name)
                if self.config.fast_model_name
                else None
            )
            self._llm = RouterLLM(primary=primary, fast=fast)
            self._llm_cache_key = cache_key
        return self._llm

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[jaxforge] {message}")

    def _allowed_dirs(self, intent: str) -> tuple[Path, ...]:
        ckpt_dir = Path(self.config.checkpoint_dir)
        if intent in ("save", "load"):
            return (ckpt_dir,)
        return (Path.cwd(), ckpt_dir)

    def _record_success(
        self,
        intent: str,
        user_prompt: str,
        code: str,
        response: dict[str, Any],
    ) -> None:
        self._last_code = code
        self._last_response = response
        if intent == "build":
            self._last_build_code = code
        elif intent == "train":
            self._last_train_code = code
        elif intent == "evaluate":
            self._last_eval_code = code
        self._memory.add_turn("user", user_prompt)
        self._memory.add_turn("model", str(response.get("description", code)))
        if intent in ("build", "train"):
            self._memory.add_model_state(
                code=code,
                description=str(response.get("description", "")),
                intent=intent,
            )

    def preview(
        self,
        user_prompt: str,
        intent: str | None = None,
        dataset: Any = None,
        diff_mode: bool = False,
    ) -> GeneratedPlan:
        """Generate and validate code cleanly without executing it in the sandbox."""
        ContextBuilder.validate_prompt(user_prompt)
        resolved_intent = intent or IntentClassifier.classify(user_prompt)
        framework = IntentClassifier.detect_framework(user_prompt, default=self.config.framework)
        profile = DatasetProfile.from_dataset(dataset) if dataset is not None else None
        prev_state = self._memory.get_current_model_state()
        allow_diff = diff_mode and resolved_intent == "build" and prev_state is not None

        llm = self._get_llm()
        context = ContextBuilder.build(
            intent=resolved_intent,
            user_prompt=user_prompt,
            memory=self._memory,
            current_model_state=prev_state,
            dataset_profile=profile,
            framework=framework,
            diff_mode=allow_diff,
        )
        schema = ContextBuilder.get_schema(resolved_intent, allow_diff=allow_diff)
        response = llm.generate(context, schema)
        previous_code = prev_state.code if (allow_diff and prev_state is not None) else None
        code = CodeParser.extract(response, resolved_intent, previous_code=previous_code)
        CodeValidator.validate(code, resolved_intent)
        diff_text = (
            build_unified_diff(prev_state.code, code)
            if (prev_state is not None and resolved_intent == "build")
            else ""
        )
        raw_warnings = response.get("warnings") or []
        warnings = [str(w) for w in raw_warnings] if isinstance(raw_warnings, list) else []
        return GeneratedPlan(
            intent=resolved_intent,
            code=code,
            description=str(response.get("description", "")),
            provider=self.config.resolve_provider(),
            model_name=self.config.resolve_model_id(),
            framework=framework,
            validation_passed=True,
            unified_diff=diff_text,
            warnings=warnings,
            metadata={"raw_response": response},
        )

    def execute_plan(
        self,
        plan: GeneratedPlan,
        runtime_args: dict[str, Any] | None = None,
        user_prompt: str = "",
        approve: ApprovalCallback | None = None,
    ) -> Any:
        """Run a previously previewed plan after checking approval policy."""
        if approve is not None:
            if not approve(plan):
                raise JaxForgeApprovalError(
                    f"Execution of intent '{plan.intent}' rejected by approval callback.",
                    code=plan.code,
                )
        elif not self.config.auto_approve:
            raise JaxForgeApprovalError(
                "auto_approve is False; pass an explicit approve callback or call execute_plan().",
                code=plan.code,
            )

        attempt_args = _snapshot_runtime_args(runtime_args)
        result = self._executor.run(
            plan.code,
            plan.intent,
            attempt_args,
            allowed_dirs=self._allowed_dirs(plan.intent),
            timeout_seconds=self.config.timeout_seconds,
            backend_name=self.config.execution_backend,
            max_memory_mb=self.config.max_memory_mb,
            docker_image=self.config.docker_image,
        )
        raw_resp = plan.metadata.get("raw_response") or {
            "intent": plan.intent,
            "code": plan.code,
            "description": plan.description,
        }
        self._record_success(
            plan.intent,
            user_prompt or plan.description or plan.intent,
            plan.code,
            raw_resp,
        )
        if plan.intent == "build":
            self._current_model = result
        elif plan.intent == "train" and isinstance(result, tuple):
            if len(result) >= 1:
                self._current_model = result[0]
            if len(result) > 1 and isinstance(result[1], dict):
                self._tracker.log_metrics(result[1])
        elif plan.intent == "evaluate" and isinstance(result, dict):
            self._tracker.log_metrics(result)
        elif plan.intent == "load":
            self._current_model = result
        return result

    def _generate_and_execute(
        self,
        user_prompt: str,
        intent: str,
        runtime_args: dict[str, Any] | None = None,
        dataset: Any = None,
        dry_run: bool = False,
        approve: ApprovalCallback | None = None,
        diff_mode: bool = False,
    ) -> Any:
        ContextBuilder.validate_prompt(user_prompt)
        llm = self._get_llm()
        framework = IntentClassifier.detect_framework(user_prompt, default=self.config.framework)
        profile = DatasetProfile.from_dataset(dataset) if dataset is not None else None
        prev_state = self._memory.get_current_model_state()
        allow_diff = diff_mode and intent == "build" and prev_state is not None

        context = ContextBuilder.build(
            intent=intent,
            user_prompt=user_prompt,
            memory=self._memory,
            current_model_state=prev_state,
            dataset_profile=profile,
            framework=framework,
            diff_mode=allow_diff,
        )
        schema = ContextBuilder.get_schema(intent, allow_diff=allow_diff)
        last_error: Exception | None = None
        last_feedback = ""
        last_code = ""

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
                    current_model_state=prev_state,
                    max_prompt_chars=MAX_CONTEXT_PROMPT_CHARS,
                    dataset_profile=profile,
                    framework=framework,
                    diff_mode=False,
                )

            try:
                response = llm.generate(context, schema, use_fast=(attempt > 1))
                previous_code = (
                    prev_state.code if (allow_diff and attempt == 1 and prev_state) else None
                )
                code = CodeParser.extract(response, intent, previous_code=previous_code)
                CodeValidator.validate(code, intent)

                plan = GeneratedPlan(
                    intent=intent,
                    code=code,
                    description=str(response.get("description", "")),
                    provider=self.config.resolve_provider(),
                    model_name=self.config.resolve_model_id(),
                    framework=framework,
                    validation_passed=True,
                    unified_diff=(
                        build_unified_diff(prev_state.code, code)
                        if (prev_state is not None and intent == "build")
                        else ""
                    ),
                    warnings=[
                        str(w) for w in (response.get("warnings") or []) if isinstance(w, str)
                    ],
                    metadata={"raw_response": response},
                )

                if dry_run:
                    return plan
                if approve is not None:
                    if not approve(plan):
                        raise JaxForgeApprovalError(
                            f"Execution of '{intent}' rejected during preview.", code=code
                        )
                elif not self.config.auto_approve:
                    raise JaxForgeApprovalError(
                        "auto_approve is False; use preview()/execute_plan() or pass approve=.",
                        code=code,
                    )

                attempt_args = _snapshot_runtime_args(runtime_args)
                result = self._executor.run(
                    code,
                    intent,
                    attempt_args,
                    allowed_dirs=self._allowed_dirs(intent),
                    timeout_seconds=self.config.timeout_seconds,
                    backend_name=self.config.execution_backend,
                    max_memory_mb=self.config.max_memory_mb,
                    docker_image=self.config.docker_image,
                )

                self._record_success(intent, user_prompt, code, response)
                self._log(f"Attempt {attempt} succeeded.")
                return result

            except JaxForgeApprovalError:
                raise

            except JaxForgeValidationError as e:
                last_error = e
                last_feedback = _sanitize_validation_feedback(e)
                last_code = getattr(e, "code", "")
                self._log(f"Attempt {attempt} failed validation: {e}")

            except JaxForgeExecutionError as e:
                last_error = e
                last_feedback = str(e)
                last_code = getattr(e, "code", "")
                self._log(f"Attempt {attempt} failed execution: {e}")

            except JaxForgeLLMError as e:
                last_error = e
                last_feedback = (
                    "Previous response was malformed or failed to parse as valid JSON; "
                    "respond strictly with the requested JSON schema."
                )
                last_code = ""
                self._log(f"Attempt {attempt} failed LLM response parsing: {e}")

        if isinstance(last_error, JaxForgeLLMError):
            raise last_error
        if isinstance(last_error, JaxForgeValidationError):
            raise last_error

        prompt_preview = user_prompt[:160] + ("..." if len(user_prompt) > 160 else "")
        raise JaxForgeExecutionError(
            f"All {self.config.max_retries} attempts failed for '{prompt_preview}'.\n"
            f"Last error: {last_error}",
            code=last_code,
            attempts=self.config.max_retries,
        )

    def build(
        self,
        prompt: str,
        dry_run: bool = False,
        approve: ApprovalCallback | None = None,
    ) -> Any:
        """Build a JAX neural network module from natural language (or return GeneratedPlan)."""
        result = self._generate_and_execute(
            prompt,
            intent="build",
            runtime_args={"rngs": nnx.Rngs(0)},
            dry_run=dry_run,
            approve=approve,
        )
        if dry_run:
            return result
        self._current_model = result
        return result

    def modify(
        self,
        prompt: str,
        dry_run: bool = False,
        approve: ApprovalCallback | None = None,
        use_diff: bool = True,
    ) -> Any:
        """Refine the current model iteratively using diff or full-code updates."""
        result = self._generate_and_execute(
            prompt,
            intent="build",
            runtime_args={"rngs": nnx.Rngs(0)},
            dry_run=dry_run,
            approve=approve,
            diff_mode=use_diff,
        )
        if dry_run:
            return result
        self._current_model = result
        return result

    def train(
        self,
        prompt: str,
        dataset: Any = None,
        dry_run: bool = False,
        approve: ApprovalCallback | None = None,
    ) -> Any:
        """Train the active model cleanly returning ``(trained_model, metrics_dict)``."""
        args: dict[str, Any] = {"dataset": dataset} if dataset is not None else {}
        if self._current_model is not None:
            args["model"] = self._current_model
        result = self._generate_and_execute(
            prompt,
            intent="train",
            runtime_args=args,
            dataset=dataset,
            dry_run=dry_run,
            approve=approve,
        )
        if dry_run:
            return result
        if isinstance(result, tuple):
            self._current_model = result[0]
            if len(result) > 1 and isinstance(result[1], dict):
                self._tracker.log_metrics(result[1])
            return cast("tuple[Any, dict[str, Any]]", result)
        return result, {}

    def evaluate(
        self,
        prompt: str,
        dataset: Any = None,
        dry_run: bool = False,
        approve: ApprovalCallback | None = None,
    ) -> Any:
        """Evaluate the current model and return a metrics dictionary."""
        args: dict[str, Any] = {"dataset": dataset} if dataset is not None else {}
        if self._current_model is not None:
            args["model"] = self._current_model
        result = self._generate_and_execute(
            prompt,
            intent="evaluate",
            runtime_args=args,
            dataset=dataset,
            dry_run=dry_run,
            approve=approve,
        )
        if not dry_run and isinstance(result, dict):
            self._tracker.log_metrics(result)
        return result

    def explore(
        self,
        prompt: str,
        dataset: Any,
        metric: str = "accuracy",
        maximize: bool = True,
        candidates: int = 3,
    ) -> ExperimentResult:
        """Run bounded metric-guided tree-search exploration across training variants."""
        if candidates < 1 or candidates > 10:
            raise ValueError("candidates must be between 1 and 10.")
        variants = (
            "Optimize for baseline clean convergence with AdamW.",
            "Explore stronger regularization and learning-rate warmup cosine decay.",
            "Try wider or deeper residual block capacity with gradient clipping.",
        )
        records: list[ExperimentCandidate] = []
        best: ExperimentCandidate | None = None
        saved_model = self._current_model
        saved_turns = list(self._memory._turns)
        saved_history = list(self._memory._model_history)
        saved_last_code = self._last_code
        saved_last_train_code = self._last_train_code
        saved_last_response = copy.deepcopy(self._last_response)

        for idx in range(candidates):
            hint = variants[idx % len(variants)]
            variant_prompt = f"{prompt} ({hint} Trial {idx + 1}/{candidates})"
            self._current_model = saved_model
            self._memory._turns = list(saved_turns)
            self._memory._model_history = list(saved_history)
            self._last_code = saved_last_code
            self._last_train_code = saved_last_train_code
            self._last_response = copy.deepcopy(saved_last_response)
            try:
                model, metrics = self.train(variant_prompt, dataset=dataset)
                raw_score = metrics.get(metric)
                score = float(raw_score) if isinstance(raw_score, (int, float)) else None
                cand = ExperimentCandidate(
                    index=idx + 1,
                    prompt_variant=hint,
                    code=self._last_code or "",
                    description=self.explain(),
                    metrics=metrics,
                    score=score,
                    model=model,
                )
            except Exception as exc:
                cand = ExperimentCandidate(
                    index=idx + 1,
                    prompt_variant=hint,
                    code=self._last_code or "",
                    description="",
                    metrics={},
                    score=None,
                    error=str(exc),
                    model=None,
                )
            finally:
                self._current_model = saved_model
            records.append(cand)
            if cand.score is not None and (
                best is None
                or best.score is None
                or (cand.score > best.score if maximize else cand.score < best.score)
            ):
                best = cand

        self._memory._turns = list(saved_turns)
        self._memory._model_history = list(saved_history)

        if best is None:
            best = records[0]
        if best.model is not None and best.code:
            self._current_model = best.model
            self._record_success(
                "train",
                prompt,
                best.code,
                {"intent": "train", "code": best.code, "description": best.description},
            )
        else:
            self._current_model = saved_model
            self._last_code = saved_last_code
            self._last_train_code = saved_last_train_code
            self._last_response = saved_last_response
        return ExperimentResult(
            best_candidate=best,
            candidates=records,
            metric=metric,
            maximize=maximize,
        )

    def _resolve_checkpoint_path(self, name: str) -> str:
        """Resolve a checkpoint name strictly inside self.config.checkpoint_dir."""
        if not isinstance(name, str):
            raise JaxForgeCheckpointError("Checkpoint name must be a string.")
        cleaned = name.strip()
        if (
            not cleaned
            or "\x00" in cleaned
            or "\n" in cleaned
            or "\r" in cleaned
            or "://" in cleaned
        ):
            raise JaxForgeCheckpointError(f"Invalid checkpoint name: '{name}'")
        raw = Path(cleaned)
        if raw.is_absolute() or os.path.isabs(cleaned) or ".." in raw.parts:
            raise JaxForgeCheckpointError(
                f"Checkpoint name '{name}' must be a relative path within checkpoint_dir."
            )
        base_dir = Path(self.config.checkpoint_dir).resolve()
        candidate = (base_dir / raw).resolve()
        if candidate == base_dir or not candidate.is_relative_to(base_dir):
            raise JaxForgeCheckpointError(
                f"Checkpoint path '{name}' escapes checkpoint_dir '{self.config.checkpoint_dir}'."
            )
        return str(candidate)

    def save(
        self, name: str, dry_run: bool = False, approve: ApprovalCallback | None = None
    ) -> Any:
        """Save the active model checkpoint under ``name`` inside checkpoint_dir."""
        path = self._resolve_checkpoint_path(name)
        result = self._generate_and_execute(
            "Save the model checkpoint to the provided path argument.",
            intent="save",
            runtime_args={"model": self._current_model, "path": path},
            dry_run=dry_run,
            approve=approve,
        )
        return result if dry_run else str(result)

    def load(
        self, name: str, dry_run: bool = False, approve: ApprovalCallback | None = None
    ) -> Any:
        """Load a model checkpoint by relative name within checkpoint_dir."""
        path = self._resolve_checkpoint_path(name)
        result = self._generate_and_execute(
            "Load the model checkpoint from the provided path argument.",
            intent="load",
            runtime_args={"path": path},
            dry_run=dry_run,
            approve=approve,
        )
        if dry_run:
            return result
        self._current_model = result
        return result

    def load_data(
        self,
        prompt: str,
        dry_run: bool = False,
        approve: ApprovalCallback | None = None,
    ) -> Any:
        """Load or generate a dataset based on natural language description."""
        return self._generate_and_execute(
            prompt, intent="load_data", dry_run=dry_run, approve=approve
        )

    def preprocess_data(
        self,
        prompt: str,
        dataset: Any,
        dry_run: bool = False,
        approve: ApprovalCallback | None = None,
    ) -> Any:
        """Preprocess a given dataset based on natural language instruction."""
        return self._generate_and_execute(
            prompt,
            intent="preprocess_data",
            runtime_args={"dataset": dataset},
            dataset=dataset,
            dry_run=dry_run,
            approve=approve,
        )

    def analyze_data(
        self,
        prompt: str,
        dataset: Any,
        dry_run: bool = False,
        approve: ApprovalCallback | None = None,
    ) -> Any:
        """Analyze a given dataset and return computed summary statistics."""
        return self._generate_and_execute(
            prompt,
            intent="analyze_data",
            runtime_args={"dataset": dataset},
            dataset=dataset,
            dry_run=dry_run,
            approve=approve,
        )

    def visualize_data(
        self,
        prompt: str,
        dataset: Any,
        dry_run: bool = False,
        approve: ApprovalCallback | None = None,
    ) -> Any:
        """Visualize a given dataset by generating confined plots."""
        return self._generate_and_execute(
            prompt,
            intent="visualize_data",
            runtime_args={"dataset": dataset},
            dataset=dataset,
            dry_run=dry_run,
            approve=approve,
        )

    def run(
        self,
        prompt: str,
        dataset: Any = None,
        name: str | None = None,
        dry_run: bool = False,
        approve: ApprovalCallback | None = None,
    ) -> Any:
        """Route a natural-language prompt automatically via IntentClassifier."""
        if dry_run:
            intent = IntentClassifier.classify(prompt)
            return self.preview(prompt, intent=intent, dataset=dataset)
        intent = IntentClassifier.classify(prompt)
        if intent == "build":
            return (
                self.modify(prompt, approve=approve)
                if self._current_model is not None
                else self.build(prompt, approve=approve)
            )
        if intent == "train":
            return self.train(prompt, dataset=dataset, approve=approve)
        if intent == "evaluate":
            return self.evaluate(prompt, dataset=dataset, approve=approve)
        if intent == "save":
            if not name:
                raise JaxForgeCheckpointError("Saving a model requires a checkpoint 'name'.")
            return self.save(name, approve=approve)
        if intent == "load":
            if not name:
                raise JaxForgeCheckpointError("Loading a model requires a checkpoint 'name'.")
            return self.load(name, approve=approve)
        if intent == "load_data":
            return self.load_data(prompt, approve=approve)
        if dataset is None:
            raise JaxForgeIntentError(f"Intent '{intent}' requires a 'dataset' argument.")
        if intent == "preprocess_data":
            return self.preprocess_data(prompt, dataset, approve=approve)
        if intent == "analyze_data":
            return self.analyze_data(prompt, dataset, approve=approve)
        return self.visualize_data(prompt, dataset, approve=approve)

    async def apreview(
        self, prompt: str, intent: str | None = None, dataset: Any = None
    ) -> GeneratedPlan:
        """Asynchronous wrapper for ``preview()``."""
        return await asyncio.to_thread(self.preview, prompt, intent, dataset)

    async def abuild(
        self, prompt: str, dry_run: bool = False, approve: ApprovalCallback | None = None
    ) -> Any:
        """Asynchronous wrapper for ``build()``."""
        return await asyncio.to_thread(self.build, prompt, dry_run, approve)

    async def amodify(
        self, prompt: str, dry_run: bool = False, approve: ApprovalCallback | None = None
    ) -> Any:
        """Asynchronous wrapper for ``modify()``."""
        return await asyncio.to_thread(self.modify, prompt, dry_run, approve)

    async def atrain(
        self,
        prompt: str,
        dataset: Any = None,
        dry_run: bool = False,
        approve: ApprovalCallback | None = None,
    ) -> Any:
        """Asynchronous wrapper for ``train()``."""
        return await asyncio.to_thread(self.train, prompt, dataset, dry_run, approve)

    async def aevaluate(
        self,
        prompt: str,
        dataset: Any = None,
        dry_run: bool = False,
        approve: ApprovalCallback | None = None,
    ) -> Any:
        """Asynchronous wrapper for ``evaluate()``."""
        return await asyncio.to_thread(self.evaluate, prompt, dataset, dry_run, approve)

    def export_project(
        self,
        output_dir: str | Path,
        package_name: str = "generated_model",
    ) -> Path:
        """Export the last built, training, and evaluation code as a standalone Python package."""
        model_code = self._last_build_code
        if not model_code:
            raise JaxForgeError("No generated model code available to export. Call build() first.")
        desc = (
            str(self._last_response["description"])
            if self._last_response and "description" in self._last_response
            else "Standalone JAX model exported by Jax Forge."
        )
        return write_project_package(
            output_dir=output_dir,
            model_code=model_code,
            train_code=self._last_train_code,
            eval_code=self._last_eval_code,
            package_name=package_name,
            description=desc,
            framework=self.config.framework,
        )

    def usage_summary(self) -> dict[str, Any]:
        """Return aggregated token and latency telemetry for LLM calls in this session."""
        if self._llm is None:
            return {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_latency_ms": 0.0,
            }
        hist = self._llm.usage_history
        return {
            "calls": len(hist),
            "prompt_tokens": sum(u.prompt_tokens for u in hist),
            "completion_tokens": sum(u.completion_tokens for u in hist),
            "total_tokens": sum(u.total_tokens for u in hist),
            "total_latency_ms": round(sum(u.latency_ms for u in hist), 2),
        }

    def explain(self) -> str:
        """Return a plain-English explanation of the last generated code."""
        if not self._last_code:
            return "No code has been generated yet. Call jf.build() first."
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
        self._last_build_code = None
        self._last_train_code = None
        self._last_eval_code = None
        self._last_response = None
        self._llm = None
        self._llm_cache_key = None
        self._log("Session reset. Memory and cached LLM client cleared.")


JaxGemini = JaxForge
