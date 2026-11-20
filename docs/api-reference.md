# API Reference

The `jax-gemini` package exposes both top-level session helpers (`jg.build`, `jg.train`, etc.) backed by a default `JaxGemini` instance for interactive usage, and the `JaxGemini` class for isolated multi-instance or programmatic workflows.

---

### `jg.config.set(updates: dict) -> None`
Atomically validates and updates configuration fields on `JaxGeminiConfig`.

**Fields:**
- `gemini_api_key` *(str | None)*: Gemini API key (Default: `None`, resolves `GEMINI_API_KEY` from environment).
- `model_name` *(str)*: Gemini model identifier (Default: `"gemini-3.1-pro"`).
- `temperature` *(float)*: Sampling temperature in `[0.0, 1.0]` (Default: `0.2`).
- `max_output_tokens` *(int)*: Maximum output tokens in `[1, 65536]` (Default: `4096`).
- `max_retries` *(int)*: Generation/execution retry attempts in `[1, 5]` (Default: `3`).
- `verbose` *(bool)*: Print execution logs (Default: `False`).
- `checkpoint_dir` *(str)*: Base directory for confined checkpoint persistence (Default: `"./jax_gemini_checkpoints"`).
- `timeout_seconds` *(int)*: Per-step LLM & sandbox execution timeout in `[1, 3600]` (Default: `60`).

---

### `jg.run(prompt: str, dataset: Any = None, name: str | None = None) -> Any`
Automatically classifies `prompt` with `IntentClassifier` and routes it to the matching `JaxGemini` operation (`build`/`modify`, `train`, `evaluate`, `save`, `load`, `load_data`, `preprocess_data`, `analyze_data`, or `visualize_data`).

---

### `jg.build(prompt: str) -> Any`
Generates and executes a `flax.nnx.Module` from a natural-language architecture description and sets it as the active session model.

---

### `jg.modify(prompt: str) -> Any`
Refines the current model based on a natural-language instruction while preserving session history and current model state context.

---

### `jg.train(prompt: str, dataset: tuple | None = None) -> tuple[Any, dict]`
Trains the current model with an Optax loop and returns `(trained_model, metrics_dict)`.

---

### `jg.evaluate(prompt: str, dataset: tuple | None = None) -> dict`
Evaluates the active model on `(X_test, y_test)` and returns a metrics dictionary.

---

### `jg.save(name: str) -> str` / `jg.load(name: str) -> Any`
Saves or loads the active model checkpoint under a relative `name` strictly contained within `config.checkpoint_dir`.

---

### Data Pipeline Methods
- `jg.load_data(prompt: str) -> Any`: Load or synthesize a dataset in memory.
- `jg.preprocess_data(prompt: str, dataset: Any) -> Any`: Preprocess or transform a dataset.
- `jg.analyze_data(prompt: str, dataset: Any) -> Any`: Compute dataset statistics or summaries.
- `jg.visualize_data(prompt: str, dataset: Any) -> Any`: Generate plots/figures (requires `jax-gemini[viz]`).

---

### Inspection & Session Control
- `jg.explain() -> str`: Return Gemini's plain-English description of the last operation.
- `jg.show_code() -> str`: Return the validated Python code from the last operation.
- `jg.reset() -> None`: Clear conversation memory, cached LLM client, last code/response, and active model state.

---

### Pipeline & Contrib Helpers
- `jax_gemini.pipeline`: `ModelBuilder`, `ModelTrainer`, `ModelEvaluator`, `ModelExporter`.
- `jax_gemini.contrib`: `count_parameters`, `summarize_module`, `available_optimizers`, `create_optimizer`, `count_opt_steps`.
