# API reference

`jax-gemini` exposes top-level session functions (`jg.build`, `jg.train`, and peers backed by a default `JaxGemini` instance) for interactive use, as well as the `JaxGemini` class for isolated multi-session or programmatic usage.

---

### `jg.config.set(updates: dict) -> None`

Atomically validates and applies configuration updates on `JaxGeminiConfig`. Invalid keys or out-of-range values roll back cleanly and raise `JaxGeminiConfigError`.

**Fields:**
- `gemini_api_key` *(str | None)*: Gemini API key (Default: `None`, reads `GEMINI_API_KEY` from environment on demand).
- `model_name` *(str)*: Gemini model identifier (Default: `"gemini-3.1-pro"`).
- `temperature` *(float)*: Sampling temperature in `[0.0, 1.0]` (Default: `0.2`).
- `max_output_tokens` *(int)*: Maximum output tokens in `[1, 65536]` (Default: `4096`).
- `max_retries` *(int)*: Generation and execution retry attempts in `[1, 5]` (Default: `3`).
- `verbose` *(bool)*: Print internal execution logs (Default: `False`).
- `checkpoint_dir` *(str)*: Base directory for confined checkpoint persistence (Default: `"./jax_gemini_checkpoints"`).
- `timeout_seconds` *(int)*: Per-step LLM request and sandbox execution timeout in `[1, 3600]` (Default: `60`).

---

### `jg.run(prompt: str, dataset: Any = None, name: str | None = None) -> Any`

Classifies `prompt` with `IntentClassifier` and routes it automatically to `build`/`modify`, `train`, `evaluate`, `save`, `load`, `load_data`, `preprocess_data`, `analyze_data`, or `visualize_data`.

---

### `jg.build(prompt: str) -> Any`

Generates, validates, and executes a `flax.nnx.Module` from a natural-language architecture description and sets it as the active session model.

---

### `jg.modify(prompt: str) -> Any`

Refines the current model based on a natural-language instruction while preserving session history and active model context.

---

### `jg.train(prompt: str, dataset: tuple | None = None) -> tuple[Any, dict]`

Generates and runs an Optax training loop over `dataset`, returning `(trained_model, metrics_dict)`.

---

### `jg.evaluate(prompt: str, dataset: tuple | None = None) -> dict`

Evaluates the active model on `(X_test, y_test)` and returns a metrics dictionary.

---

### `jg.save(name: str) -> str` and `jg.load(name: str) -> Any`

Saves or restores the active model checkpoint under a relative `name` strictly contained inside `config.checkpoint_dir`.

---

### Data pipeline methods

- `jg.load_data(prompt: str) -> Any`: Load or synthesize a dataset in memory.
- `jg.preprocess_data(prompt: str, dataset: Any) -> Any`: Transform or normalize an existing dataset.
- `jg.analyze_data(prompt: str, dataset: Any) -> Any`: Compute dataset statistics or summaries.
- `jg.visualize_data(prompt: str, dataset: Any) -> Any`: Generate plots or figures (requires `jax-gemini[viz]`).

---

### Inspection and session control

- `jg.explain() -> str`: Return Gemini's plain-English description of the last generated code.
- `jg.show_code() -> str`: Return the validated Python source string from the last operation.
- `jg.reset() -> None`: Clear conversation memory, cached LLM client, last code or response metadata, and active model state.

---

### Pipeline and contrib helpers

- `jax_gemini.pipeline`: `ModelBuilder`, `ModelTrainer`, `ModelEvaluator`, `ModelExporter`.
- `jax_gemini.contrib`: `count_parameters`, `summarize_module`, `available_optimizers`, `create_optimizer`, `count_opt_steps`.
