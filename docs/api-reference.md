# API reference

`jaxforge` exposes top-level session functions (`jf.preview`, `jf.execute_plan`, `jf.build`, `jf.train`, `jf.explore`, `jf.export_project`) backed by a default `JaxForge` instance, plus the `JaxForge` class for isolated multi-session or server workflows. Backward-compatible `jax_gemini` imports emit a `DeprecationWarning` and alias directly to `JaxForge`.

---

### `jf.config.set(updates: dict) -> None`

Atomically validates and applies updates on `JaxForgeConfig`.

**Fields:**
- `provider` *(str)*: `"auto"`, `"google"`, `"openai"`, `"anthropic"`, `"ollama"`, `"openrouter"`, or `"litellm"` (Default: `"auto"`).
- `model_name` *(str)*: Target model or `provider:model` prefix spec (Default: `"gemini-3.1-pro"`).
- `fast_model_name` *(str | None)*: Optional faster model used for retry repairs (Default: `None`).
- `api_key` / `gemini_api_key` *(str | None)*: API key (reads matching env var when unset).
- `base_url` *(str | None)*: Optional HTTP(S) base URL for Ollama, vLLM, or custom proxies.
- `framework` *(str)*: `"flax"`, `"equinox"`, or `"auto"` (Default: `"flax"`).
- `execution_backend` *(str)*: `"thread"`, `"subprocess"`, `"docker"`, `"e2b"`, or `"modal"` (Default: `"thread"`).
- `docker_image` *(str)*: Container image tag used when `execution_backend="docker"` (Default: `"jaxforge-sandbox:latest"`).
- `max_memory_mb` *(int | None)*: Memory limit for `subprocess` and `docker` backends (Default: `None`).
- `auto_approve` *(bool)*: Require explicit `approve=` callback when `False` (Default: `True`).
- `temperature` *(float)*: Sampling temperature in `[0.0, 1.0]` (Default: `0.2`).
- `max_output_tokens` *(int)*: Maximum output tokens in `[1, 65536]` (Default: `4096`).
- `max_retries` *(int)*: Generation retry attempts in `[1, 5]` (Default: `3`).
- `timeout_seconds` *(int)*: Request and sandbox timeout in `[1, 3600]` (Default: `60`).
- `checkpoint_dir` *(str)*: Confined base directory for checkpoints (Default: `"./jaxforge_checkpoints"`).
- `verbose` *(bool)*: Print execution logs (Default: `False`).

---

### `jf.preview(prompt: str, intent: str | None = None, dataset: Any = None, diff_mode: bool = False) -> GeneratedPlan`

Generates and AST-validates code without executing it, returning a `GeneratedPlan` with optional `unified_diff` text and Jupyter HTML rendering. Use `jf.execute_plan(plan, ...)` (or `agent.execute_plan(plan, ...)` on an explicit session) to execute after approval.

---

### `jf.build` / `jf.modify` / `jf.train` / `jf.evaluate` / `jf.run`

All primary execution methods accept `dry_run: bool = False` and `approve: Callable[[GeneratedPlan], bool] | None = None` for inspect-first gates, and async variants `abuild`, `amodify`, `atrain`, `aevaluate`, and `apreview` on `JaxForge`.

---

### `jf.explore(prompt: str, dataset: Any, metric: str = "accuracy", maximize: bool = True, candidates: int = 3) -> ExperimentResult`

Runs bounded branching trials across optimization strategies and promotes the highest-scoring candidate model.

---

### `jf.export_project(output_dir: str | Path, package_name: str = "generated_model") -> Path`

Writes generated `model.py`, optional `train.py` and `eval.py`, `pyproject.toml`, and `tests/test_smoke.py` into `output_dir`.

---

### Data pipeline, checkpoints, telemetry, and inspection

- `jf.save(name)` and `jf.load(name)`: Confined Orbax/Safetensors persistence inside `checkpoint_dir`.
- `jf.load_data()`, `jf.preprocess_data()`, `jf.analyze_data()`, `jf.visualize_data()`: Confined dataset operations profiled automatically by `DatasetProfile`.
- `jf.explain()`, `jf.show_code()`, `jf.usage_summary()`, `jf.reset()`: Inspection, token/latency accounting, and session reset.

---

### CLI, MCP, pipeline, and contrib helpers

- CLI entrypoint: `jaxforge {info,devices,validate,preview,run,mcp}`.
- `jaxforge.mcp_server`: `tool_validate_jax_code`, `tool_device_summary`, `tool_run_sandboxed_jax`, and `serve_stdio()`.
- `jaxforge.pipeline`: `ModelBuilder`, `ModelTrainer`, `ModelEvaluator`, `ModelExporter`.
- `jaxforge.contrib`: `count_parameters`, `summarize_module`, `eval_shape_summary`, `available_optimizers`, `create_optimizer`, `create_cosine_schedule`, `count_opt_steps`, `device_summary`, `create_device_mesh`, `make_replicated_sharding`, `make_data_sharding`, `InMemoryTracker`, `JsonlTracker`, `create_tracker`.
