# Architecture internals

`jaxforge` separates multi-provider LLM orchestration, dataset profiling, diff patching, AST validation, pluggable execution sandboxes, experiment search, and standalone package export into modular subsystems.

## Core modules overview

### `llm/`
Provider-agnostic LLM layer with primary/fast model routing and token/latency telemetry (`LLMUsage`).
- **`base.py`**: `AbstractLLM` and `LLMUsage`.
- **`gemini.py`**: `GeminiLLM` supporting `google-genai` or zero-dependency HTTPS REST without deprecated `google-generativeai` imports.
- **`providers.py`**: `OpenAICompatibleLLM` (OpenAI, OpenRouter, Ollama, vLLM), `AnthropicLLM`, `LiteLLMBackend`, `RouterLLM`, and `create_llm()`.

### `prompts/` and `plan.py`
Manages intent classification, framework detection, privacy-preserving `DatasetProfile` summaries, `GeneratedPlan` previews, and bounded session context assembly.
- **`intent.py`**: Word-boundary `IntentClassifier` plus Equinox/Flax framework inference.
- **`context_builder.py`**: Enforces prompt bounds, alternates `user` and `model` turns, injects `current_model_state` and `DatasetProfile`, and enables `DIFF_MODIFY_SYSTEM_PROMPT`.

### `codegen/`
Translates model JSON payloads into verified source code.
- **`parser.py`**: Extracts `code` or applies unified diffs cleanly against `previous_code` via `diff.py`.
- **`validator.py`**: `CodeValidator` performs AST inspection, enforcing byte, line, and node budgets, intent-scoped import allowlists (including optional Equinox, Grain, and Safetensors), blocked submodules, blocked builtins, names, and reflection attributes, bare `except:` rejection, and pickle guards.
- **`fixer.py`**: `CodeFixer` redacts paths and secrets, adds dynamic markdown fencing, and appends JAX/XLA diagnostic hints (`ConcretizationTypeError`, shape mismatch, signature errors).

### `sandbox/`
Isolates execution via a pluggable `ExecutionBackend` protocol.
- **`backends.py`**: `ThreadSandboxBackend` (trace deadline + unswallowable `_SandboxTimeoutError`), `SubprocessSandboxBackend` (`subprocess` + `resource.setrlimit` memory limits), `DockerSandboxBackend`, `E2BSandboxBackend`, and `ModalSandboxBackend`.
- **`namespace.py`**: Builds per-execution `WeakKeyDictionary`-backed `_SafeModuleProxy` wrappers with confined path checks across `numpy`, `orbax`, `safetensors`, and `matplotlib`.

### `contrib/`, `pipeline/`, `exporter.py`, `cli.py`, and `mcp_server.py`
- **`contrib/`**: Flax `eval_shape_summary` and parameter counters, Optax optimizers/cosine schedules, `jax.sharding` mesh helpers, and `MetricsTracker` implementations.
- **`exporter.py`**: Generates clean standalone packages (`src/`, `pyproject.toml`, smoke tests).
- **`cli.py` and `mcp_server.py`**: Terminal CLI and MCP tool server for external IDE agents.
