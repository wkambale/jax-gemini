# Changelog

## [Unreleased]

### Security & Hardening
- Hardened `CodeValidator` AST checks (size/AST node budgets, fail-closed unknown intents, relative import blocking, banned name/attribute/dunder allowlist checks, top-level side-effect restrictions, `allow_pickle=True` blocking, and intent-scoped import allowlists).
- Isolated `SandboxExecutor` and `namespace.py` with read-only `_SafeModuleProxy` wrappers, frozen `SAFE_BUILTINS`, `_safe_import`, confined path resolution for `numpy`, `orbax`, and `matplotlib`, and `timeout_seconds` enforcement.
- Enforced checkpoint path containment in `JaxGemini.save()`/`load()`, strict config field validation with atomic rollback in `JaxGeminiConfig.set()`, and thread-safe per-call `GeminiLLM` API key scoping.
- Redacted paths and fenced retry prompts in `CodeFixer`, snapshotted `runtime_args` per retry attempt, bounded `ConversationMemory`/`ContextBuilder`, and wired `current_model_state`.

### Added
- Word-boundary `IntentClassifier` with `JaxGemini.run()` (`jg.run()`).
- High-level pipeline helpers in `jax_gemini.pipeline` and Flax/Optax utilities in `jax_gemini.contrib`.
- Optional `[viz]` dependency extra in `pyproject.toml` and GitHub Actions CI workflow (`.github/workflows/ci.yml`).

## [0.1.2] - 2026-04-23

### Added
- Data pipeline intents (`load_data`, `preprocess_data`, `analyze_data`, `visualize_data`).

## [0.1.1] - 2026-04-22

### Fixed
- Hatchling wheel package discovery configuration.

## [0.1.0] - 2026-04-22

### Added
- `jg.build()` — natural language model building using Flax NNX
- `jg.train()` — natural language training loop generation
- AST-based code validation with import whitelist
- Sandboxed execution environment
- 3-attempt self-healing retry loop
- Conversation memory for multi-turn refinement
- PyPI package: `pip install jax-gemini`
