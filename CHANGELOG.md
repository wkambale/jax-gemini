# Changelog

## [0.1.3] - 2026-11-20

### Documentation and release engineering
- Applied strict editorial and typography standards across `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `docs/`, code messages, and prompt templates.
- Added MkDocs Material and `mkdocstrings` configuration (`mkdocs.yml`) with a semantic token stylesheet (`docs/stylesheets/extra.css`) and automated editorial hygiene tests (`tests/unit/test_hygiene.py`).
- Added `.github/workflows/publish.yml` for OIDC Trusted Publishing to PyPI on GitHub Releases.

## [0.1.2] - 2026-11-20

### Security and hardening
- Hardened `CodeValidator` AST checks with size and node budgets, fail-closed intent checks, relative and wildcard import blocks, banned module prefix checks, format-string traversal blocks, and bare `except:` prohibition.
- Isolated `SandboxExecutor` and `namespace.py` with `WeakKeyDictionary`-backed `_SafeModuleProxy` wrappers, frozen `SAFE_BUILTINS`, `_safe_import`, confined path resolution for `numpy`, `orbax`, and `matplotlib` (including keyword path arguments and `Figure.savefig`), and non-swallowable worker-thread trace timeouts.
- Enforced checkpoint path containment in `JaxGemini.save()` and `JaxGemini.load()`, atomic config validation and rollback in `JaxGeminiConfig.set()`, non-mutating environment API key lookup, and thread-safe per-call `GeminiLLM` API key scoping.
- Added secret and path redaction plus dynamic markdown fencing in `CodeFixer`, per-attempt `runtime_args` snapshots, bounded `ConversationMemory` and `ContextBuilder` limits, and `current_model_state` prompt injection.

### Added
- Data pipeline intents (`load_data`, `preprocess_data`, `analyze_data`, `visualize_data`).
- Word-boundary `IntentClassifier` with `JaxGemini.run()` (`jg.run()`).
- High-level pipeline wrappers in `jax_gemini.pipeline` and Flax/Optax utilities in `jax_gemini.contrib`.
- Optional `[viz]` dependency extra in `pyproject.toml`, GitHub Actions CI matrix across Python 3.10 to 3.12, and expanded unit coverage.

## [0.1.1] - 2026-04-22

### Fixed
- Hatchling wheel package discovery configuration.

## [0.1.0] - 2026-04-22

### Added
- `jg.build()`: natural language model building using Flax NNX.
- `jg.train()`: natural language training loop generation.
- AST-based code validation with import allowlist and sandboxed execution.
- 3-attempt self-healing retry loop and conversation memory for multi-turn refinement.
