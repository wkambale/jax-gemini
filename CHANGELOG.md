# Changelog

## [0.2.1] - 2026-11-21

### Security and sandbox hardening
- Blocked transitive stdlib module attributes (such as `sys`, `os`, `io`, `builtins`, and `subprocess` re-exports) inside `_SafeModuleProxy.__getattr__`, enforced confined file paths across `equinox` and `grain` I/O helpers, restricted `safetensors` to `safe_open` during `load`, and blocked `jax.profiler` (#53).
- Hardened `CodeValidator` to reject `finally` blocks, `async` constructs, `except*`, non-allowlisted decorators, and definition-time calls, and threaded `approve` callbacks through `load_data`, `preprocess_data`, `analyze_data`, `visualize_data`, and `JaxForge.run()` (#54).
- Rebuilt `DockerSandboxBackend`, `E2BSandboxBackend`, and `ModalSandboxBackend` around self-contained base64-framed binary runners with `_RestrictedUnpickler`, NNX leaf-envelope fallback, confined container mounts, `--tmpfs /tmp`, `block_network=True`, and Linux `RLIMIT_DATA` memory limits (#55).
- Isolated `api_key` and `base_url` across multi-provider `fast_model_name` routing, enforced HTTPS on non-loopback `base_url`, moved Gemini REST authentication to the `x-goog-api-key` header, blocked automatic HTTP redirects, fixed native system-prompt and JSON-schema routing across Gemini, OpenAI, Anthropic, and LiteLLM, and expanded secret redaction in `CodeFixer` (#56).
- Escaped `description` strings and rejected Python keyword `package_name` values in `export_project()`, and isolated `tool_run_sandboxed_jax` inside ephemeral temporary directories with bounded timeouts (`<= 30s`), intent allowlisting, and `auto_approve` enforcement (#57).
- Pinned all GitHub Actions in `ci.yml`, `docs.yml`, and `publish.yml` to 40-character commit SHAs, declared explicit `permissions: contents: read` in `ci.yml`, pinned release build tooling, and guarded `publish-pypi` with both `if: github.event_name == 'release'` and a `v*` tag policy on the `pypi` GitHub Environment (#60).

### Fixed and improved
- Preserved and promoted winning candidate state cleanly across `explore()` trials, enforced hunk line counts and single-file headers in `apply_unified_diff()`, bounded `DatasetProfile` sampling without consuming `Iterator` streams, and normalized `usage_summary()` keys (#58).
- Exported `execute_plan` at package root (`jf.execute_plan`), added `JaxGeminiApprovalError` and full `jax_gemini` subpackage deprecation shims, handled CLI errors gracefully without raw tracebacks, and added strict input validation in `create_cosine_schedule` and `create_device_mesh` (#59).
- Aligned all documentation pages, `README.md`, Jupyter notebooks, and runnable example scripts (`05_custom_training_loop.py`, `06_data_pipeline.py`) with current signatures and runtime behavior (#61).

## [0.2.0] - 2026-11-20

### Added
- Rebranded core distribution to `jaxforge` (**Jax Forge**) with backward-compatible `jax_gemini` compatibility shims.
- Inspect-first workflow via `preview()`, `dry_run=True`, `approve=` callbacks, `GeneratedPlan` (including Jupyter HTML rendering and unified diffs), and incremental diff patching in `modify()`.
- Multi-provider LLM engine under `jaxforge.llm` supporting Google Gemini (`google-genai` SDK or HTTPS REST without deprecated `google-generativeai` warnings), OpenAI, Anthropic Claude, OpenRouter, LiteLLM, Ollama/vLLM local endpoints, primary/fast `RouterLLM`, and token/latency telemetry (`usage_summary()`).
- Pluggable sandbox `ExecutionBackend` implementations (`thread`, `subprocess` with `rlimit`, `docker`, `e2b`, `modal`).
- Privacy-preserving `DatasetProfile` injection, AIDE-inspired metric-guided `explore()`, `export_project()` standalone package export, async helpers (`abuild`, `atrain`, `aevaluate`, `apreview`), `jax.sharding` mesh/sharding utilities, `eval_shape_summary()`, cosine learning-rate schedules, experiment metric trackers, CLI (`jaxforge`), and MCP server (`jaxforge.mcp_server`).
- Support for optional Equinox modeling, Grain data loaders, and confined Safetensors weight loading.

## [0.1.3] - 2026-11-20

### Documentation and release engineering
- Applied strict editorial and typography standards across documentation, code messages, and prompt templates.
- Added MkDocs Material configuration (`mkdocs.yml`), token stylesheet (`docs/stylesheets/extra.css`), automated editorial hygiene tests, GitHub Pages deployment, and `publish.yml`.

## [0.1.2] - 2026-11-20

### Security and hardening
- Hardened `CodeValidator` AST checks, `_SafeModuleProxy` isolation, confined path resolution, non-swallowable sandbox timeouts, atomic config rollback, secret redaction, and bounded memory and prompt sizes.
- Added data pipeline intents, word-boundary `IntentClassifier`, `pipeline` and `contrib` modules, and GitHub Actions CI matrix.

## [0.1.1] - 2026-04-22

### Fixed
- Hatchling wheel package discovery configuration.

## [0.1.0] - 2026-04-22

### Added
- Initial `build()`, `train()`, AST validation, sandboxed execution, and conversation memory.
