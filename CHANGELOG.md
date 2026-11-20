# Changelog

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
