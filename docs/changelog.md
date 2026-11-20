# Changelog

All notable releases for `jaxforge` (and prior `jax-gemini` releases) are documented here.

## Version 0.2.0 (2026-11-20)

- Rebranded core distribution to `jaxforge` (**Jax Forge**) with backward-compatible `jax_gemini` deprecation shims.
- Added inspect-first workflow via `preview()`, `dry_run=True`, `approve=` callbacks, `GeneratedPlan` (with Jupyter HTML rendering and unified diffs), and incremental diff patching in `modify()`.
- Added multi-provider LLM routing (`jaxforge.llm`) across Google Gemini (`google-genai` SDK or HTTPS REST without deprecated imports), OpenAI, Anthropic Claude, OpenRouter, LiteLLM, Ollama/vLLM endpoints, `RouterLLM`, and token/latency telemetry (`usage_summary()`).
- Added pluggable `ExecutionBackend` sandboxes (`thread`, `subprocess`, `docker`, `e2b`, `modal`), `DatasetProfile` statistics injection, metric-guided `explore()`, standalone `export_project()`, async wrappers, `jax.sharding` helpers, `eval_shape_summary()`, cosine schedules, `CLI` (`jaxforge`), MCP server, Equinox, Grain, and confined Safetensors support.

## Version 0.1.3 (2026-11-20)

- Applied strict typography and editorial standards across documentation and messages.
- Added MkDocs Material configuration, design token stylesheet, automated editorial hygiene tests, GitHub Pages deployment, and Trusted Publishing workflow.

## Version 0.1.2 (2026-11-20)

- Hardened `CodeValidator` AST checks, `_SafeModuleProxy` isolation, confined path resolution, non-swallowable sandbox timeouts, atomic config rollback, secret redaction, and bounded memory and prompt sizes.
- Added data pipeline intents, word-boundary `IntentClassifier`, `pipeline` and `contrib` modules, and GitHub Actions CI matrix.

## Version 0.1.1 (2026-04-22)

- Fixed Hatchling wheel package discovery configuration.

## Version 0.1.0 (2026-04-22)

- Initial `build()`, `train()`, AST validation, sandboxed execution, and conversation memory.
