# Changelog

All notable releases for `jaxforge` (and prior `jax-gemini` releases) are documented here.

## Version 0.2.1 (2026-11-21)

- Hardened `_SafeModuleProxy` against transitive stdlib module escapes, confined `equinox` and `grain` file readers, restricted `safetensors` to read-only `safe_open` on `load`, and blocked `jax.profiler` (#53).
- Rejected `finally`, `async`, `except*`, non-allowlisted decorators, and definition-time calls in `CodeValidator`, and threaded `approve` callbacks through all data-pipeline methods and `run()` (#54).
- Upgraded `docker`, `e2b`, and `modal` execution backends with self-contained binary runners, `_RestrictedUnpickler` deserialization, NNX weight-leaf envelope fallback, `docker_image` configuration, `--tmpfs /tmp`, `block_network=True`, and Linux `RLIMIT_DATA` prioritization (#55).
- Isolated provider credentials across `fast_model_name` routing, enforced HTTPS on non-loopback `base_url`, moved Gemini REST auth to `x-goog-api-key`, disabled HTTP redirects, and fixed native system-prompt and JSON-schema routing across Gemini, OpenAI, Anthropic, and LiteLLM (#56).
- Hardened `export_project()` string/keyword sanitization and isolated `tool_run_sandboxed_jax` in ephemeral temporary directories with bounded timeouts and intent/approval checks (#57).
- Isolated session state across `explore()` trials, enforced diff hunk counts in `apply_unified_diff()`, bounded `DatasetProfile` sampling without exhausting iterators, and normalized `usage_summary()` keys (#58).
- Exported top-level `jf.execute_plan()`, completed `jax_gemini` subpackage and exception shims, added graceful CLI error handling, and tightened validation in `create_cosine_schedule` and `create_device_mesh` (#59).
- Pinned all GitHub Actions to 40-character commit SHAs, declared explicit CI permissions, guarded `publish-pypi` with `v*` release tag protection, and aligned all docs, notebooks, and runnable examples (#60, #61).

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
