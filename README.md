<div align="center">
  <h1>Jax Forge</h1>
  <p><b>Model-agnostic natural language engineering workbench for JAX and Flax</b></p>

  [![PyPI version](https://img.shields.io/pypi/v/jaxforge)](https://pypi.org/project/jaxforge/)
  [![Python Versions](https://img.shields.io/pypi/pyversions/jaxforge)](https://pypi.org/project/jaxforge/)
  [![License](https://img.shields.io/github/license/wkambale/jaxforge)](LICENSE)
  [![CI](https://github.com/wkambale/jaxforge/actions/workflows/ci.yml/badge.svg)](https://github.com/wkambale/jaxforge/actions/workflows/ci.yml)
</div>

`jaxforge` (product name **Jax Forge**, formerly `jax-gemini`) enables machine learning engineers and researchers to preview, validate, build, train, explore, and export JAX and Flax workflows from natural language across any frontier or local LLM provider.

Unlike blind code-execution wrappers, `jaxforge` supports an inspect-first gate (`preview()`, `dry_run=True`, or `approve=` callbacks), privacy-preserving `DatasetProfile` summaries, unified diff edits, metric-guided tree-search exploration (`explore()`), isolated execution backends, and standalone Python package exports.

## Core features

- **Inspect-first preview and approval**: Generate and AST-validate code with `jf.preview()` or `dry_run=True` to inspect source code, unified diffs, and validation metadata before running compute, and verify abstract tensor shapes with `eval_shape_summary()`.
- **Multi-provider LLM routing**: Use Google Gemini, OpenAI, Anthropic Claude, OpenRouter, LiteLLM, or local Ollama and vLLM servers through `provider` or `model_name` prefixes (such as `anthropic:claude-sonnet-4-5`, `openai:gpt-5`, or `ollama:qwen2.5-coder:32b`) with fast/primary routing.
- **Production JAX workflows**: Supports Flax NNX and Equinox modeling, Optax schedules, `jax.sharding` device meshes, Grain streaming datasets, Orbax and Safetensors checkpoints, and `jax.eval_shape` checks.
- **Pluggable sandbox isolation**: Enforces AST allowlists, `WeakKeyDictionary` module proxies, and confined file I/O across `thread`, `subprocess` (with memory/CPU limits), `docker`, `e2b`, and `modal` backends.
- **Branching exploration and project export**: Compare candidate strategies with `jf.explore()` and materialize clean standalone Python packages via `jf.export_project()`, the `jaxforge` CLI, or the MCP server.

## Quickstart

### Installation

Install `jaxforge` from [PyPI](https://pypi.org/project/jaxforge/):

```bash
pip install jaxforge
```

OpenAI, Anthropic, OpenRouter, Ollama, and the Docker backend work out of the box using Python's standard library and the `docker` CLI. Optional extras install additional SDKs, modeling frameworks, cloud sandboxes, or plotting:

```bash
pip install "jaxforge[google,litellm,viz]"
pip install "jaxforge[equinox,grain,safetensors]"
pip install "jaxforge[e2b,modal,mcp]"
```

Configure an API key for your chosen provider (or point `base_url` at a local Ollama or vLLM endpoint):

```bash
export GEMINI_API_KEY="your-key"      # or OPENAI_API_KEY / ANTHROPIC_API_KEY
```

### Preview, approve, train, and export

```python
import jaxforge as jf
import numpy as np

# Switch providers freely: "gemini-3.1-pro", "openai:gpt-5", "anthropic:claude-sonnet-4-5", "ollama:llama3"
jf.config.set({"model_name": "anthropic:claude-sonnet-4-5"})

# Preview code and validation status before running anything
plan = jf.preview("Build a 4-layer MLP for 28x28 digit classification")
print(plan.code)

# Execute the previewed plan directly, or use build/modify/train with approval gates
model = jf.execute_plan(plan)
model = jf.modify("Add dropout with rate 0.2 between hidden layers")

X_train = np.random.randn(100, 28, 28, 1).astype(np.float32)
y_train = np.random.randint(0, 10, size=(100,))

# Explore multiple candidate training runs against validation accuracy
experiment = jf.explore(
    "Train classifier with cosine learning-rate decay",
    dataset=(X_train, y_train),
    metric="accuracy",
    candidates=3,
)
print("Best accuracy:", experiment.best_candidate.score)

# Export generated modules into a standalone installable repository
jf.export_project("./exported_classifier", package_name="digit_classifier")
```

### CLI and MCP server

```bash
jaxforge info
jaxforge devices
jaxforge preview "Build a Flax NNX residual classifier" --export ./out_pkg
jaxforge mcp
```

See [examples/](examples/) for end-to-end Python scripts and notebooks.

## Documentation

Full hosted documentation is published at [wkambale.github.io/jaxforge](https://wkambale.github.io/jaxforge/):

- [Quickstart guide](docs/quickstart.md): Installation, provider configuration, preview gates, training, and export.
- [LLM providers](docs/providers.md): Google Gemini, OpenAI, Anthropic Claude, OpenRouter, Ollama, and routing.
- [Preview and diffs](docs/preview-and-diffs.md): Inspect-first execution, `approve` callbacks, diffs, and `DatasetProfile`.
- [Training and exploration](docs/training-and-exploration.md): Single-shot training, `explore()` trial search, and shape checks.
- [Export and checkpoints](docs/export-and-checkpoints.md): Standalone project export, Orbax persistence, and Safetensors loading.
- [Execution backends](docs/backends.md): Thread, subprocess resource limits, Docker, E2B, and Modal isolation.
- [Distributed sharding](docs/sharding.md): Device discovery, meshes, and `NamedSharding` rules.
- [Asynchronous support](docs/async.md): Non-blocking session wrappers (`apreview`, `abuild`, `atrain`, `aevaluate`).
- [CLI and MCP server](docs/cli-and-mcp.md): Terminal commands and Model Context Protocol tool integration.
- [API reference](docs/api-reference.md): Public functions, `GeneratedPlan`, execution backends, and contrib helpers.
- [Architecture internals](docs/architecture.md): Multi-provider routing, AST validation, sandbox backends, and memory.
- [Security protocol](docs/security.md): AST rules, import scoping, runtime proxies, path confinement, and timeout enforcement.

## License and contributing

Read the [contribution guidelines](CONTRIBUTING.md) before opening issues or pull requests. Distributed under the MIT License; see [LICENSE](LICENSE) for full terms.
