# Jax Forge

[![PyPI version](https://img.shields.io/pypi/v/jaxforge.svg)](https://pypi.org/project/jaxforge/0.2.0/)
[![Python Versions](https://img.shields.io/pypi/pyversions/jaxforge.svg)](https://pypi.org/project/jaxforge/0.2.0/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/wkambale/jaxforge/blob/main/LICENSE)
[![CI](https://github.com/wkambale/jaxforge/actions/workflows/ci.yml/badge.svg)](https://github.com/wkambale/jaxforge/actions/workflows/ci.yml)

<div class="jg-meta">
Version <code>0.2.0</code> (<a href="https://pypi.org/project/jaxforge/0.2.0/">PyPI</a>) | Python 3.10 to 3.12 | MIT License
</div>

`jaxforge` (**Jax Forge**) turns plain English prompts into validated, inspectable, executable JAX and Flax pipelines across Gemini, OpenAI, Anthropic Claude, OpenRouter, LiteLLM, and local Ollama/vLLM models. Generated code can be previewed as diffs before running, verified against `DatasetProfile` shapes, executed inside pluggable sandboxes, explored across branching trials, and exported as standalone Python packages.

<div class="jg-grid">
  <article class="jg-card">
    <h3>Quickstart guide</h3>
    <p>Install Jax Forge, choose any LLM provider, preview plans before execution, explore experiments, and export projects.</p>
    <a class="jg-card-link" href="quickstart/">Read quickstart</a>
  </article>
  <article class="jg-card">
    <h3>API reference</h3>
    <p>Review configuration options, multi-provider adapters, preview and explore APIs, sharding, CLI, and MCP tools.</p>
    <a class="jg-card-link" href="api-reference/">Browse API reference</a>
  </article>
  <article class="jg-card">
    <h3>Architecture internals</h3>
    <p>Understand how provider routing, diff patching, AST validation, sandbox backends, and memory interact.</p>
    <a class="jg-card-link" href="architecture/">Explore architecture</a>
  </article>
  <article class="jg-card">
    <h3>Security protocol</h3>
    <p>Inspect approval gates, AST rules, import scoping, filesystem confinement, secret redaction, and compute backends.</p>
    <a class="jg-card-link" href="security/">Review security model</a>
  </article>
</div>
