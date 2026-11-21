---
hide:
  - toc
---

<section class="jf-hero">
  <svg class="jf-hero__rings" viewBox="0 0 240 240" fill="none" aria-hidden="true">
    <circle cx="150" cy="90" r="52" stroke="rgba(251, 146, 60, 0.34)" stroke-width="1.5"/>
    <circle cx="166" cy="102" r="82" stroke="rgba(220, 231, 243, 0.22)" stroke-width="1.5"/>
    <circle cx="138" cy="78" r="110" stroke="rgba(220, 231, 243, 0.12)" stroke-width="1.5"/>
  </svg>
  <div class="jf-hero__eyebrow">
    <span class="jf-hero__eyebrow-line" aria-hidden="true"></span>
    <span>Python SDK version 0.2.1</span>
  </div>
  <h1 class="jf-hero__title">Jax Forge for model-agnostic JAX engineering</h1>
  <p class="jf-hero__lead">Design, preview, validate, train, explore, and export Flax NNX and Equinox pipelines from natural language. Route across Gemini, OpenAI, Anthropic, OpenRouter, LiteLLM, or local Ollama endpoints with strict AST and sandbox guarantees.</p>
  <div class="jf-hero__actions">
    <a class="jf-btn-primary" href="quickstart/">Get started</a>
    <a class="jf-btn-secondary" href="providers/">Configure providers</a>
  </div>
  <svg class="jf-hero__chevron" viewBox="0 0 1200 18" preserveAspectRatio="none" aria-hidden="true">
    <defs>
      <pattern id="jf-chevron-pattern" width="36" height="18" patternUnits="userSpaceOnUse">
        <path d="M0 18 L18 4 L36 18" fill="none" stroke="rgba(251, 146, 60, 0.32)" stroke-width="1.5"/>
      </pattern>
    </defs>
    <rect width="1200" height="18" fill="url(#jf-chevron-pattern)"/>
  </svg>
</section>

<div class="jf-badges">
  <a href="https://pypi.org/project/jaxforge/"><img src="https://img.shields.io/pypi/v/jaxforge" alt="PyPI version"></a>
  <a href="https://pypi.org/project/jaxforge/"><img src="https://img.shields.io/pypi/pyversions/jaxforge" alt="Python Versions"></a>
  <a href="https://github.com/wkambale/jaxforge/blob/main/LICENSE"><img src="https://img.shields.io/github/license/wkambale/jaxforge" alt="License"></a>
  <a href="https://github.com/wkambale/jaxforge/actions/workflows/ci.yml"><img src="https://github.com/wkambale/jaxforge/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</div>

## Core capabilities

Jax Forge turns natural language specifications into inspectable JAX artifacts across every stage of experimentation and production packaging.

<div class="jf-card-grid">
  <article class="jf-card">
    <p class="jf-card__eyebrow">Workflow</p>
    <div class="jf-card__body">
      <h3 class="jf-card__title">Inspect and preview</h3>
      <p class="jf-card__text">Generate AST-checked plans, unified diffs, and DatasetProfile summaries with interactive approval callbacks before any compute runs.</p>
      <a class="jf-card__link" href="preview-and-diffs/">Explore preview and diffs</a>
    </div>
  </article>
  <article class="jf-card">
    <p class="jf-card__eyebrow">Providers</p>
    <div class="jf-card__body">
      <h3 class="jf-card__title">Multi-provider routing</h3>
      <p class="jf-card__text">Route primary generation and fast repair retries across Gemini, OpenAI, Anthropic Claude, OpenRouter, LiteLLM, or Ollama.</p>
      <a class="jf-card__link" href="providers/">Explore LLM providers</a>
    </div>
  </article>
  <article class="jf-card">
    <p class="jf-card__eyebrow">Experimentation</p>
    <div class="jf-card__body">
      <h3 class="jf-card__title">Branching exploration</h3>
      <p class="jf-card__text">Search candidate optimization strategies against validation metrics, evaluate shapes cleanly, and configure multi-device sharding.</p>
      <a class="jf-card__link" href="training-and-exploration/">Explore training and search</a>
    </div>
  </article>
  <article class="jf-card">
    <p class="jf-card__eyebrow">Isolation and shipping</p>
    <div class="jf-card__body">
      <h3 class="jf-card__title">Sandboxes and export</h3>
      <p class="jf-card__text">Execute inside thread, subprocess, Docker, E2B, or Modal backends and export standalone installable Python packages.</p>
      <a class="jf-card__link" href="export-and-checkpoints/">Explore export and backends</a>
    </div>
  </article>
</div>

<div class="jf-band jf-band--sand" markdown="1">

<div class="jf-band__eyebrow">
  <span class="jf-band__eyebrow-line" aria-hidden="true"></span>
  <span>Quickstart</span>
</div>

## Install and preview a model in ten lines

Install `jaxforge` from PyPI and preview validated Flax NNX source code before executing a single tensor operation:

```bash
pip install jaxforge
```

```python
import jaxforge as jf

jf.config.set({"model_name": "anthropic:claude-sonnet-4-5"})

plan = jf.preview("Build a 4-layer MLP for 28x28 digit classification")
print(plan.code)

model = jf.execute_plan(plan)
jf.export_project("./exported_classifier", package_name="digit_classifier")
```

</div>

<div class="jf-spotlight-grid">
  <div class="jf-spotlight jf-spotlight--accent">
    <h3>Sync, async, CLI, and MCP parity</h3>
    <p>Switch cleanly between Python sessions, <code>abuild</code>/<code>atrain</code> async wrappers, the <code>jaxforge</code> terminal CLI, and the MCP server with full token and latency accounting via <code>usage_summary()</code>.</p>
  </div>
  <div class="jf-spotlight jf-spotlight--brand">
    <h3>Security by default</h3>
    <p>Every plan enforces fail-closed AST validation, intent-scoped import allowlists, <code>WeakKeyDictionary</code> module proxies, symlink-safe path confinement, secret redaction, and non-swallowable execution timeouts.</p>
  </div>
</div>

<div class="jf-band jf-band--mist" markdown="1">

<div class="jf-band__eyebrow">
  <span class="jf-band__eyebrow-line" aria-hidden="true"></span>
  <span>Documentation directory</span>
</div>

## Explore the guides

<ul class="jf-directory">
  <li><a href="quickstart/">Quickstart guide</a></li>
  <li><a href="providers/">LLM provider routing</a></li>
  <li><a href="preview-and-diffs/">Preview, approval gates, and diffs</a></li>
  <li><a href="training-and-exploration/">Training and exploration</a></li>
  <li><a href="export-and-checkpoints/">Export and checkpoints</a></li>
  <li><a href="backends/">Execution backends</a></li>
  <li><a href="sharding/">Distributed sharding</a></li>
  <li><a href="async/">Asynchronous support</a></li>
  <li><a href="cli-and-mcp/">CLI and MCP server</a></li>
  <li><a href="api-reference/">API reference</a></li>
  <li><a href="architecture/">Architecture internals</a></li>
  <li><a href="security/">Security protocol</a></li>
  <li><a href="changelog/">Changelog</a></li>
  <li><a href="contributing/">Contributing guide</a></li>
</ul>

</div>
