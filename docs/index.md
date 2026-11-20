# Jax Gemini

[![PyPI version](https://img.shields.io/pypi/v/jax-gemini.svg)](https://pypi.org/project/jax-gemini/0.1.3/)
[![Python Versions](https://img.shields.io/pypi/pyversions/jax-gemini.svg)](https://pypi.org/project/jax-gemini/0.1.3/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/wkambale/jax-gemini/blob/main/LICENSE)
[![CI](https://github.com/wkambale/jax-gemini/actions/workflows/ci.yml/badge.svg)](https://github.com/wkambale/jax-gemini/actions/workflows/ci.yml)

<div class="jg-meta">
Version <code>0.1.3</code> (<a href="https://pypi.org/project/jax-gemini/0.1.3/">PyPI</a>) | Python 3.10 to 3.12 | MIT License
</div>

`jax-gemini` turns plain English prompts into validated, executable JAX and Flax pipelines powered by Google Gemini. Generated code is checked by an AST auditor, executed inside a restricted runtime namespace, and returned directly as live Python and `flax.nnx.Module` objects.

<div class="jg-grid">
  <article class="jg-card">
    <h3>Quickstart guide</h3>
    <p>Install the package, configure your Gemini API key, and run build, training, evaluation, and checkpoint workflows.</p>
    <a class="jg-card-link" href="quickstart/">Read quickstart</a>
  </article>
  <article class="jg-card">
    <h3>API reference</h3>
    <p>Review configuration options, top-level functions, pipeline stage wrappers, and Flax or Optax helpers.</p>
    <a class="jg-card-link" href="api-reference/">Browse API reference</a>
  </article>
  <article class="jg-card">
    <h3>Architecture internals</h3>
    <p>Understand how prompt building, AST validation, sandbox proxies, and bounded conversation memory connect.</p>
    <a class="jg-card-link" href="architecture/">Explore architecture</a>
  </article>
  <article class="jg-card">
    <h3>Security protocol</h3>
    <p>Inspect the AST rules, import scoping, filesystem confinement, secret redaction, and timeout mechanisms.</p>
    <a class="jg-card-link" href="security/">Review security model</a>
  </article>
</div>
