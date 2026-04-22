# jax-gemini Index Map

Welcome to the `jax-gemini` documentation! 

`jax-gemini` allows executing and dynamically instantiating robust deep-learning pipelines using JAX and Flax strictly out of conversational natural language instructions. Powered by Gemini, the package orchestrates automatic AST compilation handling code execution inside isolated runtime namespaces.

## Table of Contents

- **[Quickstart](quickstart.md)** — Start utilizing the primary pipeline building, refining, evaluating, and checkpoint saving paradigms.
- **[API Reference](api-reference.md)** — Detailed explanations traversing methods encompassing `build`, `train`, `modify`, `evaluate`, and config setups.
- **[Architecture](architecture.md)** — Insight into how Code Generators, AST Validators, Sandbox Executors, Intent Classifications and the Retry Controller merge together.
- **[Security Protocol](security.md)** — Unpacking what safeguards `jax-gemini` implements out-of-the-box avoiding malignant prompt injections into arbitrary runtime code execution.
