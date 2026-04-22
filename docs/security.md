# Security
`jax-gemini` never executes unvalidated code. All code runs through an AST check enforcing strict rules on built-in uses (no eval/exec/open) and a whitelist for modules (only JAX, Flax, Optax, etc are allowed). Execution uses `exec` into an isolated namespace.
