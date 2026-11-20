# Security Protocol & Hardening Procedures

Translating unconstrained natural language into executable Python/JAX code requires layered validation and runtime controls. `jax-gemini` enforces defense-in-depth checks across parsing, AST validation, sandbox execution, checkpoints, and retry prompts.

## Validation Tiers

### 1. Code Parsing & Abstract Syntax Tree (AST) Auditor
Before any generated code is executed, `CodeParser` verifies that the LLM response intent matches the requested operation, and `CodeValidator` parses the source into an AST and validates structure and limits:

- **Input & Resource Limits**: Rejects unknown intents (fail-closed) and enforces `MAX_CODE_BYTES`, `MAX_CODE_LINES`, and `MAX_AST_NODES`, while rejecting top-level side effects outside `import` / `def` / `class` / simple constant assignments and blocking relative imports.
- **Intent-Scoped Import Allowlist**:
  - Core allowed root modules across all intents: `jax`, `flax`, `optax`, `numpy`, `functools`, `typing`, `math`, `dataclasses`.
  - Checkpoint intents (`save`, `load`): additionally allow `orbax`.
  - Plotting intent (`visualize_data`): additionally allow `matplotlib`.
- **Blocked Names & Attributes**: Rejects dangerous builtins and references (e.g., `eval`, `exec`, `compile`, `input`, `__import__`, `open`, `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `type`, `object`), unsafe dunder attributes (e.g., `__class__`, `__globals__`, `__subclasses__`, `__bases__`, `gi_frame`, `f_globals`), `allow_pickle=True`, and unsafe NumPy file-reader calls (`fromfile`, `loadtxt`, `genfromtxt`, etc.).

### 2. Runtime Execution Sandbox
`SandboxExecutor` enforces `CodeValidator.validate()` even on direct calls and executes validated code inside a fresh isolated namespace (`namespace.py`):

- **Controlled Builtins & Import Guard**: Exposes `SAFE_BUILTINS` (excluding dynamic introspection/code-execution builtins such as `eval`, `exec`, `getattr`, `type`, and `open`) wrapped per execution in a read-only `MappingProxyType` alongside a controlled `_safe_import` enforcing the intent's import allowlist.
- **Read-Only Module Proxies & Confined File I/O**: Exposes sandbox modules via immutable `_SafeModuleProxy` wrappers so code cannot monkeypatch shared modules; `numpy`, `orbax.checkpoint`, and `matplotlib.pyplot` file APIs validate targets against `allowed_dirs` using symlink-resolving path containment (`_resolve_confined_path`).
- **Execution Timeouts & Rollback**: Bounds sandbox runtime via `config.timeout_seconds` (POSIX `signal.setitimer` on the main thread and `sys.settrace` deadline checks in worker threads) and snapshots caller `runtime_args` per retry attempt so failed executions do not mutate caller objects or `_current_model`.
- **Retry Prompt Sanitization & Bounds**: Redacts absolute paths and secrets from execution errors before sending fix prompts, uses dynamic markdown fencing, omits internal allowlist details from retry feedback, and caps prompt and conversation memory sizes.

## Deployment Recommendations
When embedding `jax-gemini` in multi-tenant or public services, isolate workloads in dedicated OS/container sandboxes (such as Docker/gVisor with seccomp, cgroup CPU/RAM limits, and egress controls) in addition to the in-process AST and namespace protections above.
