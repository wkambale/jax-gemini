# Security protocol and hardening procedures

Translating unconstrained natural language into executable Python and JAX code requires layered validation and runtime controls. `jax-gemini` enforces defense-in-depth checks across parsing, AST validation, sandbox execution, checkpoints, and retry prompts.

## Validation tiers

### Code parsing and AST auditor
Before any generated code is executed, `CodeParser` verifies that the LLM response intent matches the requested operation, and `CodeValidator` parses the source into an AST and validates structure and limits:

- **Input and resource limits**: Rejects unknown intents (fail-closed) and enforces `MAX_CODE_BYTES`, `MAX_CODE_LINES`, and `MAX_AST_NODES`, while rejecting top-level side effects outside imports, function or class definitions, and call-free constant assignments, and blocking relative and wildcard `*` imports.
- **Intent-scoped import allowlist**:
  - Core allowed root modules across all intents: `jax`, `flax`, `optax`, `numpy`, `functools`, `typing`, `math`, `dataclasses`.
  - Checkpoint intents (`save`, `load`): additionally allow `orbax`.
  - Plotting intent (`visualize_data`): additionally allow `matplotlib`.
- **Blocked submodules, names, and attributes**: Rejects dangerous subpackages (`numpy.ctypeslib`, `numpy.f2py`, `numpy.distutils`, `numpy.testing`, `numpy.lib`, `flax.io`, `flax.training.checkpoints`, `orbax.checkpoint.test_utils`), dangerous builtins and names (`eval`, `exec`, `compile`, `input`, `__import__`, `open`, `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `type`, `BaseException`), bare `except:` handlers, unsafe dunder and reflection attributes (`__class__`, `__globals__`, `__subclasses__`, `__bases__`, `format`, `format_map`, `gi_frame`, `f_globals`), `allow_pickle=True`, and unsafe NumPy file readers (`fromfile`, `memmap`, `DataSource`).

### Runtime execution sandbox
`SandboxExecutor` enforces `CodeValidator.validate()` even on direct calls and executes validated code inside a fresh isolated namespace (`namespace.py`):

- **Controlled builtins and import guard**: Exposes `SAFE_BUILTINS` wrapped per execution alongside a controlled `_safe_import` enforcing the intent's import allowlist, resolving subpackages through root proxies without bypassing wrapper restrictions.
- **Read-only module proxies and confined file I/O**: Exposes sandbox modules via immutable `_SafeModuleProxy` instances backed by `WeakKeyDictionary` state so code cannot monkeypatch shared modules or read internal handles; `numpy`, `orbax.checkpoint`, and `matplotlib.pyplot` (including `Figure.savefig`) validate positional and keyword path arguments against `allowed_dirs` using symlink-resolving containment (`_resolve_confined_path`).
- **Execution timeouts and rollback**: Bounds sandbox runtime in a daemon worker thread via `sys.settrace` deadlines and thread join timeouts (`config.timeout_seconds`) using an internal `BaseException` subclass so sandboxed `except Exception:` blocks cannot swallow timeouts; snapshots caller `runtime_args` per retry attempt so failed executions do not mutate caller objects or `_current_model`.
- **Retry prompt sanitization and bounds**: Redacts absolute paths, API keys, and auth tokens from errors before sending fix prompts, uses dynamic markdown fencing, omits internal allowlist details from retry feedback, and caps prompt and conversation memory sizes.

## Deployment recommendations

When embedding `jax-gemini` in multi-tenant or public services, isolate workloads in dedicated OS or container sandboxes (such as Docker or gVisor with seccomp, cgroup CPU and RAM limits, and network egress controls) in addition to the in-process AST and namespace protections above.
