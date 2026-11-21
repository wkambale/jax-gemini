# Security protocol and hardening procedures

Translating natural language into executable Python and JAX code requires layered static checks, runtime confinement, and optional process or VM isolation. `jaxforge` (Jax Forge) enforces defense-in-depth checks across parsing, AST validation, sandbox execution, checkpoints, retry prompts, and preview approval gates.

## Validation tiers

### Inspect-first approval gates
Users and automated policies can inspect every generated code artifact, `unified_diff`, provider metadata, and validation state via `preview()`, `dry_run=True`, or `approve=` callbacks (`auto_approve=False`) before any code runs across model, training, evaluation, and data-pipeline methods.

### Code parsing and AST auditor
Before execution, `CodeParser` verifies that the LLM response intent matches the requested operation, and `CodeValidator` parses the source into an AST and enforces structure and limits:

- **Input, control-flow, and resource limits**: Rejects unknown intents (fail-closed) and enforces `MAX_CODE_BYTES`, `MAX_CODE_LINES`, and `MAX_AST_NODES`, while rejecting top-level side effects outside imports, function or class definitions, and call-free constant assignments. Blocks relative and wildcard `*` imports, `finally` blocks, `async` functions/loops/with blocks, `except*`, non-allowlisted decorators, and calls in default arguments or class definitions.
- **Intent-scoped import allowlist**:
  - Core allowed root modules across all intents: `jax`, `flax`, `optax`, `numpy`, `functools`, `math`, `typing`, `dataclasses`, `equinox`, `grain`.
  - Checkpoint intents (`save`, `load`): additionally allow `orbax` (`save` and `load`) and `safetensors` (`load` only; `save_file` is blocked during `load`).
  - Plotting intent (`visualize_data`): additionally allow `matplotlib`.
- **Blocked submodules, names, and attributes**: Rejects dangerous subpackages (`jax.profiler`, `numpy.ctypeslib`, `numpy.f2py`, `numpy.distutils`, `numpy.testing`, `numpy.lib`, `flax.io`, `flax.training.checkpoints`, `orbax.checkpoint.test_utils`), dangerous builtins and names (`eval`, `exec`, `compile`, `input`, `__import__`, `open`, `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `type`, `BaseException`), bare `except:` handlers, unsafe dunder and reflection attributes (`__class__`, `__globals__`, `__subclasses__`, `__bases__`, `format`, `format_map`, `gi_frame`, `f_globals`), `allow_pickle=True`, and unsafe NumPy file readers (`fromfile`, `memmap`, `DataSource`).

### Runtime execution sandbox and backends
`SandboxExecutor` enforces `CodeValidator.validate()` even on direct calls and delegates execution to a configured `ExecutionBackend`:

- **Controlled builtins and import guard**: Exposes `SAFE_BUILTINS` wrapped per execution alongside a controlled `_safe_import` enforcing the intent allowlist and blocking transitive stdlib module attributes (such as `sys`, `os`, `io`, `builtins`, and `subprocess` re-exports on allowed stdlib modules).
- **Read-only module proxies and confined file I/O**: Exposes sandbox modules via immutable `_SafeModuleProxy` instances backed by `WeakKeyDictionary` state; `numpy`, `orbax.checkpoint`, `safetensors`, `equinox`, `grain`, and `matplotlib.pyplot` (including `Figure.savefig`) validate positional and keyword path arguments against `allowed_dirs` using symlink-resolving containment (`_resolve_confined_path`).
- **Execution timeouts, resource limits, and deserialization guards**: `ThreadSandboxBackend` enforces `timeout_seconds` using `sys.settrace` deadlines and thread join bounds with an unswallowable `BaseException` timeout; `SubprocessSandboxBackend` adds OS-process isolation and `resource.setrlimit` memory limits (`max_memory_mb`); `DockerSandboxBackend`, `E2BSandboxBackend`, and `ModalSandboxBackend` provide container and microVM isolation with restricted unpickling (`_RestrictedUnpickler`) on returned payloads. Caller `runtime_args` are deep-copied per attempt so failed executions do not mutate caller objects or `_current_model`.
- **Retry prompt sanitization and privacy bounds**: Redacts absolute paths, API keys, bearer headers, query credentials, and auth tokens before retry prompts, uses dynamic markdown fencing, restricts retry guidance to the target intent's allowed root modules without exposing internal AST validator implementation details, injects only structural `DatasetProfile` statistics (never raw rows by default), and bounds prompt and conversation memory sizes.

## Deployment recommendations

For multi-tenant or untrusted public web services, configure `execution_backend` to `"subprocess"`, `"docker"`, `"e2b"`, or `"modal"` with `auto_approve=False`, `max_memory_mb`, and network-egress restrictions.
