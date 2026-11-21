# Execution backends

Jax Forge separates static AST auditing from runtime isolation using the `ExecutionBackend` protocol. Switch backends with `execution_backend` in `JaxForgeConfig`:

```python
import jaxforge as jf

jf.config.set({
    "execution_backend": "subprocess",
    "max_memory_mb": 2048,
    "timeout_seconds": 60,
})
```

## Available backends

| Backend key | Isolation boundary | Recommended environment | Key options |
| --- | --- | --- | --- |
| `thread` | In-process worker thread, `sys.settrace` deadline, read-only module proxies | Local interactive notebooks | `timeout_seconds` |
| `subprocess` | Fresh OS child process with `resource.setrlimit` memory cap | CI pipelines and local automation | `max_memory_mb`, `timeout_seconds` |
| `docker` | Ephemeral Docker container (`--network=none`, `--read-only`, `--tmpfs /tmp`, `--memory`) | Self-hosted multi-user APIs | `docker_image`, `max_memory_mb`, `timeout_seconds` |
| `e2b` | Remote cloud microVM (`e2b_code_interpreter`) | Managed SaaS sandboxes | `E2B_API_KEY`, `timeout_seconds` |
| `modal` | Ephemeral remote serverless sandbox (`modal` with `block_network=True`) | Scalable burst compute | Modal CLI auth, `timeout_seconds` |

## Custom backend implementations

Any class implementing `execute(...) -> Any` matching `ExecutionBackend` can be injected or tested without modifying `SandboxExecutor`. Before any backend receives code, `SandboxExecutor` runs `CodeValidator.validate()` fail-closed regardless of the selected backend.
