# CLI and MCP server

Jax Forge ships both a command-line interface (`jaxforge`) and a Model Context Protocol (MCP) stdio server (`jaxforge.mcp_server`) so external coding agents and CI workflows can validate, execute, and export JAX code safely.

## Command-line interface

Inspect available commands or run preview and export workflows from your shell:

```bash
jaxforge info
jaxforge devices
jaxforge validate path/to/model.py --intent build
jaxforge preview "Build a 3-layer Flax NNX classifier" --export ./exported_pkg
jaxforge run "Build a 3-layer Flax NNX classifier"
jaxforge mcp
```

### Subcommand reference

| Subcommand | Purpose | Key flags |
| --- | --- | --- |
| `jaxforge info` | Print active configuration, provider, and backend | `--model`, `--backend` |
| `jaxforge devices` | Emit discovered JAX accelerators as JSON | None |
| `jaxforge validate <file>` | AST-validate an existing Python source file | `--intent` |
| `jaxforge preview <prompt>` | Generate and AST-check code without execution | `--model`, `--export` |
| `jaxforge run <prompt>` | Generate, validate, and execute code in sandbox | `--model`, `--backend`, `--export` |
| `jaxforge mcp` | Start the MCP stdio tool server | None |

## Model Context Protocol server

`jaxforge.mcp_server` exposes three read/execution tools over stdio JSON-RPC when `mcp` extra dependencies are installed (`pip install "jaxforge[mcp]"`):

- **`tool_validate_jax_code(code, intent="build")`**: Runs `CodeValidator.validate()` and returns `{valid, error}`.
- **`tool_device_summary()`**: Returns discovered JAX devices and platform metadata.
- **`tool_run_sandboxed_jax(code, intent="build", timeout_seconds=30)`**: Validates and runs generated code inside the configured sandbox without host shell access.

Register Jax Forge in any MCP client configuration:

```json
{
  "mcpServers": {
    "jaxforge": {
      "command": "jaxforge",
      "args": ["mcp"]
    }
  }
}
```
