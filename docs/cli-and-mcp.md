# CLI and MCP server

Jax Forge ships both a command-line interface (`jaxforge`) and a Model Context Protocol (MCP) stdio server (`jaxforge.mcp_server`) so external coding agents and CI workflows can validate, execute, and export JAX code safely.

## Command-line interface

Inspect available commands or run preview and export workflows from your shell:

```bash
jaxforge info
jaxforge devices
jaxforge validate path/to/model.py --intent build
jaxforge preview "Build a 3-layer Flax NNX classifier" --export ./exported_pkg
jaxforge run "Build a 3-layer Flax NNX classifier" --yes
jaxforge mcp
```

### Subcommand reference

| Subcommand | Purpose | Key flags |
| --- | --- | --- |
| `jaxforge info` | Print package version, supported providers, and execution backends | None |
| `jaxforge devices` | Emit discovered JAX accelerators as JSON | None |
| `jaxforge validate <file>` | AST-validate an existing Python source file | `--intent` |
| `jaxforge preview <prompt>` | Generate and AST-check code without execution (exports standalone packages for `build` prompts) | `--model`, `--export` |
| `jaxforge run <prompt>` | Preview, confirm, and execute a prompt inside the configured sandbox | `--model`, `--yes` |
| `jaxforge mcp` | Start the MCP stdio tool server | None |

All CLI subcommands catch validation, configuration, I/O, and interactive cancellation errors cleanly, writing concise diagnostics to `stderr` with exit status `1`.

## Model Context Protocol server

`jaxforge.mcp_server` exposes three read/execution tools over stdio JSON-RPC when `mcp` extra dependencies are installed (`pip install "jaxforge[mcp]"`):

- **`tool_validate_jax_code(code, intent="build")`**: Runs `CodeValidator.validate()` and returns `{valid, error}`.
- **`tool_device_summary()`**: Returns discovered JAX devices and platform metadata.
- **`tool_run_sandboxed_jax(code, intent="analyze_data", timeout_seconds=15.0)`**: Validates and executes code inside an isolated temporary directory (`0 < timeout_seconds <= 30.0`, non-persistent intents only, respecting `auto_approve`) without host working-directory or shell access.

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
