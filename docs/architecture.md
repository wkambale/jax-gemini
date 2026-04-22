# Architecture Internals

`jax-gemini` features advanced sub-module structuring ensuring scalability separating AI interpretation handling layers, Python syntax compilation parsing, code evaluation metrics, and protected sandboxing procedures.

## Core Modules Overview

### `llm/`
Operatively orchestrating standardizations ensuring prompt integrations hit specific API pipelines formatting `application/json` payloads enforcing deterministic code abstractions explicitly bypassing generic outputs.

### `codegen/`
Pivotal translation abstraction tier:
- **`parser.py`**: Isolates `code` parameters strictly inside AI payload structured JSON returns.
- **`validator.py`**: Evaluates parsing nodes translating syntax abstract trees `(AST)`. Runs rigorous blocklists halting generic inputs processing.
- **`fixer.py`**: Interacts mapping Traceback logic converting runtime exceptions returning formats interpretable to Gemini automatically invoking self-healing iterations upon generated code blocks.

### `sandbox/`
The defensive barrier operating the execution loops.
- **`executor.py`**: Integrates mapped builtins encapsulating custom scripts protecting raw filesystem executions. Maps outputs instantiating modules returning natively to Python runtime configurations.
- **`namespace.py`**: Houses whitelist definitions configuring acceptable isolated module dictionaries strictly allowing execution integrations limiting modules natively enabling `["jax", "flax", "numpy", "optax"]`.

### `memory/`
Manages context conversational histories spanning outputs capturing changes enforcing sequential logic tracking LLM context constraints and tracking iterations.
