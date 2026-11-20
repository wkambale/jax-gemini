# Architecture Internals

`jax-gemini` separates prompt orchestration, AST safety validation, isolated runtime execution, conversation memory, and composable ML pipeline helpers into distinct modules.

## Core Modules Overview

### `llm/`
Handles Gemini API integration and structured JSON schema generation (`application/json`) with thread-safe credential scoping and configured timeouts.
- **`base.py`**: Abstract LLM client interface (`AbstractLLM`).
- **`gemini.py`**: `GeminiLLM` implementation calling Google Gemini with `request_options={"timeout": ...}`.

### `prompts/`
Manages intent classification, system prompt templates, and session context assembly.
- **`intent.py`**: Word-boundary rule-based `IntentClassifier` routing natural-language prompts (`jg.run()`).
- **`context_builder.py`**: Enforces prompt size limits, preserves alternating `user`/`model` turn structure, bounds conversation history, and injects `current_model_state`.
- **`system_prompt.py`**: Task-specific instructions for model building, training, evaluation, checkpointing, and data pipeline intents.

### `codegen/`
Translates model JSON payloads into verified Python source code.
- **`parser.py`**: Extracts `code` from structured JSON payloads and rejects intent mismatches.
- **`validator.py`**: `CodeValidator` performs AST inspection, enforcing byte/line/node budgets, intent-scoped import allowlists, blocked builtins/names/attributes, pickle guards, and top-level side-effect restrictions.
- **`fixer.py`**: `CodeFixer` redacts filesystem paths and sensitive error details and uses dynamic markdown fencing in self-healing retry prompts.

### `sandbox/`
Isolates generated code execution from the host environment.
- **`executor.py`**: `SandboxExecutor` re-validates code, runs it inside an isolated execution namespace, and enforces `config.timeout_seconds` deadlines.
- **`namespace.py`**: Builds per-execution `MappingProxyType`-protected builtins, controlled `_safe_import`, and read-only `_SafeModuleProxy` wrappers with confined path checks across `numpy`, `orbax`, and `matplotlib`.

### `memory/`
- **`conversation.py`**: `ConversationMemory` stores bounded turn history and `ModelState` snapshots.

### `pipeline/` & `contrib/`
- **`pipeline/`**: High-level composable helpers (`ModelBuilder`, `ModelTrainer`, `ModelEvaluator`, `ModelExporter`).
- **`contrib/`**: Flax NNX and Optax utilities (`count_parameters`, `summarize_module`, `available_optimizers`, `create_optimizer`, `count_opt_steps`).
