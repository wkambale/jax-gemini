# Architecture internals

`jax-gemini` separates prompt orchestration, AST safety validation, isolated runtime execution, conversation memory, and composable ML pipeline helpers into dedicated modules.

## Core modules overview

### `llm/`
Handles Gemini API integration and structured JSON schema generation (`application/json`) with thread-safe credential scoping and configured request timeouts.
- **`base.py`**: Abstract LLM client interface (`AbstractLLM`).
- **`gemini.py`**: `GeminiLLM` implementation calling Google Gemini with `request_options={"timeout": ...}` and guarded response extraction.

### `prompts/`
Manages intent classification, system prompt templates, and bounded session context assembly.
- **`intent.py`**: Word-boundary rule-based `IntentClassifier` routing natural-language prompts (`jg.run()`).
- **`context_builder.py`**: Enforces prompt length bounds, preserves alternating `user` and `model` turn structure, bounds conversation history, and injects `current_model_state`.
- **`system_prompt.py`**: Task-specific instructions for model building, training, evaluation, checkpointing, and data pipeline intents.

### `codegen/`
Translates model JSON payloads into verified Python source code.
- **`parser.py`**: Extracts `code` from structured JSON payloads and rejects intent mismatches.
- **`validator.py`**: `CodeValidator` performs AST inspection, enforcing byte, line, and node budgets, intent-scoped import allowlists, blocked submodules, blocked builtins, names, and attributes, bare `except:` rejection, pickle guards, and top-level side-effect restrictions.
- **`fixer.py`**: `CodeFixer` redacts filesystem paths and credentials and uses dynamic markdown fencing in self-healing retry prompts.

### `sandbox/`
Isolates generated code execution from the host environment.
- **`executor.py`**: `SandboxExecutor` re-validates code, runs it in a dedicated daemon thread under trace deadlines and join timeouts (`config.timeout_seconds`), and raises unswallowable timeout errors.
- **`namespace.py`**: Builds per-execution `MappingProxyType`-protected builtins, controlled `_safe_import`, and `WeakKeyDictionary`-backed `_SafeModuleProxy` wrappers with confined path checks across `numpy`, `orbax`, and `matplotlib`.

### `memory/`
- **`conversation.py`**: `ConversationMemory` stores bounded turn history and `ModelState` snapshots.

### `pipeline/` and `contrib/`
- **`pipeline/`**: High-level composable helpers (`ModelBuilder`, `ModelTrainer`, `ModelEvaluator`, `ModelExporter`).
- **`contrib/`**: Flax NNX and Optax utilities (`count_parameters`, `summarize_module`, `available_optimizers`, `create_optimizer`, `count_opt_steps`).
