# API Reference

The `jax-gemini` surface is optimized to be minimalistic while empowering high-scale deployment patterns seamlessly interacting underneath. Here are the core methods exposed for utilizing the overarching engine state.

---

### `jg.config.set(updates: dict) -> None`
Overrides configuration options securely initializing connections.

**Parameters:**
- `updates` *(dict)*: Dictionary mapping override configuration elements.
  - `gemini_api_key` *(str)*: Direct API Key setting (Default: None — resolves os variables).
  - `model_name` *(str)*: Target AI identifier (Default: `gemini-3.1-pro`).
  - `temperature` *(float)*: Deterministic variance control (Default: `0.2`).
  - `max_retries` *(int)*: Loop attempts upon code failure checks (Default: `3`).
  - `verbose` *(bool)*: Prints internal system execution states (Default: `False`).

---

### `jg.build(prompt: str) -> Any`
Primary mechanism resolving unstructured natural language concepts translating directly to executable model classes. Function maintains the global Context Session Memory caching changes.

**Parameters:**
- `prompt` *(str)*: Natural language descriptive instruction block dictating architecture.

**Returns:**
- Generated `flax.nnx.Module` object reflecting instantiation requested in prompt parameters.

---

### `jg.modify(prompt: str) -> Any`
Alters pre-existing internal active `model` parameters. It preserves execution histories and appends modifier steps updating representations in LLM conversational turns.

**Parameters:**
- `prompt` *(str)*: Refinement syntax command describing changes mapping across previous nodes.

**Returns:**
- Mutated fresh `flax.nnx.Module` object replacement.

---

### `jg.train(prompt: str, dataset: tuple = None) -> tuple[Any, dict]`
Translates human inputs generating encapsulated Optax training epochs safely returning back results.

**Parameters:**
- `prompt` *(str)*: Natural language detailing hyperparameters `optim... (i.e. 'adam')`, `loss...`, epochs.
- `dataset` *(tuple)* Numpy Array structured tuples representing dataset splits: `(X_train, y_train)`.

**Returns:**
- Generates a Tuple element formatted natively spanning `(trained_model, {"accuracy": value, "loss": value})`.

---

### `jg.evaluate(prompt: str, dataset: tuple = None) -> dict`
Passes evaluations measuring targets against model configurations accurately resolving outputs directly.

**Parameters:**
- `prompt` *(str)*: Metric evaluations descriptions.
- `dataset` *(tuple)*: Evaluated `(X_test, y_test)` splits.

**Returns:**
- Returns Python `dict` output tracking evaluated indices such as `{"accuracy": 0.95, ...}`.

---

### `jg.save(name: str) -> str`
Handles state representation abstractions safely triggering Orbax `PyTreeCheckpointer` executions persisting paths securely.

**Parameters:**
- `name` *(str)*: Checkpoint relative output name (appends to root config settings `./jax_gemini_checkpoints`).

**Returns:**
- Serialized path string resolution output dictating active storage directories.

---

### `jg.reset() -> None`
Wipes out conversational memory stores explicitly unmounting previous historical architecture components avoiding bleeding states across subsequent test attempts.
