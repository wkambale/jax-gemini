# AGENTS.md — jax-gemini Build Specification

> **This file is the authoritative instruction set for AI agents implementing jax-gemini.**
> Read every section before writing a single line of code. Do not skip sections.
> Follow every rule exactly. When in doubt, ask before assuming.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack & Dependencies](#2-tech-stack--dependencies)
3. [Repository Structure](#3-repository-structure)
4. [Implementation Phases](#4-implementation-phases)
5. [Module Specifications](#5-module-specifications)
6. [Code Style & Quality](#6-code-style--quality)
7. [Testing Requirements](#7-testing-requirements)
8. [Security Rules](#8-security-rules)
9. [Prompt Engineering](#9-prompt-engineering)
10. [PyPI Packaging](#10-pypi-packaging)
11. [CI/CD Pipeline](#11-cicd-pipeline)
12. [Documentation Standards](#12-documentation-standards)
13. [Error Handling Contract](#13-error-handling-contract)
14. [Definition of Done](#14-definition-of-done)

---

## 1. Project Overview

### What is jax-gemini?

`jax-gemini` is a Python library that lets non-technical AI/ML practitioners build, train, evaluate, save, and load JAX/Flax neural network models using plain English prompts — without writing JAX code themselves.

It follows the same pattern as PandasAI: the LLM (Gemini) generates runnable Python code, the library executes it in a controlled sandbox, and a **real JAX/Flax object** is returned to the user — not a string, not a description, a real `flax.nnx.Module`.

### Target Users

- Researchers who understand ML concepts but not JAX syntax
- Students learning deep learning who want to experiment quickly
- Domain experts (biologists, economists) who need to train models on their data
- Developers prototyping architectures before writing production code

### Core Design Principles

1. **Real objects, not text** — Every operation returns a real Python/JAX object
2. **Natural language first** — The user never needs to write JAX code
3. **Safe by default** — All generated code is validated and sandboxed before execution
4. **Self-healing** — Failed code generation is automatically retried with error context
5. **Conversational** — Multi-turn memory so users can iteratively refine models
6. **Minimal surface** — Simple enough to use in 3 lines; powerful enough for production

### Inspiration & References

| Project | What to Learn From It |
|---|---|
| `keras-gemini` (wkambale) | Proof that Gemini can generate valid Keras code from NL prompts. Gaps: no structured output, no sandbox, no training pipeline, no retry loop. |
| `blazerpc` (Ifihan) | Gold standard for Python ML library structure: `src/` layout, `pyproject.toml`, decorator API, layered architecture, MkDocs docs. |
| `pandasai` (sinaptik-ai) | The UX model. Natural language → LLM generates code → library executes it → real object returned. User never sees generated code unless they ask. |

### What jax-gemini Does NOT Do

- It does not replace JAX. It wraps it.
- It does not serve models. Use blazerpc or BentoML for serving.
- It does not manage datasets. Users pass numpy arrays or jax arrays.
- It does not provide a UI. It is a Python library only.
- It does not support PyTorch or TensorFlow. JAX/Flax only.

---

## 2. Tech Stack & Dependencies

### Runtime Dependencies

```toml
[project.dependencies]
google-generativeai = ">=0.8.0"   # Gemini API — structured output support
jax = ">=0.4.25"                  # Core numerical computing
flax = ">=0.9.0"                  # Neural network library (use NNX API, not Linen)
optax = ">=0.2.0"                 # Gradient-based optimization
orbax-checkpoint = ">=0.6.0"      # Model checkpointing
numpy = ">=1.24.0"                # Array utilities
```

### Optional Dependencies

```toml
[project.optional-dependencies]
gpu    = ["jax[cuda12]"]
tpu    = ["jax[tpu]"]
dev    = ["pytest>=8.0", "pytest-cov", "ruff", "mypy", "pre-commit", "hatch"]
docs   = ["mkdocs-material", "mkdocstrings[python]"]
```

### Build System

```toml
[build-system]
requires      = ["hatchling"]
build-backend = "hatchling.build"
```

### Python Version

- Minimum: Python 3.10
- Tested on: 3.10, 3.11, 3.12
- Reason: uses `match/case` for intent routing, `X | Y` union types, `ParamSpec`

### Key API Choices

- **Flax NNX** (not Linen): NNX is the current Flax API as of 2025. It uses stateful modules similar to PyTorch. Do NOT use `flax.linen`.
- **Gemini structured output**: Use `response_mime_type="application/json"` with `response_schema` — never parse free text.
- **Orbax**: Use `orbax.checkpoint.PyTreeCheckpointer` for model save/load.
- **Optax**: Use `optax.chain()` and `optax.apply_updates()` patterns in generated training code.

---

## 3. Repository Structure

Create this exact structure. Do not deviate.

```
jax-gemini/
├── src/
│   └── jax_gemini/
│       ├── __init__.py
│       ├── agent.py
│       ├── config.py
│       ├── exceptions.py
│       ├── prompts/
│       │   ├── __init__.py
│       │   ├── system_prompt.py
│       │   ├── intent.py
│       │   └── context_builder.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── gemini.py
│       ├── codegen/
│       │   ├── __init__.py
│       │   ├── parser.py
│       │   ├── validator.py
│       │   └── fixer.py
│       ├── sandbox/
│       │   ├── __init__.py
│       │   ├── executor.py
│       │   └── namespace.py
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── builder.py
│       │   ├── trainer.py
│       │   ├── evaluator.py
│       │   └── exporter.py
│       ├── memory/
│       │   ├── __init__.py
│       │   └── conversation.py
│       └── contrib/
│           ├── __init__.py
│           ├── flax.py
│           └── optax.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_parser.py
│   │   ├── test_validator.py
│   │   ├── test_executor.py
│   │   ├── test_memory.py
│   │   └── test_intent.py
│   └── integration/
│       ├── test_build_pipeline.py
│       ├── test_train_pipeline.py
│       └── test_full_conversation.py
├── examples/
│   ├── 01_build_basic_mlp.py
│   ├── 02_build_and_train_mnist.py
│   ├── 03_conversation_refinement.py
│   ├── 04_save_and_load.py
│   └── 05_custom_training_loop.py
├── docs/
│   ├── index.md
│   ├── quickstart.md
│   ├── api-reference.md
│   ├── architecture.md
│   ├── security.md
│   └── examples/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── publish.yml
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
└── AGENTS.md           ← this file
```

---

## 4. Implementation Phases

Implement in this exact order. Do not start Phase N+1 until Phase N passes all its tests.

### Phase 1 — Foundation (Weeks 1–2)

**Goal:** `pip install jax-gemini` works and `jg.build("Build a 3-layer MLP")` returns a real `flax.nnx.Module`.

**Tasks:**

1. Set up repository with the full directory structure above
2. Create `pyproject.toml` (see Section 10)
3. Implement `config.py` — `JaxGeminiConfig` dataclass
4. Implement `exceptions.py` — full exception hierarchy
5. Implement `llm/base.py` — `AbstractLLM` interface
6. Implement `llm/gemini.py` — `GeminiLLM` with structured output
7. Implement `prompts/system_prompt.py` — BUILD system prompt (see Section 9)
8. Implement `prompts/intent.py` — intent classification
9. Implement `prompts/context_builder.py` — context assembly
10. Implement `codegen/parser.py` — JSON code extraction
11. Implement `codegen/validator.py` — AST validation
12. Implement `sandbox/namespace.py` — whitelisted namespace
13. Implement `sandbox/executor.py` — safe execution
14. Implement `pipeline/builder.py` — BUILD handler
15. Implement `memory/conversation.py` — conversation history
16. Implement `agent.py` — `JaxGemini` orchestrator class
17. Implement `__init__.py` — public API surface
18. Write all unit tests for Phase 1 modules
19. Write integration test: `test_build_pipeline.py`
20. Verify: `python -m build` produces valid wheel and sdist

**Acceptance Criteria:**

```python
import jax_gemini as jg

jg.config.set({"gemini_api_key": "KEY"})
model = jg.build("Build a 3-layer MLP classifier with 128 hidden units")
assert hasattr(model, "__call__")   # Is a callable Flax module
print(jg.show_code())               # Shows generated Python
print(jg.explain())                 # Plain-English explanation
```

---

### Phase 2 — Training Pipeline (Weeks 3–4)

**Goal:** `jg.train()` executes a real JAX training loop and returns metrics.

**Tasks:**

1. Implement `prompts/system_prompt.py` — TRAIN system prompt
2. Implement `pipeline/trainer.py` — TRAIN handler
3. Implement `codegen/fixer.py` — error-feedback retry loop
4. Add `train()` method to `agent.py`
5. Write unit tests for `trainer.py` and `fixer.py`
6. Write integration test: `test_train_pipeline.py`

**Acceptance Criteria:**

```python
import numpy as np
import jax_gemini as jg

jg.config.set({"gemini_api_key": "KEY"})
model = jg.build("Build a small MLP for MNIST digit classification")

X = np.random.randn(100, 784).astype(np.float32)
y = np.random.randint(0, 10, size=(100,))

model, metrics = jg.train(
    "Train for 5 epochs with Adam optimizer and cross-entropy loss",
    dataset=(X, y)
)
assert "loss" in metrics
assert "accuracy" in metrics
assert metrics["accuracy"] > 0.0
```

---

### Phase 3 — Evaluate + Save/Load (Week 5)

**Goal:** Full lifecycle works end-to-end.

**Tasks:**

1. Implement `prompts/system_prompt.py` — EVALUATE system prompt
2. Implement `pipeline/evaluator.py` — EVALUATE handler
3. Implement `pipeline/exporter.py` — SAVE/LOAD handler using Orbax
4. Add `evaluate()`, `save()`, `load()` methods to `agent.py`
5. Write tests for all three handlers

**Acceptance Criteria:**

```python
model, metrics = jg.train("Train for 3 epochs", dataset=(X_train, y_train))
eval_metrics = jg.evaluate("Evaluate on test set", dataset=(X_test, y_test))
assert "accuracy" in eval_metrics

path = jg.save("my_mnist_model")
assert os.path.exists(path)

loaded_model = jg.load("my_mnist_model")
assert loaded_model is not None
```

---

### Phase 4 — Conversation Memory (Week 6)

**Goal:** Multi-turn context works. `jg.modify()` refines the current model.

**Tasks:**

1. Extend `ConversationMemory` to track model state and code history
2. Implement `jg.modify()` method
3. Implement `jg.reset()` method
4. Ensure all methods contribute to and read from memory
5. Write integration test: `test_full_conversation.py`

**Acceptance Criteria:**

```python
model = jg.build("Build a basic 2-layer MLP")
model = jg.modify("Add dropout with rate 0.3 between the layers")
model = jg.modify("Change the output to 5 classes instead of 10")
model, metrics = jg.train("Train it for 3 epochs with SGD", dataset=(X, y))
jg.reset()
# After reset, next build starts with no prior context
model2 = jg.build("Build a CNN")
```

---

### Phase 5 — Self-Healing Loop (Week 6, continued)

**Goal:** Failed code generation retries automatically with error context.

**Tasks:**

1. Integrate `codegen/fixer.py` into `agent._generate_and_execute()`
2. Max 3 attempts per operation
3. Each retry sends: original prompt + generated code + error traceback to Gemini
4. On 3rd failure: raise `JaxGeminiExecutionError` with all context attached
5. Write unit tests that mock Gemini to fail on attempt 1, succeed on attempt 2

**Acceptance Criteria:**

```python
# Configurable retry count
jg.config.set({"max_retries": 3, "verbose": True})

# When verbose=True, retries print to stdout:
# [jax-gemini] Attempt 1 failed: NameError: name 'nn' is not defined. Retrying...
# [jax-gemini] Attempt 2 succeeded.
```

---

### Phase 6 — Documentation & PyPI Release (Week 7)

**Tasks:**

1. Write full `README.md` (see Section 12)
2. Write `docs/quickstart.md`
3. Write `docs/api-reference.md`
4. Write `docs/security.md`
5. Write `docs/architecture.md`
6. Configure MkDocs with Material theme
7. Write all 5 example scripts in `examples/`
8. Set up GitHub Actions CI/CD (see Section 11)
9. Publish v0.1.0 to PyPI

---

## 5. Module Specifications

### 5.1 `config.py`

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JaxGeminiConfig:
    gemini_api_key: Optional[str] = None
    model_name: str = "gemini-3.1-pro"
    temperature: float = 0.2          # Low temp for deterministic code
    max_output_tokens: int = 4096
    max_retries: int = 3
    verbose: bool = False
    checkpoint_dir: str = "./jax_gemini_checkpoints"
    timeout_seconds: int = 60

    def set(self, updates: dict) -> None:
        """Update config from a dict. Used as jg.config.set({...})."""
        for key, value in updates.items():
            if not hasattr(self, key):
                raise JaxGeminiConfigError(f"Unknown config key: '{key}'")
            setattr(self, key, value)

    def validate(self) -> None:
        """Raise JaxGeminiConfigError if config is invalid."""
        if not self.gemini_api_key:
            raise JaxGeminiConfigError(
                "gemini_api_key is not set. "
                "Call jg.config.set({'gemini_api_key': 'YOUR_KEY'}) first, "
                "or set the GEMINI_API_KEY environment variable."
            )
        if self.temperature < 0 or self.temperature > 1:
            raise JaxGeminiConfigError("temperature must be between 0.0 and 1.0")
        if self.max_retries < 1 or self.max_retries > 5:
            raise JaxGeminiConfigError("max_retries must be between 1 and 5")
```

**Rules:**
- `JaxGeminiConfig` must be a dataclass, not a dict or plain class
- `set()` must validate keys — never silently ignore unknown keys
- Auto-read `GEMINI_API_KEY` env var in `validate()` if `gemini_api_key` is None
- All fields must have defaults so `JaxGeminiConfig()` works with zero arguments

---

### 5.2 `exceptions.py`

```python
class JaxGeminiError(Exception):
    """Base exception for all jax-gemini errors."""
    pass


class JaxGeminiConfigError(JaxGeminiError):
    """Raised when configuration is invalid or missing."""
    pass


class JaxGeminiLLMError(JaxGeminiError):
    """Raised when the Gemini API call fails."""
    def __init__(self, message: str, raw_response: str = ""):
        super().__init__(message)
        self.raw_response = raw_response


class JaxGeminiValidationError(JaxGeminiError):
    """Raised when generated code fails AST validation."""
    def __init__(self, message: str, code: str = ""):
        super().__init__(message)
        self.code = code


class JaxGeminiExecutionError(JaxGeminiError):
    """Raised when generated code fails to execute after all retries."""
    def __init__(self, message: str, code: str = "", attempts: int = 0):
        super().__init__(message)
        self.code = code
        self.attempts = attempts


class JaxGeminiIntentError(JaxGeminiError):
    """Raised when the intent classifier cannot determine the operation."""
    pass


class JaxGeminiCheckpointError(JaxGeminiError):
    """Raised when model save or load fails."""
    pass
```

**Rules:**
- All exceptions must inherit from `JaxGeminiError`
- Error messages must be actionable — tell the user what to do, not just what went wrong
- Attach relevant context (code, response) to exceptions as attributes

---

### 5.3 `llm/base.py`

```python
from abc import ABC, abstractmethod
from typing import Any


class AbstractLLM(ABC):

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], schema: dict) -> dict:
        """
        Send messages to the LLM and return a parsed JSON response.

        Args:
            messages: List of {"role": "user"|"model", "parts": [str]} dicts
            schema: JSON schema the response must conform to

        Returns:
            Parsed dict matching the provided schema

        Raises:
            JaxGeminiLLMError: If the API call fails or response is malformed
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM is reachable and the API key is valid."""
        ...
```

---

### 5.4 `llm/gemini.py`

```python
import json
import os
import google.generativeai as genai
from jax_gemini.llm.base import AbstractLLM
from jax_gemini.exceptions import JaxGeminiLLMError, JaxGeminiConfigError


class GeminiLLM(AbstractLLM):

    def __init__(self, config):
        api_key = config.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise JaxGeminiConfigError(
                "No Gemini API key found. Set gemini_api_key in config "
                "or export GEMINI_API_KEY as an environment variable."
            )
        genai.configure(api_key=api_key)
        self._config = config

    def generate(self, messages: list[dict], schema: dict) -> dict:
        try:
            model = genai.GenerativeModel(
                model_name=self._config.model_name,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=self._config.temperature,
                    max_output_tokens=self._config.max_output_tokens,
                ),
            )
            response = model.generate_content(messages)
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            raise JaxGeminiLLMError(
                f"Gemini returned invalid JSON: {e}",
                raw_response=response.text if response else ""
            )
        except Exception as e:
            raise JaxGeminiLLMError(f"Gemini API error: {e}")

    def is_available(self) -> bool:
        try:
            model = genai.GenerativeModel(self._config.model_name)
            model.generate_content("ping")
            return True
        except Exception:
            return False
```

**Rules:**
- Always use `response_mime_type="application/json"` — never parse free text
- Always set `temperature=0.2` for code generation (low = deterministic)
- Wrap ALL API calls in try/except — never let `google.generativeai` exceptions propagate raw
- Never log or print the API key

---

### 5.5 `codegen/parser.py`

```python
import json
from jax_gemini.exceptions import JaxGeminiLLMError


class CodeParser:

    @staticmethod
    def extract(response: dict, intent: str) -> str:
        """
        Extract the 'code' field from a structured Gemini response.

        Args:
            response: Parsed JSON dict from GeminiLLM.generate()
            intent: One of 'build', 'train', 'evaluate', 'save', 'load'

        Returns:
            Python code string

        Raises:
            JaxGeminiLLMError: If 'code' field is missing or empty
        """
        if "code" not in response:
            raise JaxGeminiLLMError(
                "Gemini response missing 'code' field. "
                f"Got fields: {list(response.keys())}",
                raw_response=str(response)
            )
        code = response["code"].strip()
        if not code:
            raise JaxGeminiLLMError("Gemini returned an empty code block.")
        return code
```

---

### 5.6 `codegen/validator.py`

This is the security boundary. Implement it with maximum care.

```python
import ast
from typing import Any
from jax_gemini.exceptions import JaxGeminiValidationError


WHITELIST_IMPORTS: frozenset[str] = frozenset({
    "jax", "flax", "optax", "orbax", "numpy",
    "functools", "math", "typing", "dataclasses",
})

BANNED_BUILTINS: frozenset[str] = frozenset({
    "eval", "exec", "compile", "__import__",
    "open", "input", "memoryview", "breakpoint",
})

BANNED_ATTRIBUTES: frozenset[str] = frozenset({
    "__class__", "__bases__", "__subclasses__",
    "__globals__", "__locals__", "__builtins__",
    "__code__", "__closure__", "__reduce__",
})

REQUIRED_FUNCTION_NAMES: dict[str, str] = {
    "build":    "build_model",
    "train":    "train_model",
    "evaluate": "evaluate_model",
    "save":     "save_model",
    "load":     "load_model",
}

MAX_CODE_LINES = 200


class CodeValidator:

    @staticmethod
    def validate(code: str, intent: str) -> None:
        """
        Validate generated code via AST analysis.

        Raises JaxGeminiValidationError with a descriptive message
        if any security or structural rule is violated.
        """
        # 1. Line limit check
        lines = code.strip().splitlines()
        if len(lines) > MAX_CODE_LINES:
            raise JaxGeminiValidationError(
                f"Generated code is {len(lines)} lines — maximum is {MAX_CODE_LINES}. "
                "This may indicate a hallucination or prompt injection attempt.",
                code=code
            )

        # 2. Parse to AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise JaxGeminiValidationError(
                f"Generated code has a syntax error: {e}",
                code=code
            )

        # 3. Walk AST and check every node
        for node in ast.walk(tree):

            # Import validation
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in WHITELIST_IMPORTS:
                        raise JaxGeminiValidationError(
                            f"Import '{alias.name}' is not in the allowed list. "
                            f"Allowed: {sorted(WHITELIST_IMPORTS)}",
                            code=code
                        )

            if isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root not in WHITELIST_IMPORTS:
                        raise JaxGeminiValidationError(
                            f"Import from '{node.module}' is not allowed.",
                            code=code
                        )

            # Banned builtin calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BANNED_BUILTINS:
                        raise JaxGeminiValidationError(
                            f"Call to '{node.func.id}' is not allowed in generated code.",
                            code=code
                        )

            # Banned attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in BANNED_ATTRIBUTES:
                    raise JaxGeminiValidationError(
                        f"Access to attribute '{node.attr}' is not allowed.",
                        code=code
                    )

        # 4. Required function check
        required_fn = REQUIRED_FUNCTION_NAMES.get(intent)
        if required_fn:
            defined_functions = {
                node.name for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            }
            if required_fn not in defined_functions:
                raise JaxGeminiValidationError(
                    f"Generated code must define a function named '{required_fn}'. "
                    f"Found functions: {sorted(defined_functions)}",
                    code=code
                )
```

**Rules:**
- This class must have zero external dependencies beyond `ast` and project exceptions
- Every check must raise a `JaxGeminiValidationError` with a message explaining what was wrong
- The validation error message is fed back to Gemini on retry — it must be specific enough to guide a fix
- Add new banned items conservatively; removing items requires a security review
- Never catch exceptions inside this class

---

### 5.7 `sandbox/namespace.py`

```python
import jax
import jax.numpy as jnp
import flax
import flax.nnx as nnx
import optax
import numpy as np

try:
    import orbax.checkpoint as ocp
    _orbax_available = True
except ImportError:
    _orbax_available = False

SAFE_BUILTINS = {
    name: getattr(__builtins__, name, None)
    for name in [
        "print", "range", "len", "enumerate", "zip", "map", "filter",
        "sorted", "reversed", "min", "max", "sum", "abs", "round",
        "int", "float", "str", "bool", "list", "dict", "tuple", "set",
        "isinstance", "hasattr", "getattr", "type", "repr",
        "True", "False", "None",
    ]
    if getattr(__builtins__, name, None) is not None
}

BASE_NAMESPACE: dict = {
    "jax": jax,
    "jnp": jnp,
    "flax": flax,
    "nnx": nnx,
    "optax": optax,
    "np": np,
    "__builtins__": SAFE_BUILTINS,
}

if _orbax_available:
    import orbax.checkpoint as ocp
    BASE_NAMESPACE["ocp"] = ocp
    BASE_NAMESPACE["orbax"] = __import__("orbax")


def get_execution_namespace() -> dict:
    """Return a fresh copy of the safe namespace for each execution."""
    return dict(BASE_NAMESPACE)
```

---

### 5.8 `sandbox/executor.py`

```python
import traceback
from typing import Any
from jax_gemini.sandbox.namespace import get_execution_namespace
from jax_gemini.exceptions import JaxGeminiExecutionError


INTENT_FUNCTION_MAP: dict[str, str] = {
    "build":    "build_model",
    "train":    "train_model",
    "evaluate": "evaluate_model",
    "save":     "save_model",
    "load":     "load_model",
}

INTENT_DEFAULT_ARGS: dict[str, dict] = {
    "build":    {},
    "train":    {"dataset": None, "epochs": 5, "learning_rate": 1e-3},
    "evaluate": {"dataset": None},
    "save":     {"path": "./checkpoint"},
    "load":     {"path": "./checkpoint"},
}


class SandboxExecutor:

    def run(
        self,
        code: str,
        intent: str,
        args: dict | None = None,
    ) -> Any:
        """
        Execute validated code in a restricted namespace.

        Args:
            code: Python source code string (already AST-validated)
            intent: Operation intent for function name lookup
            args: Runtime arguments to pass to the generated function

        Returns:
            Whatever the generated function returns

        Raises:
            JaxGeminiExecutionError: If execution fails
        """
        namespace = get_execution_namespace()
        fn_name = INTENT_FUNCTION_MAP[intent]
        call_args = {**INTENT_DEFAULT_ARGS.get(intent, {}), **(args or {})}

        try:
            exec(code, namespace)  # noqa: S102 — code is pre-validated by CodeValidator
        except Exception as e:
            tb = traceback.format_exc()
            raise JaxGeminiExecutionError(
                f"Code failed during definition phase: {e}\n\n{tb}",
                code=code,
                attempts=1,
            )

        if fn_name not in namespace:
            raise JaxGeminiExecutionError(
                f"Function '{fn_name}' not found in namespace after exec. "
                "This is a validator bug — please report it.",
                code=code,
            )

        try:
            result = namespace[fn_name](**call_args)
            return result
        except Exception as e:
            tb = traceback.format_exc()
            raise JaxGeminiExecutionError(
                f"Function '{fn_name}' raised an error during execution:\n"
                f"{type(e).__name__}: {e}\n\nTraceback:\n{tb}",
                code=code,
                attempts=1,
            )
```

---

### 5.9 `codegen/fixer.py`

```python
from jax_gemini.exceptions import JaxGeminiExecutionError, JaxGeminiValidationError


class CodeFixer:

    @staticmethod
    def build_fix_prompt(
        original_prompt: str,
        failed_code: str,
        error_message: str,
        attempt: int,
    ) -> str:
        """
        Build a correction prompt to send back to Gemini when code fails.

        The error message must be specific enough that Gemini can fix it.
        """
        return (
            f"The following code was generated for the request: '{original_prompt}'\n\n"
            f"```python\n{failed_code}\n```\n\n"
            f"It failed with this error (attempt {attempt}):\n"
            f"```\n{error_message}\n```\n\n"
            f"Please fix the code. Return ONLY corrected code in the same JSON format. "
            f"Do not change the function name or signature. "
            f"Only import from: jax, flax, optax, orbax, numpy."
        )
```

---

### 5.10 `memory/conversation.py`

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationTurn:
    role: str          # "user" or "model"
    content: str       # The message text


@dataclass
class ModelState:
    code: str          # Python code that produced this model
    description: str   # Gemini's description of the model
    intent: str        # What operation produced this state


class ConversationMemory:

    def __init__(self, max_turns: int = 20):
        self._turns: list[ConversationTurn] = []
        self._model_history: list[ModelState] = []
        self._max_turns = max_turns

    def add_turn(self, role: str, content: str) -> None:
        self._turns.append(ConversationTurn(role=role, content=content))
        if len(self._turns) > self._max_turns * 2:
            # Keep first 2 turns (system context) + latest turns
            self._turns = self._turns[:2] + self._turns[-(self._max_turns * 2 - 2):]

    def add_model_state(self, code: str, description: str, intent: str) -> None:
        self._model_history.append(ModelState(code=code, description=description, intent=intent))

    def get_current_model_state(self) -> ModelState | None:
        return self._model_history[-1] if self._model_history else None

    def to_gemini_messages(self) -> list[dict]:
        """Convert to the format expected by google-generativeai."""
        return [
            {"role": turn.role, "parts": [turn.content]}
            for turn in self._turns
        ]

    def reset(self) -> None:
        self._turns.clear()
        self._model_history.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)
```

---

### 5.11 `agent.py`

This is the orchestrator. It must be clean and delegate — no business logic here.

```python
from typing import Any
import os
import jax.numpy as jnp
import numpy as np

from jax_gemini.config import JaxGeminiConfig
from jax_gemini.exceptions import JaxGeminiExecutionError, JaxGeminiValidationError
from jax_gemini.llm.gemini import GeminiLLM
from jax_gemini.prompts.context_builder import ContextBuilder
from jax_gemini.prompts.intent import IntentClassifier
from jax_gemini.codegen.parser import CodeParser
from jax_gemini.codegen.validator import CodeValidator
from jax_gemini.codegen.fixer import CodeFixer
from jax_gemini.sandbox.executor import SandboxExecutor
from jax_gemini.memory.conversation import ConversationMemory
from jax_gemini.pipeline import builder, trainer, evaluator, exporter


class JaxGemini:

    def __init__(self, config: JaxGeminiConfig | None = None):
        self.config = config or JaxGeminiConfig()
        self._llm: GeminiLLM | None = None
        self._memory = ConversationMemory()
        self._executor = SandboxExecutor()
        self._last_code: str | None = None
        self._last_response: dict | None = None
        self._current_model: Any = None

    def _get_llm(self) -> GeminiLLM:
        if self._llm is None:
            self.config.validate()
            self._llm = GeminiLLM(self.config)
        return self._llm

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[jax-gemini] {message}")

    def _generate_and_execute(
        self,
        user_prompt: str,
        intent: str,
        runtime_args: dict | None = None,
    ) -> Any:
        llm = self._get_llm()
        context = ContextBuilder.build(
            intent=intent,
            user_prompt=user_prompt,
            memory=self._memory,
            current_model_state=self._memory.get_current_model_state(),
        )
        schema = ContextBuilder.get_schema(intent)
        last_error: Exception | None = None
        last_code: str = ""

        for attempt in range(1, self.config.max_retries + 1):
            self._log(f"Attempt {attempt}/{self.config.max_retries} for intent '{intent}'...")

            if attempt > 1 and last_code and last_error:
                fix_prompt = CodeFixer.build_fix_prompt(
                    original_prompt=user_prompt,
                    failed_code=last_code,
                    error_message=str(last_error),
                    attempt=attempt,
                )
                context = ContextBuilder.build(
                    intent=intent,
                    user_prompt=fix_prompt,
                    memory=self._memory,
                    current_model_state=self._memory.get_current_model_state(),
                )

            try:
                response = llm.generate(context, schema)
                code = CodeParser.extract(response, intent)
                CodeValidator.validate(code, intent)
                result = self._executor.run(code, intent, runtime_args)

                # Success — update state
                self._last_code = code
                self._last_response = response
                self._memory.add_turn("user", user_prompt)
                self._memory.add_turn("model", response.get("description", code))
                if intent in ("build", "train"):
                    self._memory.add_model_state(
                        code=code,
                        description=response.get("description", ""),
                        intent=intent,
                    )
                self._log(f"Attempt {attempt} succeeded.")
                return result

            except (JaxGeminiValidationError, JaxGeminiExecutionError) as e:
                last_error = e
                last_code = getattr(e, "code", "")
                self._log(f"Attempt {attempt} failed: {e}")

        raise JaxGeminiExecutionError(
            f"All {self.config.max_retries} attempts failed for '{user_prompt}'.\n"
            f"Last error: {last_error}",
            code=last_code,
            attempts=self.config.max_retries,
        )

    def build(self, prompt: str) -> Any:
        """Build a JAX/Flax model from a natural language description."""
        model = self._generate_and_execute(prompt, intent="build")
        self._current_model = model
        return model

    def modify(self, prompt: str) -> Any:
        """Refine the current model based on a natural language instruction."""
        model = self._generate_and_execute(prompt, intent="build")
        self._current_model = model
        return model

    def train(self, prompt: str, dataset: tuple | None = None) -> tuple[Any, dict]:
        """Train the current model. Returns (model, metrics_dict)."""
        args = {"dataset": dataset} if dataset else {}
        result = self._generate_and_execute(prompt, intent="train", runtime_args=args)
        if isinstance(result, tuple):
            self._current_model = result[0]
            return result
        return result, {}

    def evaluate(self, prompt: str, dataset: tuple | None = None) -> dict:
        """Evaluate the current model. Returns a metrics dict."""
        args = {"dataset": dataset} if dataset else {}
        return self._generate_and_execute(prompt, intent="evaluate", runtime_args=args)

    def save(self, name: str) -> str:
        """Save the current model checkpoint. Returns the checkpoint path."""
        path = os.path.join(self.config.checkpoint_dir, name)
        return self._generate_and_execute(
            f"Save the model to {path}", intent="save",
            runtime_args={"model": self._current_model, "path": path}
        )

    def load(self, name: str) -> Any:
        """Load a model checkpoint by name."""
        path = os.path.join(self.config.checkpoint_dir, name)
        model = self._generate_and_execute(
            f"Load the model from {path}", intent="load",
            runtime_args={"path": path}
        )
        self._current_model = model
        return model

    def explain(self) -> str:
        """Return a plain-English explanation of the last generated code."""
        if not self._last_code:
            return "No code has been generated yet. Call jg.build() first."
        if self._last_response and "description" in self._last_response:
            return self._last_response["description"]
        return "No explanation available for the last operation."

    def show_code(self) -> str:
        """Return the raw generated Python code from the last operation."""
        if not self._last_code:
            return "No code has been generated yet."
        return self._last_code

    def reset(self) -> None:
        """Clear all conversation history and model state."""
        self._memory.reset()
        self._current_model = None
        self._last_code = None
        self._last_response = None
        self._log("Session reset. Memory cleared.")
```

---

### 5.12 `__init__.py`

```python
"""
jax-gemini: Natural language-driven JAX/Flax model building powered by Gemini.

Quick start:
    import jax_gemini as jg
    jg.config.set({"gemini_api_key": "YOUR_KEY"})
    model = jg.build("Build a 3-layer MLP for image classification")
"""

from jax_gemini.agent import JaxGemini
from jax_gemini.config import JaxGeminiConfig
from jax_gemini.exceptions import (
    JaxGeminiError,
    JaxGeminiConfigError,
    JaxGeminiLLMError,
    JaxGeminiValidationError,
    JaxGeminiExecutionError,
    JaxGeminiIntentError,
    JaxGeminiCheckpointError,
)

__version__ = "0.1.0"
__author__ = "Wesley Kambale"
__license__ = "MIT"

__all__ = [
    "JaxGemini",
    "JaxGeminiConfig",
    "JaxGeminiError",
    "JaxGeminiConfigError",
    "JaxGeminiLLMError",
    "JaxGeminiValidationError",
    "JaxGeminiExecutionError",
    "JaxGeminiIntentError",
    "JaxGeminiCheckpointError",
    "config",
    "build",
    "modify",
    "train",
    "evaluate",
    "save",
    "load",
    "explain",
    "show_code",
    "reset",
]

# Module-level singleton for simple usage pattern (jg.build, jg.train, etc.)
_default_agent = JaxGemini()
config = _default_agent.config
build = _default_agent.build
modify = _default_agent.modify
train = _default_agent.train
evaluate = _default_agent.evaluate
save = _default_agent.save
load = _default_agent.load
explain = _default_agent.explain
show_code = _default_agent.show_code
reset = _default_agent.reset
```

---

## 6. Code Style & Quality

### Formatter & Linter

Use `ruff` for all formatting and linting. Configure in `pyproject.toml`:

```toml
[tool.ruff]
target-version  = "py310"
line-length     = 100
src             = ["src"]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
    "S",    # bandit security checks
    "RUF",  # ruff-specific rules
]
ignore = [
    "S102", # Use of exec — allowed only in sandbox/executor.py
    "S301", # pickle — not used
]

[tool.ruff.lint.per-file-ignores]
"sandbox/executor.py" = ["S102"]  # exec is intentional here
"tests/*"             = ["S101"]  # assert is fine in tests
```

### Type Checking

Use `mypy`. Configure in `pyproject.toml`:

```toml
[tool.mypy]
python_version         = "3.10"
strict                 = true
warn_return_any        = true
warn_unused_configs    = true
ignore_missing_imports = true    # JAX stubs are incomplete
```

All public functions and methods **must** have full type annotations. Internal helpers should have annotations. Only generated code passed to the sandbox is exempt.

### Naming Conventions

| Thing | Convention | Example |
|---|---|---|
| Module | `snake_case` | `context_builder.py` |
| Class | `PascalCase` | `GeminiLLM`, `CodeValidator` |
| Function/method | `snake_case` | `build_fix_prompt()` |
| Constant | `UPPER_SNAKE_CASE` | `WHITELIST_IMPORTS` |
| Private method | `_snake_case` | `_get_llm()` |
| Type alias | `PascalCase` | `MessageList` |

### Docstring Style

Use Google-style docstrings on all public classes, methods, and functions:

```python
def generate(self, messages: list[dict], schema: dict) -> dict:
    """Send messages to Gemini and return a parsed JSON response.

    Args:
        messages: List of message dicts with 'role' and 'parts' keys.
        schema: JSON schema the response must conform to.

    Returns:
        Parsed dict matching the provided schema.

    Raises:
        JaxGeminiLLMError: If the API call fails or response is malformed.

    Example:
        >>> llm = GeminiLLM(config)
        >>> result = llm.generate([{"role": "user", "parts": ["hi"]}], schema)
    """
```

### Import Order

Always in this order (enforced by ruff isort):

1. Standard library
2. Third-party packages (`jax`, `flax`, `google.generativeai`)
3. Internal imports (`from jax_gemini.config import ...`)

### What NOT to Do

- **No `print()` in library code** — use `self._log()` which respects `verbose` config
- **No bare `except:`** — always catch specific exceptions
- **No mutable default arguments** — use `None` and assign inside function
- **No global state** outside `__init__.py` singleton — use `JaxGemini` instances
- **No hardcoded strings** for intent names — use the constants in `INTENT_FUNCTION_MAP`
- **No `from module import *`** — explicit imports only

---

## 7. Testing Requirements

### Test Runner

```bash
pytest tests/ -v --cov=src/jax_gemini --cov-report=term-missing
```

### Coverage Requirement

- Minimum 85% line coverage for Phase 1 completion
- Minimum 90% line coverage before PyPI publish
- Critical paths (validator, executor, retry loop) must be 100% covered

### Test Structure

#### `tests/conftest.py`

```python
import pytest
from unittest.mock import MagicMock
from jax_gemini.config import JaxGeminiConfig


@pytest.fixture
def config() -> JaxGeminiConfig:
    return JaxGeminiConfig(
        gemini_api_key="test-key-not-real",
        verbose=False,
        max_retries=2,
    )


@pytest.fixture
def mock_llm():
    """A mock LLM that returns a valid build response."""
    llm = MagicMock()
    llm.generate.return_value = {
        "intent": "build",
        "code": (
            "import flax.nnx as nnx\n"
            "def build_model(rngs):\n"
            "    return nnx.Linear(784, 10, rngs=rngs)\n"
        ),
        "description": "A simple linear model mapping 784 inputs to 10 outputs.",
        "warnings": [],
    }
    return llm
```

#### Unit Tests — What to Test

**`tests/unit/test_validator.py`** — This is the most important test file:

```python
import pytest
from jax_gemini.codegen.validator import CodeValidator
from jax_gemini.exceptions import JaxGeminiValidationError

VALID_BUILD_CODE = """
import flax.nnx as nnx

def build_model(rngs):
    return nnx.Linear(784, 10, rngs=rngs)
"""

class TestCodeValidator:

    def test_valid_code_passes(self):
        CodeValidator.validate(VALID_BUILD_CODE, "build")  # Should not raise

    def test_missing_required_function_raises(self):
        code = "def wrong_name(rngs):\n    pass"
        with pytest.raises(JaxGeminiValidationError, match="build_model"):
            CodeValidator.validate(code, "build")

    def test_banned_import_raises(self):
        code = "import os\ndef build_model(rngs): pass"
        with pytest.raises(JaxGeminiValidationError, match="os"):
            CodeValidator.validate(code, "build")

    def test_banned_builtin_call_raises(self):
        code = "def build_model(rngs):\n    eval('print(1)')"
        with pytest.raises(JaxGeminiValidationError, match="eval"):
            CodeValidator.validate(code, "build")

    def test_banned_attribute_access_raises(self):
        code = "def build_model(rngs):\n    x = rngs.__class__"
        with pytest.raises(JaxGeminiValidationError, match="__class__"):
            CodeValidator.validate(code, "build")

    def test_syntax_error_raises(self):
        code = "def build_model(rngs):\n    return ("
        with pytest.raises(JaxGeminiValidationError, match="syntax"):
            CodeValidator.validate(code, "build")

    def test_too_many_lines_raises(self):
        code = "\n".join(["# line"] * 201) + "\ndef build_model(rngs): pass"
        with pytest.raises(JaxGeminiValidationError, match="lines"):
            CodeValidator.validate(code, "build")

    def test_subprocess_import_raises(self):
        code = "import subprocess\ndef build_model(rngs): pass"
        with pytest.raises(JaxGeminiValidationError):
            CodeValidator.validate(code, "build")

    def test_from_import_whitelist(self):
        code = "from flax.nnx import Linear\ndef build_model(rngs): pass"
        CodeValidator.validate(code, "build")  # Should not raise

    def test_from_import_non_whitelist_raises(self):
        code = "from pathlib import Path\ndef build_model(rngs): pass"
        with pytest.raises(JaxGeminiValidationError, match="pathlib"):
            CodeValidator.validate(code, "build")
```

**`tests/unit/test_executor.py`:**

```python
import pytest
from jax_gemini.sandbox.executor import SandboxExecutor
from jax_gemini.exceptions import JaxGeminiExecutionError

class TestSandboxExecutor:

    def setup_method(self):
        self.executor = SandboxExecutor()

    def test_valid_build_code_executes(self):
        code = (
            "import flax.nnx as nnx\n"
            "def build_model(rngs):\n"
            "    return nnx.Linear(784, 10, rngs=rngs)\n"
        )
        result = self.executor.run(code, "build", {})
        assert result is not None

    def test_runtime_error_raises_execution_error(self):
        code = "def build_model(rngs):\n    raise ValueError('test error')\n"
        with pytest.raises(JaxGeminiExecutionError, match="test error"):
            self.executor.run(code, "build", {})

    def test_namespace_is_isolated(self):
        """Ensure exec does not leak into subsequent calls."""
        code1 = "SECRET = 'top_secret'\ndef build_model(rngs): return 42\n"
        self.executor.run(code1, "build", {})
        code2 = "def build_model(rngs): return SECRET\n"
        with pytest.raises(JaxGeminiExecutionError):
            self.executor.run(code2, "build", {})
```

**`tests/unit/test_memory.py`:**

```python
from jax_gemini.memory.conversation import ConversationMemory

class TestConversationMemory:

    def test_add_and_retrieve_turns(self):
        memory = ConversationMemory()
        memory.add_turn("user", "Build a model")
        memory.add_turn("model", "Here is the code...")
        assert memory.turn_count == 2

    def test_reset_clears_everything(self):
        memory = ConversationMemory()
        memory.add_turn("user", "hello")
        memory.reset()
        assert memory.turn_count == 0
        assert memory.get_current_model_state() is None

    def test_max_turns_enforced(self):
        memory = ConversationMemory(max_turns=3)
        for i in range(20):
            memory.add_turn("user", f"message {i}")
            memory.add_turn("model", f"response {i}")
        # Should not exceed 2 * max_turns * 2 (with system context)
        assert memory.turn_count <= 12
```

#### Integration Tests

Integration tests **must** mock the Gemini API — they must not make real network calls:

```python
# tests/integration/test_build_pipeline.py
from unittest.mock import patch, MagicMock
import pytest
import jax_gemini as jg

MOCK_BUILD_RESPONSE = {
    "intent": "build",
    "code": (
        "import flax.nnx as nnx\n\n"
        "def build_model(rngs):\n"
        "    model = nnx.Sequential(\n"
        "        nnx.Linear(784, 256, rngs=rngs),\n"
        "        nnx.relu,\n"
        "        nnx.Linear(256, 10, rngs=rngs),\n"
        "    )\n"
        "    return model\n"
    ),
    "description": "A 2-layer MLP for classification.",
    "warnings": [],
}

class TestBuildPipeline:

    @patch("jax_gemini.llm.gemini.genai")
    def test_build_returns_flax_module(self, mock_genai):
        mock_response = MagicMock()
        mock_response.text = '{"intent": "build", "code": "...", ...}'
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response

        # ... set up proper mock and assert model is returned
        pass

    def test_build_without_api_key_raises_config_error(self):
        agent = jg.JaxGemini()
        # No API key set
        with pytest.raises(jg.JaxGeminiConfigError):
            agent.build("Build a model")
```

### What Must Be Mocked

- All calls to `google.generativeai` — never make real API calls in tests
- File system operations in exporter tests — use `tmp_path` pytest fixture
- No real JAX model execution is required in unit tests — mock the executor

### What Must NOT Be Mocked

- `CodeValidator` — always run real AST validation in tests
- `CodeParser` — always run real parsing logic
- `ConversationMemory` — always run real memory logic
- `JaxGeminiConfig` — always use real config with test values

---

## 8. Security Rules

These are non-negotiable. Every rule here is a hard requirement.

### Rule 1 — Never Execute Unvalidated Code

The call order is always: `CodeParser → CodeValidator → SandboxExecutor`. Skipping `CodeValidator` is never acceptable. There are no exceptions.

### Rule 2 — Import Whitelist is the Security Boundary

The `WHITELIST_IMPORTS` set in `validator.py` is the single source of truth for what generated code is allowed to import. To add a new allowed import:

1. Add it to `WHITELIST_IMPORTS` in `validator.py`
2. Add it to `BASE_NAMESPACE` in `sandbox/namespace.py`
3. Add a test case to `test_validator.py` confirming it validates
4. Document the addition in `CHANGELOG.md`

### Rule 3 — Sandbox Namespace Must Be Fresh Per Execution

Never reuse a namespace dict across executor calls. `get_execution_namespace()` returns a new `dict` every time. This prevents state leakage between executions.

### Rule 4 — Never Log Sensitive Data

The following must never appear in logs, print output, or error messages:
- The Gemini API key (full or partial)
- Dataset contents passed by the user
- File paths beyond what the user provided

### Rule 5 — Never Eval or Exec Outside the Sandbox

The only place in the entire codebase where `exec()` is called is `sandbox/executor.py`. No other module may use `exec`, `eval`, or `compile`. Ruff rule `S102` is suppressed only for that file.

### Rule 6 — User Input Is Untrusted

All user prompts must be treated as potentially adversarial. The `ContextBuilder` must not interpolate raw user strings into system prompt sections. User input goes into the "user" turn of the message history only, never into the system prompt template.

### Rule 7 — Dependency Security

- Pin minimum versions (not exact versions) in `pyproject.toml`
- Run `pip audit` in CI to detect known vulnerabilities
- Never add a dependency that has fewer than 500 GitHub stars without explicit justification

### Rule 8 — API Key Handling

```python
# CORRECT — read from env, never hardcode
api_key = config.gemini_api_key or os.environ.get("GEMINI_API_KEY")

# WRONG — never do this
api_key = "AIzaSy..."
```

The library must auto-read `GEMINI_API_KEY` from environment variables before raising an error.

---

## 9. Prompt Engineering

### 9.1 System Prompt Design Principles

The system prompt is the most important artifact in this project. It determines whether Gemini generates correct, safe, executable JAX code.

**Every system prompt must:**
1. State the exact Flax NNX API version being targeted
2. Include 2–3 few-shot examples with correct code
3. State the required function name and signature explicitly
4. State the exact allowed imports
5. State output format as JSON with exact field names
6. Include at least one counter-example showing what NOT to do

### 9.2 BUILD System Prompt

Store in `jax_gemini/prompts/system_prompt.py` as a constant:

```python
BUILD_SYSTEM_PROMPT = """
You are an expert JAX/Flax neural network engineer. Your task is to generate
executable Python code that builds a Flax NNX neural network model based on
the user's natural language description.

RULES:
- Use Flax NNX (flax.nnx), NOT flax.linen.
- The generated code MUST define exactly one function named `build_model`.
- `build_model` MUST accept a single argument `rngs` of type `nnx.Rngs`.
- `build_model` MUST return an `nnx.Module` instance.
- Only import from: jax, jax.numpy (as jnp), flax, flax.nnx (as nnx), numpy (as np).
- Do NOT import os, sys, subprocess, pathlib, socket, or any other module.
- Do NOT use eval, exec, compile, or open.
- Do NOT write more than 100 lines of code.

AVAILABLE FLAX NNX COMPONENTS:
- nnx.Linear(in_features, out_features, rngs=rngs)
- nnx.Conv(in_features, out_features, kernel_size, rngs=rngs)
- nnx.BatchNorm(num_features, rngs=rngs)
- nnx.Dropout(rate, rngs=rngs)
- nnx.Sequential(*layers)
- nnx.relu, nnx.tanh, nnx.sigmoid, nnx.gelu
- nnx.Rngs  (always pass rngs through to sub-modules)

EXAMPLE 1 — 3-layer MLP:
```python
import flax.nnx as nnx

def build_model(rngs: nnx.Rngs) -> nnx.Module:
    return nnx.Sequential(
        nnx.Linear(784, 256, rngs=rngs),
        nnx.relu,
        nnx.Linear(256, 128, rngs=rngs),
        nnx.relu,
        nnx.Linear(128, 10, rngs=rngs),
    )
```

EXAMPLE 2 — CNN for image classification:
```python
import flax.nnx as nnx

class SimpleCNN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.linear = nnx.Linear(64 * 5 * 5, 10, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = x.reshape(x.shape[0], -1)
        return self.linear(x)

def build_model(rngs: nnx.Rngs) -> nnx.Module:
    return SimpleCNN(rngs)
```

COUNTER-EXAMPLE — DO NOT DO THIS:
```python
import flax.linen as nn  # WRONG — use flax.nnx, not flax.linen
import os               # WRONG — os is not allowed
```

Respond ONLY with a JSON object matching this schema:
{
    "intent": "build",
    "code": "<python code string>",
    "description": "<one paragraph plain English explanation for a non-technical user>",
    "parameters": {"<key>": "<value>"},
    "warnings": ["<optional warning strings>"]
}
"""
```

### 9.3 TRAIN System Prompt

```python
TRAIN_SYSTEM_PROMPT = """
You are an expert JAX/Flax training engineer. Generate a complete training loop
for a Flax NNX model using JAX and Optax.

RULES:
- The function MUST be named `train_model`.
- Signature: `train_model(model, dataset, epochs=5, learning_rate=1e-3) -> tuple[model, dict]`
- `dataset` is a tuple of (X, y) numpy arrays.
- Return a tuple of (trained_model, metrics_dict).
- metrics_dict MUST contain at least "loss" and "accuracy" keys with float values.
- Use optax for the optimizer (optax.adam, optax.sgd, etc.).
- Use `nnx.Rngs(0)` for any random operations inside the loop.
- Do NOT use jax.grad directly — use optax's update pattern.
- Only import from: jax, jax.numpy, flax, flax.nnx, optax, numpy.

EXAMPLE training loop:
```python
import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx

def train_model(model, dataset, epochs=5, learning_rate=1e-3):
    X, y = dataset
    X = jnp.array(X)
    y = jnp.array(y)

    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))

    @nnx.jit
    def train_step(model, optimizer, x_batch, y_batch):
        def loss_fn(model):
            logits = model(x_batch)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y_batch).mean()
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    final_loss = 0.0
    for epoch in range(epochs):
        final_loss = float(train_step(model, optimizer, X, y))

    logits = model(X)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = float(jnp.mean(predictions == y))

    return model, {"loss": final_loss, "accuracy": accuracy}
```

Respond ONLY with JSON matching:
{
    "intent": "train",
    "code": "<python code>",
    "description": "<plain English explanation>",
    "hyperparameters": {"optimizer": "...", "loss": "..."},
    "warnings": []
}
"""
```

### 9.4 Prompt Versioning

- Every system prompt must have a version comment at the top: `# v1.0.0`
- When a system prompt is updated, increment the version
- Log which prompt version was used when `verbose=True`
- Do NOT embed the system prompt in API call messages — always use the `system` parameter equivalent (pass as first "user" turn if Gemini SDK does not support a dedicated system turn)

### 9.5 Context Builder

The `ContextBuilder.build()` method assembles the full message list sent to Gemini:

```python
# Message structure sent to Gemini:
[
    {"role": "user",  "parts": [SYSTEM_PROMPT_FOR_INTENT]},
    {"role": "model", "parts": ["Understood. I will generate JAX/Flax code."]},
    # ... conversation history turns ...
    {"role": "user",  "parts": [CURRENT_USER_PROMPT]},
]
```

The system prompt always goes first. Conversation history follows. User prompt is last.

---

## 10. PyPI Packaging

### `pyproject.toml` (complete)

```toml
[build-system]
requires      = ["hatchling"]
build-backend = "hatchling.build"

[project]
name            = "jax-gemini"
version         = "0.1.0"
description     = "Natural language-driven JAX/Flax model building powered by Gemini"
readme          = "README.md"
license         = { file = "LICENSE" }
authors         = [{ name = "Wesley Kambale", email = "your@email.com" }]
requires-python = ">=3.10"
keywords        = ["jax", "flax", "gemini", "machine-learning", "natural-language", "ai", "neural-networks"]
classifiers     = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "google-generativeai>=0.8.0",
    "jax>=0.4.25",
    "flax>=0.9.0",
    "optax>=0.2.0",
    "orbax-checkpoint>=0.6.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
gpu  = ["jax[cuda12]"]
tpu  = ["jax[tpu]"]
dev  = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.4.0", "mypy>=1.9.0", "pre-commit", "hatch"]
docs = ["mkdocs-material>=9.5", "mkdocstrings[python]>=0.24.0"]

[project.urls]
Homepage      = "https://github.com/wkambale/jax-gemini"
Documentation = "https://wkambale.github.io/jax-gemini"
Repository    = "https://github.com/wkambale/jax-gemini"
Issues        = "https://github.com/wkambale/jax-gemini/issues"
Changelog     = "https://github.com/wkambale/jax-gemini/blob/main/CHANGELOG.md"

[tool.hatch.build.targets.wheel]
packages = ["src/jax_gemini"]

[tool.ruff]
target-version = "py310"
line-length    = 100
src            = ["src"]

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP", "S", "RUF"]
ignore = ["S102"]

[tool.ruff.lint.per-file-ignores]
"src/jax_gemini/sandbox/executor.py" = ["S102"]
"tests/*" = ["S101", "S105"]

[tool.mypy]
python_version         = "3.10"
strict                 = true
warn_return_any        = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths    = ["tests"]
addopts      = "-v --tb=short"
filterwarnings = ["ignore::DeprecationWarning"]

[tool.coverage.run]
source   = ["src/jax_gemini"]
omit     = ["*/tests/*", "*/__init__.py"]

[tool.coverage.report]
fail_under = 85
show_missing = true
```

### Publishing Steps

1. `python -m build` — produces `dist/jax_gemini-0.1.0.tar.gz` and `dist/jax_gemini-0.1.0-py3-none-any.whl`
2. `python -m twine check dist/*` — verify the package metadata
3. `python -m twine upload --repository testpypi dist/*` — upload to TestPyPI first
4. Test install: `pip install --index-url https://test.pypi.org/simple/ jax-gemini`
5. `python -m twine upload dist/*` — upload to real PyPI
6. Tag the release: `git tag v0.1.0 && git push --tags`

---

## 11. CI/CD Pipeline

### `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Lint with ruff
        run: ruff check src/ tests/

      - name: Type check with mypy
        run: mypy src/jax_gemini/

      - name: Run tests with coverage
        run: |
          pytest tests/ -v \
            --cov=src/jax_gemini \
            --cov-report=term-missing \
            --cov-fail-under=85

      - name: Verify package builds
        run: |
          pip install hatch
          hatch build
          pip install twine
          twine check dist/*

      - name: Security audit
        run: |
          pip install pip-audit
          pip-audit
```

### `.github/workflows/publish.yml`

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - "v*"

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write  # OIDC trusted publishing — no API key needed

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install build tools
        run: pip install hatch twine

      - name: Build package
        run: hatch build

      - name: Check package
        run: twine check dist/*

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

---

## 12. Documentation Standards

### README.md Structure

The README must contain these sections in this order:

1. **Badge row** — PyPI version, Python versions, License, CI status, Downloads
2. **One-line description**
3. **Quick demo** — 5 lines of code, no setup, just the magic
4. **Installation** — `pip install jax-gemini` and optional extras
5. **Usage** — Build, Train, Evaluate, Save/Load, Multi-turn conversation
6. **How it works** — One paragraph explaining the generate → validate → execute loop
7. **Configuration** — All config options in a table
8. **Security** — What the sandbox does and does not allow
9. **Contributing** — Link to CONTRIBUTING.md
10. **License**

### The Quick Demo (must be exactly this in the README)

```python
import jax_gemini as jg

jg.config.set({"gemini_api_key": "YOUR_KEY"})

# Build a model from natural language
model = jg.build("Build a 4-layer MLP for handwritten digit classification")

# Train it on your data
model, metrics = jg.train("Train for 10 epochs with Adam", dataset=(X_train, y_train))
print(f"Accuracy: {metrics['accuracy']:.2%}")   # Accuracy: 94.30%

# Refine the architecture conversationally
model = jg.modify("Add dropout with rate 0.2 between each layer")

# Save the checkpoint
jg.save("digit_classifier_v1")
```

### CHANGELOG.md Format

Follow Keep a Changelog format (https://keepachangelog.com):

```markdown
# Changelog

## [Unreleased]

## [0.1.0] - 2026-04-22

### Added
- `jg.build()` — natural language model building using Flax NNX
- `jg.train()` — natural language training loop generation
- AST-based code validation with import whitelist
- Sandboxed execution environment
- 3-attempt self-healing retry loop
- Conversation memory for multi-turn refinement
- PyPI package: `pip install jax-gemini`
```

---

## 13. Error Handling Contract

### User-Facing Error Messages

Every error that can reach the user must follow this format:

```
JaxGeminiXxxError: <What went wrong in plain English>
  <What the user should do to fix it>
  <Optional: relevant context (code, config key, etc.)>
```

**Good:**
```
JaxGeminiConfigError: gemini_api_key is not set.
  Call jg.config.set({"gemini_api_key": "YOUR_KEY"}) before using jax-gemini,
  or set the GEMINI_API_KEY environment variable.
```

**Bad:**
```
KeyError: 'gemini_api_key'
```

### Error Propagation Rules

- Errors from `google.generativeai` → always wrap in `JaxGeminiLLMError`
- Errors from `ast.parse()` → always wrap in `JaxGeminiValidationError`
- Errors from `exec()` execution → always wrap in `JaxGeminiExecutionError`
- `JaxGeminiExecutionError` after all retries → include attempt count and last code
- Config errors → raise at the earliest possible point, not deep in the stack

### Retry Transparency

When `verbose=True`, print to stdout (not stderr) at each retry:

```
[jax-gemini] Attempt 1/3 for intent 'build'...
[jax-gemini] Attempt 1 failed: JaxGeminiValidationError: Import 'os' is not allowed.
[jax-gemini] Sending error context to Gemini for correction...
[jax-gemini] Attempt 2/3 for intent 'build'...
[jax-gemini] Attempt 2 succeeded.
```

---

## 14. Definition of Done

A phase is done when ALL of the following are true:

### Code Quality
- [ ] All files pass `ruff check` with zero errors
- [ ] All files pass `mypy` with zero errors
- [ ] All public APIs have docstrings with Args, Returns, and Raises sections
- [ ] No `TODO` or `FIXME` comments remain in committed code

### Tests
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Coverage meets the phase threshold (`--cov-fail-under` passes)
- [ ] No tests are skipped with `@pytest.mark.skip` without a documented reason
- [ ] Integration tests use mocked Gemini — no real API calls

### Security
- [ ] `pip-audit` reports zero known vulnerabilities
- [ ] `CodeValidator` covers all banned patterns
- [ ] No API keys appear in any committed file or test fixture

### Package
- [ ] `python -m build` completes without errors
- [ ] `twine check dist/*` passes
- [ ] `pip install dist/jax_gemini-*.whl` works in a clean venv
- [ ] `import jax_gemini as jg; jg.config.set(...)` works after install

### CI
- [ ] GitHub Actions CI passes on all three Python versions
- [ ] All status checks are green before merging to `main`

### Documentation (Phase 6 only)
- [ ] README has the quick demo that actually runs
- [ ] `mkdocs build` completes without errors
- [ ] Every example in `examples/` runs end-to-end with a real API key

---

*End of AGENTS.md*

> This document is the single source of truth for jax-gemini implementation.
> All architectural decisions are recorded here.
> When making a decision not covered by this document, add it here before implementing.
```
