import flax.nnx as nnx
import pytest

from jax_gemini.exceptions import JaxGeminiExecutionError, JaxGeminiValidationError
from jax_gemini.sandbox.executor import SandboxExecutor
from jax_gemini.sandbox.namespace import SAFE_BUILTINS


class TestSandboxExecutor:
    def setup_method(self):
        self.executor = SandboxExecutor()

    def test_valid_build_code_executes(self):
        code = (
            "import flax.nnx as nnx\n"
            "def build_model(rngs):\n"
            "    return nnx.Linear(784, 10, rngs=rngs)\n"
        )
        result = self.executor.run(code, "build", {"rngs": nnx.Rngs(0)})
        assert result is not None

    def test_runtime_error_raises_execution_error(self):
        code = "def build_model(rngs):\n    raise ValueError('test error')\n"
        with pytest.raises(JaxGeminiExecutionError, match="test error"):
            self.executor.run(code, "build", {"rngs": None})

    def test_namespace_is_isolated(self):
        """Ensure exec does not leak into subsequent calls."""
        code1 = "SECRET = 'top_secret'\ndef build_model(rngs): return 42\n"
        self.executor.run(code1, "build", {"rngs": None})
        code2 = "def build_model(rngs): return SECRET\n"
        with pytest.raises(JaxGeminiExecutionError):
            self.executor.run(code2, "build", {"rngs": None})

    def test_executor_enforces_validator_directly(self):
        code = "import os\ndef build_model(rngs): return os.getcwd()\n"
        with pytest.raises(JaxGeminiValidationError, match="os"):
            self.executor.run(code, "build", {"rngs": None})

    def test_safe_builtins_excludes_getattr_type_and_blocks_os_import(self):
        assert "getattr" not in SAFE_BUILTINS
        assert "type" not in SAFE_BUILTINS
        with pytest.raises(ImportError, match="not allowed"):
            SAFE_BUILTINS["__import__"]("os")
        with pytest.raises(ImportError, match="Relative imports"):
            SAFE_BUILTINS["__import__"]("x", level=1)

    def test_class_module_builds_in_sandbox(self):
        code = (
            "import flax.nnx as nnx\n"
            "class SimpleMLP(nnx.Module):\n"
            "    def __init__(self, rngs):\n"
            "        self.l1 = nnx.Linear(4, 2, rngs=rngs)\n"
            "    def __call__(self, x):\n"
            "        return self.l1(x)\n"
            "def build_model(rngs):\n"
            "    return SimpleMLP(rngs)\n"
        )
        model = self.executor.run(code, "build", {"rngs": nnx.Rngs(0)})
        assert model is not None
