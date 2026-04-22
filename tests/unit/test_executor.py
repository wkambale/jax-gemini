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
        result = self.executor.run(code, "build", {"rngs": None})
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
