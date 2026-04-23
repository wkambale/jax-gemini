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
