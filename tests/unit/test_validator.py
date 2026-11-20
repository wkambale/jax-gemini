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

    def test_relative_import_raises(self):
        code = "from . import secret\ndef build_model(rngs): pass"
        with pytest.raises(JaxGeminiValidationError, match="Relative imports"):
            CodeValidator.validate(code, "build")

    def test_aliased_banned_builtin_raises(self):
        code = "def build_model(rngs):\n    imp = __import__\n    return imp('os')"
        with pytest.raises(JaxGeminiValidationError, match="__import__"):
            CodeValidator.validate(code, "build")

    def test_builtins_name_access_raises(self):
        code = "def build_model(rngs):\n    return __builtins__['eval']"
        with pytest.raises(JaxGeminiValidationError, match="__builtins__"):
            CodeValidator.validate(code, "build")

    def test_getattr_name_raises(self):
        code = "def build_model(rngs):\n    return getattr(rngs, '__class__')"
        with pytest.raises(JaxGeminiValidationError, match="getattr"):
            CodeValidator.validate(code, "build")

    def test_traceback_frame_attribute_raises(self):
        code = (
            "def build_model(rngs):\n"
            "    try:\n"
            "        1 / 0\n"
            "    except Exception as e:\n"
            "        return e.__traceback__.tb_frame.f_back\n"
        )
        with pytest.raises(JaxGeminiValidationError, match="not allowed"):
            CodeValidator.validate(code, "build")

    def test_unknown_intent_fails_closed(self):
        code = "def build_model(rngs): pass"
        with pytest.raises(JaxGeminiValidationError, match="Unknown intent"):
            CodeValidator.validate(code, "unknown_intent")

    def test_top_level_side_effect_raises(self):
        code = "print('side effect')\ndef build_model(rngs): pass"
        with pytest.raises(JaxGeminiValidationError, match="Top-level statement"):
            CodeValidator.validate(code, "build")

    def test_oversized_single_line_code_raises(self):
        code = "def build_model(rngs):\n    x = '" + ("a" * 20001) + "'"
        with pytest.raises(JaxGeminiValidationError, match="maximum byte length"):
            CodeValidator.validate(code, "build")

    def test_class_definition_with_init_and_call_passes(self):
        code = (
            "import flax.nnx as nnx\n"
            "class MLP(nnx.Module):\n"
            "    def __init__(self, rngs):\n"
            "        self.linear = nnx.Linear(4, 2, rngs=rngs)\n"
            "    def __call__(self, x):\n"
            "        return self.linear(x)\n"
            "def build_model(rngs):\n"
            "    return MLP(rngs)\n"
        )
        CodeValidator.validate(code, "build")

    def test_allow_pickle_true_raises(self):
        code = (
            "import numpy as np\n"
            "def load_data():\n"
            "    return np.load('x.npy', allow_pickle=True)\n"
        )
        with pytest.raises(JaxGeminiValidationError, match="allow_pickle"):
            CodeValidator.validate(code, "load_data")

    def test_orbax_blocked_for_non_checkpoint_intent(self):
        code = "import orbax.checkpoint as ocp\ndef build_model(rngs): pass"
        with pytest.raises(JaxGeminiValidationError, match="orbax"):
            CodeValidator.validate(code, "build")

    def test_numpy_fromfile_blocked(self):
        code = "import numpy as np\ndef load_data():\n    return np.fromfile('/etc/passwd')\n"
        with pytest.raises(JaxGeminiValidationError, match="fromfile"):
            CodeValidator.validate(code, "load_data")
