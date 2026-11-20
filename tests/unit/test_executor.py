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

    def test_numpy_allow_pickle_and_path_traversal_blocked(self, tmp_path):
        import numpy as np

        outside = tmp_path.parent / "outside.npy"
        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        np.save(sandbox_dir / "ok.npy", np.array([1, 2, 3]))

        ok_code = "import numpy as np\ndef load_data():\n    return np.load('ok.npy')\n"
        arr = self.executor.run(ok_code, "load_data", allowed_dirs=(sandbox_dir,))
        assert arr.tolist() == [1, 2, 3]

        traversal_code = (
            "import numpy as np\n"
            "def load_data():\n"
            f"    return np.load('{outside}')\n"
        )
        with pytest.raises(JaxGeminiExecutionError, match="escapes allowed directories"):
            self.executor.run(traversal_code, "load_data", allowed_dirs=(sandbox_dir,))

        pickle_kw_code = (
            "import numpy as np\n"
            "def load_data():\n"
            "    flag = True\n"
            "    return np.load('ok.npy', allow_pickle=flag)\n"
        )
        with pytest.raises(JaxGeminiValidationError, match="allow_pickle"):
            self.executor.run(pickle_kw_code, "load_data", allowed_dirs=(sandbox_dir,))

        pickle_pos_code = (
            "import numpy as np\n"
            "def load_data():\n"
            "    return np.load('ok.npy', None, True)\n"
        )
        with pytest.raises(JaxGeminiExecutionError, match="Pickle deserialization"):
            self.executor.run(pickle_pos_code, "load_data", allowed_dirs=(sandbox_dir,))

    def test_module_proxies_are_read_only_and_builtins_isolated(self):
        from jax_gemini.sandbox.namespace import get_execution_namespace

        poison_code = (
            "import flax.nnx as nnx\n"
            "def build_model(rngs):\n"
            "    nnx.POISON = 'bad'\n"
            "    return 1\n"
        )
        with pytest.raises(JaxGeminiExecutionError, match="read-only"):
            self.executor.run(poison_code, "build", {"rngs": None})

        ns1 = get_execution_namespace()
        ns2 = get_execution_namespace()
        assert ns1["__builtins__"] is not ns2["__builtins__"]
        ns1["__builtins__"]["MARKER"] = "leaked"
        assert "MARKER" not in ns2["__builtins__"]
        assert "MARKER" not in SAFE_BUILTINS
        with pytest.raises(TypeError):
            SAFE_BUILTINS["MARKER"] = "fail"  # type: ignore[index]

    def test_infinite_loop_times_out(self):
        loop_code = (
            "def build_model(rngs):\n"
            "    while True:\n"
            "        pass\n"
        )
        with pytest.raises(JaxGeminiExecutionError, match="timed out"):
            self.executor.run(loop_code, "build", {"rngs": None}, timeout_seconds=0.1)

    def test_infinite_loop_times_out_in_worker_thread(self):
        import concurrent.futures

        loop_code = (
            "def build_model(rngs):\n"
            "    while True:\n"
            "        pass\n"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(
                self.executor.run, loop_code, "build", {"rngs": None}, None, 0.1
            )
            with pytest.raises(JaxGeminiExecutionError, match="timed out"):
                fut.result(timeout=5.0)

    def test_safe_import_and_keyword_path_confinement(self, tmp_path):
        from jax_gemini.sandbox.namespace import _SafeModuleProxy, get_execution_namespace

        ns = get_execution_namespace(intent="load_data", allowed_dirs=(tmp_path,))
        safe_import = ns["__builtins__"]["__import__"]

        with pytest.raises(ImportError, match="not allowed"):
            safe_import("numpy.ctypeslib", None, None, ("load_library",), 0)
        with pytest.raises(ImportError, match="not allowed"):
            safe_import("numpy", None, None, ("fromfile",), 0)
        with pytest.raises(ImportError, match="not allowed"):
            safe_import("numpy", None, None, ("*",), 0)

        # Proxy does not expose _target and rejects writes
        with pytest.raises(AttributeError):
            _ = ns["np"]._target
        with pytest.raises(AttributeError):
            ns["np"].load = lambda *_a, **_k: None

        # Keyword-argument path confinement: inside tmp_path works, outside fails
        target = tmp_path / "vec.npy"
        save_kw_code = (
            "import numpy as np\n"
            "def load_data():\n"
            f"    np.save(file={str(target)!r}, arr=np.array([1, 2, 3]))\n"
            f"    return np.load(file={str(target)!r})\n"
        )
        arr = self.executor.run(save_kw_code, "load_data", allowed_dirs=(tmp_path,))
        assert arr.tolist() == [1, 2, 3]

        escape_kw_code = (
            "import numpy as np\n"
            "def load_data():\n"
            "    np.savetxt(fname='/tmp/escape.txt', X=np.array([1.0]))\n"
        )
        with pytest.raises(JaxGeminiExecutionError, match="escapes allowed directories"):
            self.executor.run(escape_kw_code, "load_data", allowed_dirs=(tmp_path,))
        assert isinstance(safe_import("flax.nnx", None, None, ("Linear",), 0), _SafeModuleProxy)

    def test_except_exception_cannot_swallow_timeout(self):
        loop_code = (
            "def build_model(rngs):\n"
            "    while True:\n"
            "        try:\n"
            "            x = 1 + 1\n"
            "        except Exception:\n"
            "            pass\n"
        )
        with pytest.raises(JaxGeminiExecutionError, match="timed out"):
            self.executor.run(loop_code, "build", {"rngs": None}, timeout_seconds=0.1)

    def test_timeout_and_namespace_direct_guards(self, tmp_path):
        from jax_gemini.sandbox.executor import _run_with_timeout
        from jax_gemini.sandbox.namespace import (
            _resolve_confined_path,
            _safe_import,
            get_execution_namespace,
        )

        with pytest.raises(ValueError, match="positive"):
            _run_with_timeout(lambda: 1, timeout_seconds=0)

        with pytest.raises(PermissionError, match="file handle"):
            _resolve_confined_path(123, (tmp_path,))  # type: ignore[arg-type]
        with pytest.raises(PermissionError, match="Disallowed sandbox path"):
            _resolve_confined_path("file:///etc/passwd", (tmp_path,))
        with pytest.raises(PermissionError, match="escapes allowed directories"):
            _resolve_confined_path("/etc/passwd", (tmp_path,))
        assert _resolve_confined_path("ok.npy", (tmp_path,)).endswith("ok.npy")

        ns = get_execution_namespace(intent="save", allowed_dirs=(tmp_path,))
        with pytest.raises(AttributeError, match="read-only"):
            del ns["np"].load
        with pytest.raises(AttributeError, match="blocked"):
            _ = ns["np"].fromfile
        with pytest.raises(ImportError, match="Relative"):
            _safe_import("numpy", level=1)
        assert callable(ns["np"].zeros)
