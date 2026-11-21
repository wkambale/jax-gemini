import flax.nnx as nnx
import pytest

from jaxforge.exceptions import JaxForgeExecutionError, JaxForgeValidationError
from jaxforge.sandbox.executor import SandboxExecutor
from jaxforge.sandbox.namespace import SAFE_BUILTINS


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
        with pytest.raises(JaxForgeExecutionError, match="test error"):
            self.executor.run(code, "build", {"rngs": None})

    def test_namespace_is_isolated(self):
        """Ensure exec does not leak into subsequent calls."""
        code1 = "SECRET = 'top_secret'\ndef build_model(rngs): return 42\n"
        self.executor.run(code1, "build", {"rngs": None})
        code2 = "def build_model(rngs): return SECRET\n"
        with pytest.raises(JaxForgeExecutionError):
            self.executor.run(code2, "build", {"rngs": None})

    def test_executor_enforces_validator_directly(self):
        code = "import os\ndef build_model(rngs): return os.getcwd()\n"
        with pytest.raises(JaxForgeValidationError, match="os"):
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
        with pytest.raises(JaxForgeExecutionError, match="escapes allowed directories"):
            self.executor.run(traversal_code, "load_data", allowed_dirs=(sandbox_dir,))

        pickle_kw_code = (
            "import numpy as np\n"
            "def load_data():\n"
            "    flag = True\n"
            "    return np.load('ok.npy', allow_pickle=flag)\n"
        )
        with pytest.raises(JaxForgeValidationError, match="allow_pickle"):
            self.executor.run(pickle_kw_code, "load_data", allowed_dirs=(sandbox_dir,))

        pickle_pos_code = (
            "import numpy as np\n"
            "def load_data():\n"
            "    return np.load('ok.npy', None, True)\n"
        )
        with pytest.raises(JaxForgeExecutionError, match="Pickle deserialization"):
            self.executor.run(pickle_pos_code, "load_data", allowed_dirs=(sandbox_dir,))

    def test_module_proxies_are_read_only_and_builtins_isolated(self):
        from jaxforge.sandbox.namespace import get_execution_namespace

        poison_code = (
            "import flax.nnx as nnx\n"
            "def build_model(rngs):\n"
            "    nnx.POISON = 'bad'\n"
            "    return 1\n"
        )
        with pytest.raises(JaxForgeExecutionError, match="read-only"):
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
        with pytest.raises(JaxForgeExecutionError, match="timed out"):
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
            with pytest.raises(JaxForgeExecutionError, match="timed out"):
                fut.result(timeout=5.0)

    def test_safe_import_and_keyword_path_confinement(self, tmp_path):
        from jaxforge.sandbox.namespace import _SafeModuleProxy, get_execution_namespace

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
        with pytest.raises(JaxForgeExecutionError, match="escapes allowed directories"):
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
        with pytest.raises(JaxForgeExecutionError, match="timed out"):
            self.executor.run(loop_code, "build", {"rngs": None}, timeout_seconds=0.1)

    def test_timeout_and_namespace_direct_guards(self, tmp_path):
        from jaxforge.sandbox.executor import _run_with_timeout
        from jaxforge.sandbox.namespace import (
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

    def test_transitive_stdlib_and_ml_io_confinement(self, tmp_path):
        from types import ModuleType

        import jaxforge.sandbox.namespace as ns_mod
        from jaxforge.codegen.validator import CodeValidator
        from jaxforge.exceptions import JaxForgeValidationError

        assert not hasattr(ns_mod, "BASE_NAMESPACE")

        ns = ns_mod.get_execution_namespace(intent="build", allowed_dirs=(tmp_path,))
        importer = ns["__builtins__"]["__import__"]
        for mod_name in ("dataclasses", "typing", "functools"):
            mod = importer(mod_name)
            for forbidden_attr in ("sys", "os", "io", "builtins", "subprocess"):
                with pytest.raises(AttributeError, match="blocked"):
                    getattr(mod, forbidden_attr)

        # Unwhitelisted ModuleType attribute attached to a module must be blocked
        fake_host_mod = ModuleType("http.client")
        raw_target = ModuleType("math")
        raw_target.custom_leak = fake_host_mod  # type: ignore[attr-defined]
        proxy = ns_mod._SafeModuleProxy(raw_target, allowed_imports=frozenset({"math"}))
        with pytest.raises(AttributeError, match="blocked"):
            _ = proxy.custom_leak

        # jax.profiler blocked in both validator and runtime proxy
        bad_prof = "import jax.profiler\ndef build_model(rngs): return 1\n"
        with pytest.raises(JaxForgeValidationError):
            CodeValidator.validate(bad_prof, "build")
        with pytest.raises(AttributeError, match="blocked"):
            _ = ns["jax"].profiler

        # Equinox serialization blocked in build, confined in save/load
        fake_eqx = ModuleType("equinox")
        saved_paths: list[str] = []
        fake_eqx.tree_serialise_leaves = lambda path_or_file, tree=None: saved_paths.append(
            str(path_or_file)
        )  # type: ignore[attr-defined]
        fake_eqx.tree_deserialise_leaves = lambda path_or_file, like=None: str(path_or_file)  # type: ignore[attr-defined]

        eqx_build = ns_mod._create_safe_equinox(
            fake_eqx, (tmp_path,), "build", frozenset({"equinox"}), {}
        )
        with pytest.raises(AttributeError, match="blocked"):
            _ = eqx_build.tree_serialise_leaves

        eqx_save = ns_mod._create_safe_equinox(
            fake_eqx, (tmp_path,), "save", frozenset({"equinox"}), {}
        )
        eqx_save.tree_serialise_leaves("ok.eqx", {})
        assert saved_paths and saved_paths[0].endswith("ok.eqx")
        with pytest.raises(PermissionError, match="escapes"):
            eqx_save.tree_serialise_leaves("/etc/passwd", {})

        eqx_load = ns_mod._create_safe_equinox(
            fake_eqx, (tmp_path,), "load", frozenset({"equinox"}), {}
        )
        assert eqx_load.tree_deserialise_leaves(path_or_file="ok.eqx", like={}).endswith("ok.eqx")
        with pytest.raises(AttributeError, match="blocked"):
            _ = eqx_load.tree_serialise_leaves

        # Safetensors confines safe_open and blocks save_file during load
        fake_st = ModuleType("safetensors")
        fake_st.safe_open = lambda filename, *a, **kw: filename  # type: ignore[attr-defined]
        fake_st_np = ModuleType("safetensors.numpy")
        fake_st_np.load_file = lambda filename: filename  # type: ignore[attr-defined]
        fake_st_np.save_file = lambda filename: filename  # type: ignore[attr-defined]
        fake_st_flax = ModuleType("safetensors.flax")
        fake_st_flax.load_file = lambda filename: filename  # type: ignore[attr-defined]
        fake_st_flax.save_file = lambda filename: filename  # type: ignore[attr-defined]

        safe_st = ns_mod._create_safe_safetensors(
            fake_st,
            fake_st_np,
            fake_st_flax,
            (tmp_path,),
            "load",
            frozenset({"safetensors"}),
            {},
        )
        assert safe_st.safe_open("weights.safetensors").endswith("weights.safetensors")
        with pytest.raises(PermissionError, match="escapes"):
            safe_st.safe_open("/etc/passwd")
        with pytest.raises(AttributeError, match="blocked"):
            _ = safe_st.numpy.save_file

        safe_st_save = ns_mod._create_safe_safetensors(
            fake_st,
            fake_st_np,
            fake_st_flax,
            (tmp_path,),
            "save",
            frozenset({"safetensors"}),
            {},
        )
        assert safe_st_save.flax.save_file(filename="w.safetensors").endswith("w.safetensors")

        # Grain confines file data sources (including grain.python and keyword args)
        fake_grain = ModuleType("grain")
        fake_grain_py = ModuleType("grain.python")
        fake_grain_py.ArrayRecordDataSource = lambda paths: paths  # type: ignore[attr-defined]
        fake_grain.python = fake_grain_py  # type: ignore[attr-defined]
        fake_grain.ParquetDataSource = lambda paths: paths  # type: ignore[attr-defined]
        fake_grain.multiprocessing = ModuleType("grain.multiprocessing")  # type: ignore[attr-defined]
        safe_grain = ns_mod._create_safe_grain(
            fake_grain, (tmp_path,), frozenset({"grain"}), {}
        )
        assert safe_grain.ParquetDataSource(["a.parquet"])[0].endswith("a.parquet")
        assert safe_grain.python.ArrayRecordDataSource(paths="b.arecord").endswith("b.arecord")
        with pytest.raises(PermissionError, match="sequence of paths"):
            safe_grain.ParquetDataSource(123)
        with pytest.raises(PermissionError, match="confined path"):
            safe_grain.ParquetDataSource(other=1)
        with pytest.raises(PermissionError, match="escapes"):
            safe_grain.ParquetDataSource(["/etc/passwd"])
        with pytest.raises(AttributeError, match="blocked"):
            _ = safe_grain.multiprocessing

        # Exercise get_execution_namespace optional library branches
        from unittest.mock import patch

        with patch.multiple(
            ns_mod,
            _equinox_available=True,
            eqx=fake_eqx,
            _grain_available=True,
            grain=fake_grain,
            _safetensors_available=True,
            safetensors=fake_st,
            st_np=fake_st_np,
            st_flax=fake_st_flax,
        ):
            ns_load = ns_mod.get_execution_namespace(intent="load", allowed_dirs=(tmp_path,))
            assert "eqx" in ns_load and "grain" in ns_load and "safetensors" in ns_load
