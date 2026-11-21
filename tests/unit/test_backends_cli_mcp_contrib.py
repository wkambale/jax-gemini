import json
import sys
import warnings

import jax
import jax.numpy as jnp
import pytest

from jaxforge.cli import main as cli_main
from jaxforge.contrib.flax import eval_shape_summary
from jaxforge.contrib.optax import create_cosine_schedule
from jaxforge.contrib.sharding import (
    create_device_mesh,
    device_summary,
    make_data_sharding,
    make_replicated_sharding,
)
from jaxforge.contrib.tracking import InMemoryTracker, JsonlTracker, create_tracker
from jaxforge.exceptions import JaxForgeConfigError
from jaxforge.mcp_server import tool_device_summary, tool_run_sandboxed_jax, tool_validate_jax_code
from jaxforge.sandbox.backends import (
    DockerSandboxBackend,
    E2BSandboxBackend,
    ModalSandboxBackend,
    SubprocessSandboxBackend,
    create_backend,
)


class TestBackendsCliMcpContrib:
    def test_subprocess_docker_e2b_modal_backends(self, tmp_path):
        import pickle
        from types import ModuleType
        from unittest.mock import MagicMock, patch

        import numpy as np

        from jaxforge.config import JaxForgeConfig
        from jaxforge.exceptions import JaxForgeExecutionError
        from jaxforge.sandbox.backends import (
            _build_self_contained_runner,
            _decode_stdout_result,
            _encode_remote_payload,
            _safe_pickle_loads,
        )

        sub = SubprocessSandboxBackend()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = sub.execute(
                code=(
                    "import flax.nnx as nnx\n"
                    "class CustomMLP(nnx.Module):\n"
                    "    def __init__(self, rngs):\n"
                    "        self.l = nnx.Linear(4, 2, rngs=rngs)\n"
                    "def build_model(rngs):\n"
                    "    return CustomMLP(rngs)\n"
                ),
                intent="build",
                fn_name="build_model",
                call_args={"rngs": __import__("flax.nnx").nnx.Rngs(0)},
                allowed_dirs=(tmp_path,),
                timeout_seconds=15.0,
                max_memory_mb=2048,
            )
        assert type(res).__name__ == "CustomMLP"

        # Restricted unpickler blocks host callables such as os.system
        malicious_bytes = b"cos\nsystem\n(S'id'\ntR."
        with pytest.raises(JaxForgeExecutionError, match="Blocked unpickling"):
            _safe_pickle_loads(malicious_bytes)

        # NNX leaves envelope rehydrates cleanly on closure-heavy Flax versions (e.g. Python 3.10)
        nnx_mod = __import__("flax.nnx").nnx
        leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(nnx_mod.state(res))]
        env_bytes = pickle.dumps(("__jaxforge_nnx_leaves__", leaves))
        rehydrated = _safe_pickle_loads(
            env_bytes,
            code=(
                "import flax.nnx as nnx\n"
                "def build_model(rngs): return nnx.Linear(4, 2, rngs=rngs)\n"
            ),
            intent="build",
        )
        assert isinstance(rehydrated, nnx_mod.Module)

        # Self-contained remote runner handles None, bool, and NumPy arrays and returns real types
        train_code = (
            "def train_model(dataset=None, epochs=5, learning_rate=0.001):\n"
            "    return ('trained', {'loss': 0.1, 'seen_none': dataset is None, 'ep': epochs})\n"
        )
        b64 = _encode_remote_payload(
            train_code, "train", "train_model", {"dataset": None, "epochs": 3}
        )
        script = _build_self_contained_runner(b64, out_file=None)
        proc = __import__("subprocess").run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        decoded = _decode_stdout_result(proc.stdout, code=train_code, intent="train")
        assert decoded == ("trained", {"loss": 0.1, "seen_none": True, "ep": 3})

        # Docker command includes /tmp tmpfs, custom docker_image, and executes via runner
        cfg = JaxForgeConfig(docker_image="my-org/jaxforge:0.2")
        assert cfg.docker_image == "my-org/jaxforge:0.2"
        with pytest.raises(JaxForgeConfigError, match="docker_image"):
            cfg.set({"docker_image": "-bad-flag"})

        docker_be = create_backend("docker", docker_image="my-org/jaxforge:0.2")
        assert isinstance(docker_be, DockerSandboxBackend)
        docker_cmd = docker_be.build_command(tmp_path, 30, 1024, allowed_dirs=(tmp_path,))
        assert "--network=none" in docker_cmd
        assert "--tmpfs" in docker_cmd
        assert "my-org/jaxforge:0.2" in docker_cmd

        real_subproc_run = __import__("subprocess").run

        def _fake_docker_run(cmd, **kwargs):
            pathlib_mod = __import__("pathlib")
            vol_arg = next(a for a in cmd if a.endswith(":/workspace:rw"))
            host_ws = pathlib_mod.Path(vol_arg.split(":/workspace:rw")[0])
            out_target = repr(str(host_ws / "out.pkl"))
            py_src = (
                (host_ws / "run_sandbox.py")
                .read_text(encoding="utf-8")
                .replace("'/workspace/out.pkl'", out_target)
            )
            real_subproc_run([sys.executable, "-c", py_src], check=True)
            return MagicMock(returncode=0, stderr="")

        with patch("jaxforge.sandbox.backends.subprocess.run", side_effect=_fake_docker_run):
            d_res = docker_be.execute(
                train_code,
                "train",
                "train_model",
                {"dataset": np.ones((2, 2)), "epochs": 2},
                (tmp_path,),
                10.0,
                512,
            )
            assert d_res[0] == "trained" and d_res[1]["seen_none"] is False

        # Missing optional packages raise JaxForgeConfigError cleanly
        with pytest.raises(JaxForgeConfigError, match="e2b"):
            E2BSandboxBackend().execute(
                "def build_model(): pass", "build", "build_model", {}, None, 5
            )
        with pytest.raises(JaxForgeConfigError, match="modal"):
            ModalSandboxBackend().execute(
                "def build_model(): pass", "build", "build_model", {}, None, 5
            )

        # Mocked E2B and Modal backends execute self-contained runner and deserialize objects
        fake_e2b = ModuleType("e2b_code_interpreter")

        class _FakeE2BSandbox:
            def __init__(self, timeout=5):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def run_code(self, code_str):
                out = __import__("subprocess").run(
                    [sys.executable, "-c", code_str], capture_output=True, text=True, check=True
                )
                return MagicMock(error=None, logs=MagicMock(stdout=[out.stdout.strip()]), text="")

        fake_e2b.Sandbox = _FakeE2BSandbox  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"e2b_code_interpreter": fake_e2b}):
            e2b_out = E2BSandboxBackend().execute(
                train_code, "train", "train_model", {"dataset": None, "epochs": 4}, None, 10
            )
            assert e2b_out == ("trained", {"loss": 0.1, "seen_none": True, "ep": 4})

        fake_modal = ModuleType("modal")
        created_kwargs: dict[str, object] = {}

        def _fake_modal_create(*args, **kwargs):
            created_kwargs.update(kwargs)
            out = __import__("subprocess").run(
                [sys.executable, "-c", args[2]], capture_output=True, text=True, check=True
            )
            sb = MagicMock(returncode=0)
            sb.stdout.read.return_value = out.stdout
            return sb

        fake_modal.App = MagicMock()  # type: ignore[attr-defined]
        fake_modal.Image = MagicMock()  # type: ignore[attr-defined]
        fake_modal.Sandbox = MagicMock(create=_fake_modal_create)  # type: ignore[attr-defined]
        ModalSandboxBackend._cached_image = None
        with patch.dict(sys.modules, {"modal": fake_modal}):
            modal_out = ModalSandboxBackend().execute(
                train_code, "train", "train_model", {"dataset": None, "epochs": 7}, None, 10
            )
            assert modal_out == ("trained", {"loss": 0.1, "seen_none": True, "ep": 7})
            assert created_kwargs.get("block_network") is True
            assert isinstance(pickle.dumps(modal_out), bytes)

        with pytest.raises(JaxForgeConfigError, match="Unknown execution_backend"):
            create_backend("invalid")

    def test_sharding_eval_shape_schedule_and_tracking(self, tmp_path):
        summary = device_summary()
        assert summary["device_count"] >= 1
        mesh = create_device_mesh(("data",))
        assert make_replicated_sharding(mesh).spec == jax.sharding.PartitionSpec()
        assert make_data_sharding(mesh, "data").spec == jax.sharding.PartitionSpec("data")

        num_devs = len(jax.devices())
        mesh_explicit = create_device_mesh(("data", "model"), mesh_shape=(num_devs, 1))
        assert mesh_explicit.axis_names == ("data", "model")
        with pytest.raises(ValueError, match="non-empty"):
            create_device_mesh(())
        with pytest.raises(ValueError, match="non-empty strings"):
            create_device_mesh((" ",))
        with pytest.raises(ValueError, match="unique"):
            create_device_mesh(("data", "data"))
        with pytest.raises(ValueError, match="length"):
            create_device_mesh(("data",), mesh_shape=(num_devs, 1))
        with pytest.raises(ValueError, match="positive integers"):
            create_device_mesh(("data",), mesh_shape=(True,))
        with pytest.raises(ValueError, match="positive integers"):
            create_device_mesh(("data",), mesh_shape=(0,))
        with pytest.raises(ValueError, match="prod"):
            create_device_mesh(("data",), mesh_shape=(num_devs + 1,))

        shape_info = eval_shape_summary(
            lambda x: x @ jnp.ones((4, 2)),
            jax.ShapeDtypeStruct((3, 4), jnp.float32),
        )
        assert shape_info["outputs"][0]["shape"] == [3, 2]

        sched = create_cosine_schedule(peak_lr=1e-2, warmup_steps=5, total_steps=50)
        assert float(sched(0)) == 0.0
        assert float(sched(5)) == pytest.approx(1e-2)
        for bad_lr in (True, False, 0.0, -1e-3, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="peak_lr"):
                create_cosine_schedule(peak_lr=bad_lr)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="integers"):
            create_cosine_schedule(peak_lr=1e-3, warmup_steps=True, total_steps=10)
        with pytest.raises(ValueError, match="0 <= warmup_steps < total_steps"):
            create_cosine_schedule(peak_lr=1e-3, warmup_steps=10, total_steps=10)

        mem = create_tracker("memory")
        assert isinstance(mem, InMemoryTracker)
        mem.log_params({"lr": 1e-3})
        mem.log_metrics({"loss": 0.2}, step=1)
        assert len(mem.records) == 1

        jsonl_path = tmp_path / "metrics.jsonl"
        jl = create_tracker("jsonl", jsonl_path)
        assert isinstance(jl, JsonlTracker)
        jl.log_metrics({"acc": 0.9}, step=1)
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["data"]["acc"] == 0.9

    def test_cli_and_mcp_tools(self, tmp_path, capsys):
        assert cli_main(["info"]) == 0
        info_out = json.loads(capsys.readouterr().out)
        assert info_out["version"] == "0.2.0"
        assert "anthropic" in info_out["providers"]

        assert cli_main(["devices"]) == 0
        dev_out = json.loads(capsys.readouterr().out)
        assert dev_out["device_count"] >= 1

        sample = tmp_path / "sample.py"
        sample.write_text("import flax.nnx as nnx\ndef build_model(rngs):\n    return 1\n")
        assert cli_main(["validate", str(sample), "--intent", "build"]) == 0
        capsys.readouterr()

        # CLI catches FileNotFoundError, JaxForgeValidationError, and EOFError without tracebacks
        assert cli_main(["validate", str(tmp_path / "missing.py")]) == 1
        err_missing = capsys.readouterr().err
        assert "Error:" in err_missing and "Traceback" not in err_missing

        bad_file = tmp_path / "bad.py"
        bad_file.write_text("import os\ndef build_model(rngs):\n    return 1\n")
        assert cli_main(["validate", str(bad_file), "--intent", "build"]) == 1
        err_bad = capsys.readouterr().err
        assert "Error:" in err_bad and "Traceback" not in err_bad

        from unittest.mock import patch

        from jaxforge.plan import GeneratedPlan

        dummy_plan = GeneratedPlan(
            intent="build",
            code=(
                "import flax.nnx as nnx\n"
                "def build_model(rngs):\n"
                "    return nnx.Linear(2, 1, rngs=rngs)\n"
            ),
            description="tiny",
            provider="google",
            model_name="gemini-3.1-pro",
        )
        # Missing API key in run exits cleanly with status 1 instead of traceback
        assert cli_main(["run", "build tiny model"]) == 1
        err_key = capsys.readouterr().err
        assert "Error:" in err_key and "Traceback" not in err_key

        import os
        from unittest.mock import MagicMock

        fake_llm = MagicMock()
        fake_llm.generate.return_value = {
            "intent": "build",
            "code": dummy_plan.code,
            "description": "tiny",
        }
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("jaxforge.agent.create_llm", return_value=fake_llm),
        ):
            with patch("builtins.input", side_effect=EOFError):
                assert cli_main(["run", "build tiny model"]) == 1
                err_eof = capsys.readouterr().err
                assert "Aborted." in err_eof

            assert cli_main(["run", "build tiny model", "--yes", "--model", "gemini-3.1-pro"]) == 0
            capsys.readouterr()
            export_out = tmp_path / "cli_exported"
            assert cli_main(["preview", "build tiny model", "--export", str(export_out)]) == 0
            capsys.readouterr()
            assert (export_out / "pyproject.toml").is_file()

        assert tool_validate_jax_code("def build_model(rngs): return 1")["valid"] is True
        bad = "import os\ndef build_model(rngs): return 1"
        assert tool_validate_jax_code(bad)["valid"] is False
        assert tool_device_summary()["device_count"] >= 1
        exec_res = tool_run_sandboxed_jax(
            "import numpy as np\n"
            "def analyze_data(dataset=None):\n"
            "    return int(np.sum([1, 2, 3]))\n"
        )
        assert exec_res["ok"] is True
        assert exec_res["result_repr"] == "6"

        # MCP bounds timeout, rejects save intent, isolates CWD, and respects auto_approve
        neg_res = tool_run_sandboxed_jax(
            "def analyze_data(dataset=None): return 1", timeout_seconds=-1
        )
        assert neg_res["ok"] is False
        assert tool_run_sandboxed_jax(
            "def analyze_data(dataset=None): return 1", timeout_seconds=999
        )["ok"] is False
        assert tool_run_sandboxed_jax(
            "def save_model(model=None, path='x'): return 'x'", intent="save"
        )["ok"] is False
        cwd_secret = tmp_path / "secret.npy"
        import numpy as np

        np.save(cwd_secret, np.array([42]))
        leak_attempt = tool_run_sandboxed_jax(
            f"import numpy as np\ndef load_data():\n    return np.load({str(cwd_secret)!r})\n",
            intent="load_data",
        )
        assert leak_attempt["ok"] is False

        # Exporter escapes quotes/triple-quotes in description and blocks keyword package_name
        import ast

        from jaxforge.exporter import export_project

        with pytest.raises(ValueError, match="non-keyword"):
            export_project(
                tmp_path / "bad_kw",
                "def build_model(rngs): return 1\n",
                package_name="import",
            )

        tricky_desc = 'Bad """\nimport os; os.system("id")\n[tool.evil]\nx = "injected'
        pkg_root = export_project(
            tmp_path / "safe_pkg",
            model_code="import equinox as eqx\ndef build_model(key):\n    return 1\n",
            package_name="safe_pkg",
            description=tricky_desc,
            framework="equinox",
        )
        init_src = (pkg_root / "src" / "safe_pkg" / "__init__.py").read_text(encoding="utf-8")
        init_tree = ast.parse(init_src)
        assert len(init_tree.body) == 3  # docstring Expr, ImportFrom, __all__ Assign only!
        toml_src = (pkg_root / "pyproject.toml").read_text(encoding="utf-8")
        assert "\n[tool.evil]" not in toml_src
        assert "equinox>=0.11.0" in toml_src

    def test_jax_gemini_compatibility_shim(self):
        import jaxforge as jf
        from jaxforge.plan import GeneratedPlan

        assert callable(jf.execute_plan)
        assert "execute_plan" in jf.__all__
        plan = GeneratedPlan(
            intent="analyze_data",
            code="def analyze_data(dataset=None):\n    return {'rows': 3}\n",
            description="analyze",
            provider="google",
            model_name="gemini-3.1-pro",
        )
        assert jf.execute_plan(plan) == {"rows": 3}

        sys.modules.pop("jax_gemini", None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import jax_gemini
            from jax_gemini.codegen.validator import CodeValidator
            from jax_gemini.contrib import create_optimizer
            from jax_gemini.contrib.flax import count_parameters
            from jax_gemini.exceptions import JaxGeminiApprovalError
            from jax_gemini.llm.gemini import GeminiLLM
            from jax_gemini.memory.conversation import ConversationMemory
            from jax_gemini.pipeline import ModelBuilder, ModelTrainer
            from jax_gemini.pipeline.builder import ModelBuilder as SubModelBuilder
            from jax_gemini.prompts.intent import IntentClassifier
            from jax_gemini.sandbox.executor import SandboxExecutor

            assert any(issubclass(w.category, DeprecationWarning) for w in caught)
            assert jax_gemini.JaxGemini.__name__ == "JaxForge"
            assert jax_gemini.JaxGeminiApprovalError is JaxGeminiApprovalError
            assert JaxGeminiApprovalError is jf.JaxForgeApprovalError
            assert jax_gemini.execute_plan is jf.execute_plan
            assert ModelBuilder is SubModelBuilder
            assert all(
                obj is not None
                for obj in (
                    CodeValidator,
                    create_optimizer,
                    count_parameters,
                    GeminiLLM,
                    ConversationMemory,
                    ModelTrainer,
                    IntentClassifier,
                    SandboxExecutor,
                )
            )
