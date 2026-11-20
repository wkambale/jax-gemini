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
        sub = SubprocessSandboxBackend()
        res = sub.execute(
            code="def build_model():\n    return {'layers': [4, 2]}\n",
            intent="build",
            fn_name="build_model",
            call_args={},
            allowed_dirs=(tmp_path,),
            timeout_seconds=10.0,
            max_memory_mb=None,
        )
        assert res == {"layers": [4, 2]}

        docker_cmd = DockerSandboxBackend().build_command(tmp_path, 30, 1024)
        assert "--network=none" in docker_cmd
        assert "--memory=1024m" in docker_cmd

        with pytest.raises(JaxForgeConfigError, match="e2b"):
            E2BSandboxBackend().execute(
                "def build_model(): pass", "build", "build_model", {}, None, 5
            )
        with pytest.raises(JaxForgeConfigError, match="modal"):
            ModalSandboxBackend().execute(
                "def build_model(): pass", "build", "build_model", {}, None, 5
            )
        with pytest.raises(JaxForgeConfigError, match="Unknown execution_backend"):
            create_backend("invalid")

    def test_sharding_eval_shape_schedule_and_tracking(self, tmp_path):
        summary = device_summary()
        assert summary["device_count"] >= 1
        mesh = create_device_mesh(("data",))
        assert make_replicated_sharding(mesh).spec == jax.sharding.PartitionSpec()
        assert make_data_sharding(mesh, "data").spec == jax.sharding.PartitionSpec("data")

        shape_info = eval_shape_summary(
            lambda x: x @ jnp.ones((4, 2)),
            jax.ShapeDtypeStruct((3, 4), jnp.float32),
        )
        assert shape_info["outputs"][0]["shape"] == [3, 2]

        sched = create_cosine_schedule(peak_lr=1e-2, warmup_steps=5, total_steps=50)
        assert float(sched(0)) == 0.0
        assert float(sched(5)) == pytest.approx(1e-2)

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

    def test_jax_gemini_compatibility_shim(self):
        sys.modules.pop("jax_gemini", None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import jax_gemini

            assert any(issubclass(w.category, DeprecationWarning) for w in caught)
            assert jax_gemini.JaxGemini.__name__ == "JaxForge"
