import json
import types
from unittest.mock import MagicMock, patch

import pytest

import jax_gemini as jg
from jax_gemini.exceptions import (
    JaxGeminiExecutionError,
    JaxGeminiLLMError,
    JaxGeminiValidationError,
)
from jax_gemini.sandbox import namespace as ns_mod
from jax_gemini.sandbox.executor import _run_with_timeout, _SandboxTimeoutError


class TestAgentAndNamespaceCoverage:
    def test_run_with_timeout_handles_blocked_c_sleep(self):
        import time

        with pytest.raises(_SandboxTimeoutError, match="timed out"):
            _run_with_timeout(lambda: time.sleep(0.8), timeout_seconds=0.05)

    def test_namespace_orbax_pyplot_and_import_guards(self, tmp_path, monkeypatch):
        # Empty path, null byte, and cwd default fallback
        with pytest.raises(PermissionError, match="Disallowed sandbox path"):
            ns_mod._resolve_confined_path("", (tmp_path,))
        with pytest.raises(PermissionError, match="Disallowed sandbox path"):
            ns_mod._resolve_confined_path("bad\x00path", (tmp_path,))
        # Empty allowed_dirs fails closed
        with pytest.raises(PermissionError, match="escapes allowed directories"):
            ns_mod._resolve_confined_path("rel.npy", ())

        # Positional allow_pickle=True and missing path args in _wrap_path_io
        ns = ns_mod.get_execution_namespace(intent="load_data", allowed_dirs=(tmp_path,))
        with pytest.raises(PermissionError, match="Pickle deserialization"):
            ns["np"].load(str(tmp_path / "x.npy"), None, True)
        with pytest.raises(PermissionError, match="confined path argument"):
            ns["np"].save()

        # Submodule attribute via proxy (np.random -> _SafeModuleProxy)
        assert isinstance(ns["np"].random, ns_mod._SafeModuleProxy)
        # Disallowed import via _safe_import & dotted root import
        safe_import = ns["__builtins__"]["__import__"]
        assert isinstance(safe_import("math"), ns_mod._SafeModuleProxy)
        assert isinstance(safe_import("jax.numpy"), ns_mod._SafeModuleProxy)
        with pytest.raises(ImportError, match="not allowed"):
            safe_import("os")
        with pytest.raises(ImportError, match="not allowed"):
            safe_import("jax.numpy", fromlist=("__globals__",))

        # Orbax confined checkpointer & CheckpointManager & blocked internals
        save_ns = ns_mod.get_execution_namespace(intent="save", allowed_dirs=(tmp_path,))
        assert "ocp" in save_ns
        with pytest.raises(AttributeError, match="blocked"):
            _ = save_ns["ocp"].test_utils
        ckpt = save_ns["ocp"].StandardCheckpointer()
        with pytest.raises(AttributeError, match="blocked"):
            _ = ckpt._inner
        outside = tmp_path.parent / "outside_dir"
        with pytest.raises(PermissionError, match="escapes allowed directories"):
            ckpt.save(str(outside), {"a": 1})
        with pytest.raises(PermissionError, match="escapes allowed directories"):
            ckpt.restore(str(outside))
        with pytest.raises(PermissionError, match="escapes allowed directories"):
            save_ns["ocp"].CheckpointManager(str(outside))

        # Confine figure wrapper & fake pyplot proxy
        saved_paths: list[str] = []

        class _FakeFig:
            def savefig(self, fname, **_kwargs):
                saved_paths.append(fname)

        fake_plt = types.ModuleType("matplotlib.pyplot")
        fake_plt.savefig = lambda fname, **_kw: saved_paths.append(fname)  # type: ignore[attr-defined]
        fake_plt.imread = lambda fname, **_kw: fname  # type: ignore[attr-defined]
        fake_plt.imsave = lambda fname, *_a, **_kw: saved_paths.append(fname)  # type: ignore[attr-defined]
        fake_plt.figure = lambda *_a, **_kw: _FakeFig()  # type: ignore[attr-defined]
        fake_plt.subplots = lambda *_a, **_kw: (_FakeFig(), "ax")  # type: ignore[attr-defined]
        fake_plt.switch_backend = lambda *_a: None  # type: ignore[attr-defined]
        fake_mpl = types.ModuleType("matplotlib")
        fake_mpl.pyplot = fake_plt  # type: ignore[attr-defined]

        monkeypatch.setattr(ns_mod, "_matplotlib_available", True)
        monkeypatch.setattr(ns_mod, "plt", fake_plt, raising=False)
        monkeypatch.setattr(ns_mod, "matplotlib", fake_mpl, raising=False)

        viz_ns = ns_mod.get_execution_namespace(intent="visualize_data", allowed_dirs=(tmp_path,))
        fig, _ = viz_ns["plt"].subplots()
        fig.savefig("chart.png")
        assert saved_paths[-1] == str((tmp_path / "chart.png").resolve())
        with pytest.raises(PermissionError, match="escapes allowed directories"):
            fig.savefig(str(outside / "fig.png"))
        with pytest.raises(PermissionError, match="escapes allowed directories"):
            viz_ns["plt"].savefig(str(outside / "plt.png"))
        assert isinstance(viz_ns["plt"].figure(), _FakeFig)

    @patch("jax_gemini.llm.gemini.genai")
    def test_agent_data_methods_explain_show_code_and_retry_exhaustion(self, mock_genai):
        agent = jg.JaxGemini(jg.JaxGeminiConfig(gemini_api_key="test-key", max_retries=2))
        assert "No code has been generated yet" in agent.explain()
        assert "No code has been generated yet" in agent.show_code()

        def _resp(intent: str, code: str, description: str | None = "ok"):
            payload = {"intent": intent, "code": code, "warnings": []}
            if description is not None:
                payload["description"] = description
            else:
                payload["description"] = "generated"
            m = MagicMock()
            m.text = json.dumps(payload)
            return m

        mock_genai.GenerativeModel.return_value.generate_content.side_effect = [
            _resp("load_data", "def load_data():\n    return [1, 2]\n", "loaded data"),
            _resp(
                "preprocess_data",
                "def preprocess_data(dataset):\n    return [x * 2 for x in dataset]\n",
            ),
            _resp(
                "analyze_data",
                "def analyze_data(dataset):\n    return {'count': len(dataset)}\n",
            ),
            _resp(
                "visualize_data",
                "def visualize_data(dataset):\n    return 'fig'\n",
            ),
        ]

        data = agent.load_data("load data")
        assert data == [1, 2]
        assert agent.explain() == "loaded data"
        assert "def load_data" in agent.show_code()

        assert agent.preprocess_data("preprocess", data) == [2, 4]
        assert agent.analyze_data("analyze", data) == {"count": 2}
        assert agent.visualize_data("plot", data) == "fig"

        # Retry exhaustion raises JaxGeminiLLMError
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = [
            MagicMock(text="{bad"),
            MagicMock(text="{still bad"),
        ]
        with pytest.raises(JaxGeminiLLMError):
            agent.build("Build model")

        # Retry exhaustion raises JaxGeminiValidationError
        bad_val = _resp("build", "import os\ndef build_model(rngs): return 1\n")
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = [bad_val, bad_val]
        with pytest.raises(JaxGeminiValidationError):
            agent.build("Build model")

        # Retry exhaustion raises JaxGeminiExecutionError
        bad_exec = _resp("build", "def build_model(rngs):\n    raise ValueError('boom')\n")
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = [
            bad_exec,
            bad_exec,
        ]
        with pytest.raises(JaxGeminiExecutionError, match="All 2 attempts failed"):
            agent.build("Build model")
