import asyncio
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import jaxforge as jf
from jaxforge.codegen.diff import apply_unified_diff, build_unified_diff
from jaxforge.codegen.parser import CodeParser
from jaxforge.exceptions import JaxForgeApprovalError
from jaxforge.plan import DatasetProfile, GeneratedPlan
from jaxforge.prompts.context_builder import ContextBuilder


class TestPreviewExploreExport:
    @patch("jaxforge.llm.gemini.genai")
    def test_preview_dry_run_approval_gate_and_html(self, mock_genai):
        resp = MagicMock()
        resp.text = json.dumps(
            {
                "intent": "build",
                "code": (
                    "import flax.nnx as nnx\n"
                    "def build_model(rngs):\n"
                    "    return nnx.Linear(4, 2, rngs=rngs)\n"
                ),
                "description": "Single linear layer",
                "warnings": ["check dims"],
            }
        )
        mock_genai.GenerativeModel.return_value.generate_content.return_value = resp

        agent = jf.JaxForge(jf.JaxForgeConfig(gemini_api_key="test-key", auto_approve=False))
        plan = agent.preview("Build a linear layer")
        assert isinstance(plan, GeneratedPlan)
        assert plan.validation_passed is True
        assert agent._current_model is None
        html_out = plan._repr_html_()
        assert "Jax Forge preview" in html_out
        assert "Single linear layer" in html_out

        # auto_approve=False blocks unapproved execution
        with pytest.raises(JaxForgeApprovalError, match="auto_approve is False"):
            agent.execute_plan(plan)

        # Rejection callback raises JaxForgeApprovalError without touching _current_model
        with pytest.raises(JaxForgeApprovalError, match="rejected"):
            agent.build("Build a linear layer", approve=lambda _p: False)
        assert agent._current_model is None

        # Explicit approval runs cleanly
        import flax.nnx as nnx

        model = agent.execute_plan(
            plan, runtime_args={"rngs": nnx.Rngs(0)}, approve=lambda _p: True
        )
        assert model is not None
        assert agent._current_model is model

        # dry_run returns GeneratedPlan cleanly
        dry = agent.build("Build a linear layer", dry_run=True)
        assert isinstance(dry, GeneratedPlan)

    def test_dataset_profile_and_context_injection(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        y = np.array([0, 1])
        prof = DatasetProfile.from_dataset((X, y))
        text = prof.to_prompt_text()
        assert "supervised_pair" in text
        assert "nan_or_inf_count" in text

        msgs = ContextBuilder.build(
            intent="train",
            user_prompt="Train MLP",
            memory=jf.JaxForge()._memory,
            current_model_state=None,
            dataset_profile=prof,
        )
        assert any("Dataset profile" in str(m["parts"][0]) for m in msgs)

    def test_unified_diff_roundtrip_and_parser(self):
        before = "def build_model(rngs):\n    return 1\n"
        after = "def build_model(rngs):\n    return 2\n"
        diff = build_unified_diff(before, after)
        assert "-    return 1" in diff
        assert "+    return 2" in diff
        assert apply_unified_diff(before, diff) == after

        parsed = CodeParser.extract(
            {"intent": "build", "diff": diff}, "build", previous_code=before
        )
        assert parsed == after.strip()

    @patch("jaxforge.llm.gemini.genai")
    def test_explore_selects_highest_scoring_candidate_and_export(self, mock_genai, tmp_path):
        c1 = MagicMock(
            text=json.dumps(
                {
                    "intent": "train",
                    "code": (
                        "def train_model(model, dataset, epochs=1, learning_rate=1e-3):\n"
                        "    return 'm1', {'accuracy': 0.72, 'loss': 0.5}\n"
                    ),
                    "description": "trial 1",
                }
            )
        )
        c2 = MagicMock(
            text=json.dumps(
                {
                    "intent": "train",
                    "code": (
                        "def train_model(model, dataset, epochs=1, learning_rate=1e-3):\n"
                        "    return 'm2', {'accuracy': 0.94, 'loss': 0.1}\n"
                    ),
                    "description": "trial 2",
                }
            )
        )
        c3 = MagicMock(
            text=json.dumps(
                {
                    "intent": "train",
                    "code": (
                        "def train_model(model, dataset, epochs=1, learning_rate=1e-3):\n"
                        "    return 'm3', {'accuracy': 0.85, 'loss': 0.2}\n"
                    ),
                    "description": "trial 3",
                }
            )
        )
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = [c1, c2, c3]

        agent = jf.JaxForge(jf.JaxForgeConfig(gemini_api_key="test-key", max_retries=1))
        agent._current_model = "base"
        X = np.ones((4, 2), dtype=np.float32)
        y = np.zeros((4,), dtype=np.int32)
        exp = agent.explore("Train model", dataset=(X, y), metric="accuracy", candidates=3)
        assert exp.best_candidate.index == 2
        assert exp.best_candidate.score == 0.94
        assert agent._current_model == "m2"
        assert agent.usage_summary()["calls"] == 3

        # Build + export standalone project
        agent._last_build_code = (
            "import flax.nnx as nnx\n"
            "def build_model(rngs):\n"
            "    return nnx.Linear(2, 2, rngs=rngs)\n"
        )
        pkg_root = agent.export_project(tmp_path / "pkg_out", package_name="demo_pkg")
        assert (pkg_root / "pyproject.toml").exists()
        assert (pkg_root / "src" / "demo_pkg" / "model.py").exists()
        assert (pkg_root / "src" / "demo_pkg" / "train.py").exists()
        assert (pkg_root / "tests" / "test_smoke.py").exists()

    @patch("jaxforge.llm.gemini.genai")
    def test_async_wrappers_work(self, mock_genai):
        resp = MagicMock(
            text=json.dumps(
                {
                    "intent": "build",
                    "code": "import flax.nnx as nnx\ndef build_model(rngs): return 42\n",
                    "description": "async ok",
                }
            )
        )
        mock_genai.GenerativeModel.return_value.generate_content.return_value = resp
        agent = jf.JaxForge(jf.JaxForgeConfig(gemini_api_key="test-key"))

        async def _run():
            plan = await agent.apreview("Build 42")
            assert plan.intent == "build"
            val = await agent.abuild("Build 42")
            assert val == 42

        asyncio.run(_run())
