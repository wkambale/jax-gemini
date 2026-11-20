from unittest.mock import MagicMock

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from jax_gemini import JaxGemini, JaxGeminiConfig
from jax_gemini.contrib import (
    available_optimizers,
    count_opt_steps,
    count_parameters,
    create_optimizer,
    summarize_module,
)
from jax_gemini.exceptions import JaxGeminiCheckpointError, JaxGeminiIntentError
from jax_gemini.pipeline import ModelBuilder, ModelEvaluator, ModelExporter, ModelTrainer


class _TinyNet(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.dense = nnx.Linear(4, 2, rngs=rngs)

    def __call__(self, x):
        return self.dense(x)


class TestPipelineAndContrib:
    def test_agent_run_routes_via_intent_classifier(self, monkeypatch):
        agent = JaxGemini(config=JaxGeminiConfig(gemini_api_key="test-key"))
        monkeypatch.setattr(agent, "build", MagicMock(return_value="built"))
        monkeypatch.setattr(agent, "modify", MagicMock(return_value="modified"))
        monkeypatch.setattr(agent, "train", MagicMock(return_value=("m", {"loss": 0.1})))
        monkeypatch.setattr(agent, "evaluate", MagicMock(return_value={"accuracy": 0.9}))
        monkeypatch.setattr(agent, "save", MagicMock(return_value="ckpt_path"))
        monkeypatch.setattr(agent, "load", MagicMock(return_value="loaded"))
        monkeypatch.setattr(agent, "load_data", MagicMock(return_value=(1, 2)))
        monkeypatch.setattr(agent, "preprocess_data", MagicMock(return_value="pre"))
        monkeypatch.setattr(agent, "analyze_data", MagicMock(return_value="stats"))
        monkeypatch.setattr(agent, "visualize_data", MagicMock(return_value="fig"))

        assert agent.run("build a CNN") == "built"
        agent._current_model = "existing"
        assert agent.run("add dropout layer") == "modified"
        assert agent.run("train with adam", dataset=(1, 2)) == ("m", {"loss": 0.1})
        assert agent.run("evaluate accuracy", dataset=(1, 2)) == {"accuracy": 0.9}
        assert agent.run("save checkpoint", name="v1") == "ckpt_path"
        assert agent.run("load checkpoint", name="v1") == "loaded"
        assert agent.run("load data from memory") == (1, 2)
        assert agent.run("preprocess the dataset", dataset=(1, 2)) == "pre"
        assert agent.run("analyze the dataset", dataset=(1, 2)) == "stats"
        assert agent.run("plot training curve", dataset=(1, 2)) == "fig"

        with pytest.raises(JaxGeminiCheckpointError):
            agent.run("save checkpoint")
        with pytest.raises(JaxGeminiCheckpointError):
            agent.run("load checkpoint")
        with pytest.raises(JaxGeminiIntentError):
            agent.run("analyze the dataset")

    def test_pipeline_helpers_orchestrate_agent(self):
        mock_agent = MagicMock(spec=JaxGemini)
        mock_agent.build.return_value = "m0"
        mock_agent.modify.return_value = "m1"
        mock_agent.train.return_value = ("m_trained", {"accuracy": 0.95})
        mock_agent.evaluate.return_value = {"accuracy": 0.96}
        mock_agent.save.return_value = "./jax_gemini_checkpoints/v1"
        mock_agent.load.return_value = "m_restored"

        builder = ModelBuilder(mock_agent)
        assert builder.run("build", refinements=("add dropout",)) == "m1"

        trainer = ModelTrainer(mock_agent)
        model, metrics = trainer.run("train")
        assert model == "m_trained"
        assert metrics == {"accuracy": 0.95}
        assert trainer.history == [{"accuracy": 0.95}]

        evaluator = ModelEvaluator(mock_agent)
        assert evaluator.run("evaluate") == {"accuracy": 0.96}

        exporter = ModelExporter(mock_agent)
        assert exporter.export("v1") == "./jax_gemini_checkpoints/v1"
        assert exporter.restore("v1") == "m_restored"

    def test_contrib_flax_helpers(self):
        net = _TinyNet(nnx.Rngs(0))
        # Linear(4, 2): 4*2 weights + 2 bias = 10
        assert count_parameters(net) == 10
        summary = summarize_module(net)
        assert summary == {
            "module_type": "_TinyNet",
            "parameter_count": 10,
            "submodules": ["dense"],
        }
        with pytest.raises(TypeError):
            count_parameters("not a module")  # type: ignore[arg-type]

    def test_contrib_optax_helpers(self):
        assert available_optimizers() == ("adam", "adamw", "rmsprop", "sgd")
        tx = create_optimizer("adamw", learning_rate=1e-2)
        state = tx.init({"w": jnp.ones((2, 2))})
        assert count_opt_steps(state) == 0
        _, state = tx.update({"w": jnp.ones((2, 2))}, state, {"w": jnp.ones((2, 2))})
        assert count_opt_steps(state) == 1

        with pytest.raises(ValueError):
            create_optimizer("unknown")
        with pytest.raises(ValueError):
            create_optimizer("adam", learning_rate=0)
