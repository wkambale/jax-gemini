import json
from unittest.mock import MagicMock, patch

import pytest

import jaxforge as jf
from jaxforge.exceptions import JaxForgeCheckpointError


class TestCheckpointPathContainment:
    def test_save_and_load_reject_traversal_and_absolute_paths(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        agent = jf.JaxForge(
            jf.JaxForgeConfig(gemini_api_key="test-key", checkpoint_dir=str(ckpt_dir))
        )

        bad_names = ["", "   ", ".", "..", "../escape", "sub/../../escape", "/etc/passwd", "a\nb"]
        for bad_name in bad_names:
            with pytest.raises(JaxForgeCheckpointError):
                agent.save(bad_name)
            with pytest.raises(JaxForgeCheckpointError):
                agent.load(bad_name)

    def test_save_rejects_symlink_escape(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        outside_dir = tmp_path / "outside"
        ckpt_dir.mkdir()
        outside_dir.mkdir()
        symlink_path = ckpt_dir / "link_out"
        symlink_path.symlink_to(outside_dir)

        agent = jf.JaxForge(
            jf.JaxForgeConfig(gemini_api_key="test-key", checkpoint_dir=str(ckpt_dir))
        )
        with pytest.raises(JaxForgeCheckpointError, match="escapes checkpoint_dir"):
            agent.save("link_out/model")

    @patch("jaxforge.llm.gemini.genai")
    def test_save_and_load_valid_relative_name_succeeds(self, mock_genai, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        save_resp = MagicMock()
        save_resp.text = json.dumps(
            {
                "intent": "save",
                "code": "def save_model(model, path):\n    return path\n",
                "description": "Saved.",
                "warnings": [],
            }
        )
        load_resp = MagicMock()
        load_resp.text = json.dumps(
            {
                "intent": "load",
                "code": "def load_model(path):\n    return {'loaded_from': path}\n",
                "description": "Loaded.",
                "warnings": [],
            }
        )
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = [
            save_resp,
            load_resp,
        ]

        agent = jf.JaxForge(
            jf.JaxForgeConfig(gemini_api_key="test-key", checkpoint_dir=str(ckpt_dir))
        )
        saved_path = agent.save("run1/ckpt")
        assert saved_path == str((ckpt_dir / "run1/ckpt").resolve())
        loaded = agent.load("run1/ckpt")
        assert loaded == {"loaded_from": saved_path}
