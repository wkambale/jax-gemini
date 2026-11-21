"""Export generated JAX models and training scripts as standalone Python packages."""

from __future__ import annotations

import json
import keyword
from pathlib import Path

from jaxforge.codegen.validator import CodeValidator


def _sanitize_single_line(text: str) -> str:
    """Collapse control characters and newlines so strings serialize safely."""
    cleaned = " ".join(str(text).replace("\x00", "").splitlines()).strip()
    return cleaned or "Standalone JAX model exported by Jax Forge."


def export_project(
    output_dir: str | Path,
    model_code: str,
    train_code: str | None = None,
    eval_code: str | None = None,
    package_name: str = "generated_model",
    description: str = "Standalone JAX model exported by Jax Forge.",
    framework: str = "flax",
) -> Path:
    """Write validated JAX source files into a clean installable project directory."""
    if (
        not isinstance(package_name, str)
        or not package_name.isidentifier()
        or keyword.iskeyword(package_name)
    ):
        raise ValueError(
            f"package_name must be a valid non-keyword Python identifier, got {package_name!r}"
        )
    if not isinstance(model_code, str) or not model_code.strip():
        raise ValueError("model_code must be a non-empty Python source string.")

    CodeValidator.validate(model_code, "build")
    if train_code:
        CodeValidator.validate(train_code, "train")
    if eval_code:
        CodeValidator.validate(eval_code, "evaluate")

    safe_desc_literal = json.dumps(_sanitize_single_line(description))
    uses_equinox = framework == "equinox" or "equinox" in model_code

    root = Path(output_dir).resolve()
    pkg_dir = root / "src" / package_name
    tests_dir = root / "tests"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    (pkg_dir / "__init__.py").write_text(
        f"{safe_desc_literal}\n\nfrom {package_name}.model import build_model\n\n"
        '__all__ = ["build_model"]\n',
        encoding="utf-8",
    )
    (pkg_dir / "model.py").write_text(model_code.rstrip() + "\n", encoding="utf-8")

    if train_code:
        (pkg_dir / "train.py").write_text(train_code.rstrip() + "\n", encoding="utf-8")
    if eval_code:
        (pkg_dir / "eval.py").write_text(eval_code.rstrip() + "\n", encoding="utf-8")

    deps = ["jax>=0.4.25", "flax>=0.9.0", "optax>=0.2.0", "numpy>=1.24.0"]
    if uses_equinox:
        deps.append("equinox>=0.11.0")
    deps_toml = json.dumps(deps)

    pyproject = (
        "[build-system]\n"
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n\n'
        "[project]\n"
        f'name = "{package_name}"\n'
        'version = "0.1.0"\n'
        f"description = {safe_desc_literal}\n"
        'requires-python = ">=3.10"\n'
        f"dependencies = {deps_toml}\n\n"
        "[tool.hatch.build.targets.wheel]\n"
        f'packages = ["src/{package_name}"]\n'
    )
    (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")

    smoke_test = (
        "import inspect\n"
        "import jax\n"
        "import flax.nnx as nnx\n"
        f"from {package_name}.model import build_model\n\n\n"
        "def test_build_model_smoke():\n"
        "    params = list(inspect.signature(build_model).parameters)\n"
        "    if not params:\n"
        "        model = build_model()\n"
        "    elif params[0] == 'key':\n"
        "        model = build_model(jax.random.PRNGKey(0))\n"
        "    else:\n"
        "        model = build_model(nnx.Rngs(0))\n"
        "    assert model is not None\n"
    )
    (tests_dir / "test_smoke.py").write_text(smoke_test, encoding="utf-8")
    return root
