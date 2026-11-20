"""Export generated JAX models and training scripts as standalone Python packages."""

from __future__ import annotations

from pathlib import Path


def export_project(
    output_dir: str | Path,
    model_code: str,
    train_code: str | None = None,
    eval_code: str | None = None,
    package_name: str = "generated_model",
    description: str = "Standalone JAX model exported by Jax Forge.",
) -> Path:
    """Write validated JAX source files into a clean installable project directory."""
    if not isinstance(package_name, str) or not package_name.isidentifier():
        raise ValueError(f"package_name must be a valid Python identifier, got {package_name!r}")
    if not isinstance(model_code, str) or not model_code.strip():
        raise ValueError("model_code must be a non-empty Python source string.")

    root = Path(output_dir).resolve()
    pkg_dir = root / "src" / package_name
    tests_dir = root / "tests"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    (pkg_dir / "__init__.py").write_text(
        f'"""{description}"""\n\nfrom {package_name}.model import build_model\n\n'
        '__all__ = ["build_model"]\n',
        encoding="utf-8",
    )
    (pkg_dir / "model.py").write_text(model_code.rstrip() + "\n", encoding="utf-8")

    if train_code:
        (pkg_dir / "train.py").write_text(train_code.rstrip() + "\n", encoding="utf-8")
    if eval_code:
        (pkg_dir / "eval.py").write_text(eval_code.rstrip() + "\n", encoding="utf-8")

    pyproject = (
        "[build-system]\n"
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n\n'
        "[project]\n"
        f'name = "{package_name}"\n'
        'version = "0.1.0"\n'
        f'description = "{description}"\n'
        'requires-python = ">=3.10"\n'
        'dependencies = ["jax>=0.4.25", "flax>=0.9.0", "optax>=0.2.0", "numpy>=1.24.0"]\n\n'
        "[tool.hatch.build.targets.wheel]\n"
        f'packages = ["src/{package_name}"]\n'
    )
    (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")

    smoke_test = (
        "import flax.nnx as nnx\n"
        f"from {package_name}.model import build_model\n\n\n"
        "def test_build_model_smoke():\n"
        "    model = build_model(nnx.Rngs(0))\n"
        "    assert model is not None\n"
    )
    (tests_dir / "test_smoke.py").write_text(smoke_test, encoding="utf-8")
    return root
