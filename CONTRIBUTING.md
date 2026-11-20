# Contributing guide

Contributions, bug reports, and documentation improvements to `jax-gemini` are welcome.

## Reporting issues

Open an issue on GitHub with reproduction steps, Python and JAX versions, and the relevant exception output. For AST validation or sandbox issues, include the prompt and intent used.

## Pull request workflow

1. Branch from `main` using a descriptive branch name such as `fix/validator-edge-case` or `feat/pipeline-helper`.
2. Add or update unit and integration tests covering your change.
3. Run linting, strict type checking, and the test suite with coverage before pushing.
4. Open a pull request against `main` and link the corresponding issue.

## Local development setup

Create a Python 3.10 to 3.12 virtual environment and install development extras:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,viz,docs]"
```

Run verification checks:

```bash
ruff check src tests examples
mypy src
pytest --cov=jax_gemini tests/
mkdocs build --strict
```
