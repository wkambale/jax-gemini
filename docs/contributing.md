# Contributing guide

Contributions, bug reports, and documentation improvements to `jaxforge` (Jax Forge) are welcome.

## Reporting issues

Open an issue on [GitHub](https://github.com/wkambale/jaxforge/issues) with reproduction steps, Python and JAX versions, configured provider or execution backend, and the relevant exception output.

## Pull request workflow

1. Branch from `main` using a descriptive branch name such as `fix/validator-edge-case` or `feat/provider-adapter`.
2. Add or update unit and integration tests covering your change cleanly.
3. Run linting, strict type checking, test coverage, and strict MkDocs verification before pushing.
4. Open a pull request against `main` and link any related issue.

## Local development setup

Create a Python 3.10 to 3.12 virtual environment and install development extras:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,viz,docs]"
```

Run the full verification suite:

```bash
ruff check src tests examples
mypy src
pytest --cov=jaxforge tests/
mkdocs build --strict
```
