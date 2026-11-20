import pathlib
import re

import jax_gemini

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_FORBIDDEN_DASH_RE = re.compile(r"[\u2014\u2013]")
_FORBIDDEN_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_HEADING_AMP_RE = re.compile(r"(?<!\S)&(?!\S)")
_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def _iter_repo_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "CHANGELOG.md",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "mkdocs.yml",
    ]
    for folder in ("src", "tests", "docs", "examples", ".github"):
        base = REPO_ROOT / folder
        if base.exists():
            files.extend(
                p for p in sorted(base.rglob("*")) if p.is_file() and p.suffix != ".pyc"
            )
    return [f for f in files if f.exists()]


class TestEditorialAndVersionHygiene:
    def test_package_version_matches_pyproject(self):
        pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = _VERSION_RE.search(pyproject_text)
        assert match is not None
        assert jax_gemini.__version__ == match.group(1)

    def test_no_em_or_en_dashes_emojis_or_badge_pills(self):
        for path in _iter_repo_files():
            text = path.read_text(encoding="utf-8", errors="replace")
            assert not _FORBIDDEN_DASH_RE.search(text), f"Forbidden dash in {path}"
            assert not _FORBIDDEN_EMOJI_RE.search(text), f"Forbidden emoji in {path}"
            if path.suffix == ".md":
                assert "img.shields.io" not in text, f"Pill badge found in {path}"

    def test_markdown_headings_have_no_ampersands_or_trailing_periods(self):
        for path in _iter_repo_files():
            if path.suffix != ".md":
                continue
            in_code_fence = False
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("```"):
                    in_code_fence = not in_code_fence
                    continue
                if in_code_fence or not line.startswith("#"):
                    continue
                assert not _HEADING_AMP_RE.search(line), f"Ampersand in heading of {path}: {line}"
                assert not line.rstrip().endswith("."), f"Trailing period in {path}: {line}"
