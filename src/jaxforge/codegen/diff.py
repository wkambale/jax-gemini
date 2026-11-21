"""Unified diff generation and strict hunk patching for incremental model edits."""

import difflib
import re

from jaxforge.exceptions import JaxForgeLLMError

_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_ALLOWED_PREAMBLE_PREFIXES = ("--- ", "+++ ", "diff ", "index ")


def build_unified_diff(
    before: str,
    after: str,
    from_label: str = "a/model.py",
    to_label: str = "b/model.py",
) -> str:
    """Compute a readable unified diff between two Python source strings."""
    if before == after:
        return ""
    lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=from_label,
        tofile=to_label,
    )
    return "".join(lines)


def apply_unified_diff(original: str, patch_text: str) -> str:
    """Apply a single-file unified diff patch strictly against ``original`` source code."""
    if not isinstance(patch_text, str) or not patch_text.strip():
        raise JaxForgeLLMError("Unified diff patch is empty.")

    orig_lines = original.splitlines()
    patch_lines = patch_text.splitlines()
    hunks: list[tuple[int, int, int, list[str]]] = []
    seen_file_headers = 0

    idx = 0
    while idx < len(patch_lines):
        line = patch_lines[idx]
        if not line.strip() and not hunks:
            idx += 1
            continue
        if line.startswith(("--- ", "+++ ")):
            if hunks:
                raise JaxForgeLLMError("Multi-file unified diffs are not supported.")
            seen_file_headers += 1
            if seen_file_headers > 2:
                raise JaxForgeLLMError("Multi-file unified diffs are not supported.")
            idx += 1
            continue
        if line.startswith(("diff ", "index ")) and not hunks:
            idx += 1
            continue
        match = _HUNK_HEADER_RE.match(line)
        if not match:
            raise JaxForgeLLMError(f"Unexpected diff preamble line: {line!r}")
        old_start = int(match.group(1))
        old_count = int(match.group(2)) if match.group(2) is not None else 1
        new_count = int(match.group(4)) if match.group(4) is not None else 1
        idx += 1
        body: list[str] = []
        while idx < len(patch_lines) and not _HUNK_HEADER_RE.match(patch_lines[idx]):
            cur_line = patch_lines[idx]
            if cur_line.startswith("diff --git "):
                raise JaxForgeLLMError("Multi-file unified diffs are not supported.")
            body.append(cur_line)
            idx += 1
        hunks.append((old_start, old_count, new_count, body))

    if not hunks:
        raise JaxForgeLLMError("Patch contained no valid unified diff hunks.")

    result: list[str] = []
    cursor = 0
    for old_start, old_count, new_count, body in hunks:
        target_idx = 0 if old_count == 0 and old_start == 0 else max(0, old_start - 1)
        if target_idx < cursor or target_idx > len(orig_lines):
            raise JaxForgeLLMError("Diff hunk offset is out of range for target code.")
        result.extend(orig_lines[cursor:target_idx])
        pos = target_idx
        consumed_old = 0
        consumed_new = 0
        for raw_line in body:
            if not raw_line:
                raise JaxForgeLLMError("Invalid empty diff line without prefix.")
            if (
                consumed_old == old_count
                and consumed_new == new_count
                and raw_line.startswith(_ALLOWED_PREAMBLE_PREFIXES)
            ):
                raise JaxForgeLLMError("Multi-file unified diffs are not supported.")
            marker = raw_line[0]
            content = raw_line[1:]
            if marker == " ":
                if pos >= len(orig_lines) or orig_lines[pos] != content:
                    raise JaxForgeLLMError("Diff context line does not match target code.")
                result.append(orig_lines[pos])
                pos += 1
                consumed_old += 1
                consumed_new += 1
            elif marker == "-":
                if pos >= len(orig_lines) or orig_lines[pos] != content:
                    raise JaxForgeLLMError("Diff deletion line does not match target code.")
                pos += 1
                consumed_old += 1
            elif marker == "+":
                result.append(content)
                consumed_new += 1
            elif marker == "\\":
                continue
            else:
                raise JaxForgeLLMError(f"Unsupported diff line marker '{marker}'.")
        if consumed_old != old_count or consumed_new != new_count:
            raise JaxForgeLLMError(
                f"Diff hunk line count mismatch: expected (-{old_count}, +{new_count}), "
                f"got (-{consumed_old}, +{consumed_new})."
            )
        cursor = pos

    result.extend(orig_lines[cursor:])
    trailing_nl = "\n" if original.endswith("\n") else ""
    return "\n".join(result) + trailing_nl
