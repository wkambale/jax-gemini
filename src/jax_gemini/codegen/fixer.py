import re

from jax_gemini.codegen.validator import INTENT_ALLOWED_IMPORTS, WHITELIST_IMPORTS

_ABS_PATH_RE = re.compile(r'File "([^"]+)"')
_MAX_ERROR_CHARS = 1500
_MAX_CODE_CHARS = 20000


def _safe_fence(text: str) -> str:
    """Return a backtick fence longer than any backtick run in text."""
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=2)
    return "`" * max(4, longest + 1)


def sanitize_error_for_llm(error_message: str) -> str:
    """Redact host filesystem paths, control chars, and long tracebacks before LLM retry."""
    cleaned = error_message.replace("\x00", "")

    def _redact_file(match: re.Match[str]) -> str:
        raw_path = match.group(1)
        if raw_path == "<string>":
            return 'File "<sandbox>"'
        filename = raw_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        return f'File "<redacted>/{filename}"'

    redacted = _ABS_PATH_RE.sub(_redact_file, cleaned)
    if len(redacted) > _MAX_ERROR_CHARS:
        redacted = redacted[:_MAX_ERROR_CHARS] + "\n... [truncated]"
    return redacted


class CodeFixer:
    @staticmethod
    def build_fix_prompt(
        original_prompt: str,
        failed_code: str,
        error_message: str,
        attempt: int,
        intent: str | None = None,
    ) -> str:
        """
        Build a sanitized correction prompt to send back to Gemini when code fails.

        Escapes Markdown fences and redacts host paths from error messages.
        """
        safe_code = failed_code[:_MAX_CODE_CHARS]
        safe_error = sanitize_error_for_llm(error_message)
        allowed = (
            INTENT_ALLOWED_IMPORTS.get(intent, WHITELIST_IMPORTS)
            if intent
            else WHITELIST_IMPORTS
        )
        allowed_list = ", ".join(sorted(allowed))

        code_fence = _safe_fence(safe_code)
        err_fence = _safe_fence(safe_error)
        prompt_fence = _safe_fence(original_prompt)

        return (
            f"The previous code failed on attempt {attempt} for the user request below:\n"
            f"{prompt_fence}text\n{original_prompt}\n{prompt_fence}\n\n"
            f"Failed code:\n{code_fence}python\n{safe_code}\n{code_fence}\n\n"
            f"Error summary:\n{err_fence}text\n{safe_error}\n{err_fence}\n\n"
            f"Please fix the code. Return ONLY corrected code in the same JSON format. "
            f"Do not change the function name or signature. "
            f"Only import from: {allowed_list}."
        )
