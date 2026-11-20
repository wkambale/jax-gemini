"""Extract code or patch blocks from structured LLM responses."""

from typing import Any

from jaxforge.codegen.diff import apply_unified_diff
from jaxforge.exceptions import JaxForgeLLMError


class CodeParser:
    @staticmethod
    def extract(
        response: dict[str, Any],
        intent: str,
        previous_code: str | None = None,
    ) -> str:
        """Extract the Python source from a structured JSON response (code or unified diff)."""
        if not isinstance(response, dict):
            raise JaxForgeLLMError(
                "LLM response must be a JSON object.",
                raw_response=str(response),
            )

        response_intent = response.get("intent")
        if isinstance(response_intent, str) and response_intent and response_intent != intent:
            raise JaxForgeLLMError(
                f"LLM response intent '{response_intent}' does not match '{intent}'.",
                raw_response=str(response),
            )

        if "code" not in response and "diff" not in response:
            raise JaxForgeLLMError(
                f"LLM response missing 'code' field. Got fields: {list(response.keys())}",
                raw_response=str(response),
            )

        diff_text = response.get("diff")
        if isinstance(diff_text, str) and diff_text.strip() and previous_code is not None:
            patched = apply_unified_diff(previous_code, diff_text).strip()
            if not patched:
                raise JaxForgeLLMError("Patch resulted in an empty code block.")
            return patched

        if "code" not in response:
            raise JaxForgeLLMError(
                f"LLM response missing 'code' field. Got fields: {list(response.keys())}",
                raw_response=str(response),
            )

        raw_code = response["code"]
        if not isinstance(raw_code, str):
            raise JaxForgeLLMError(
                "LLM response 'code' field must be a string.",
                raw_response=str(response),
            )
        code = raw_code.strip()
        if not code:
            raise JaxForgeLLMError("LLM returned an empty code block.")
        return code
