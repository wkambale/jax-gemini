from typing import Any

from jax_gemini.exceptions import JaxGeminiValidationError
from jax_gemini.memory.conversation import ConversationMemory, ModelState
from jax_gemini.prompts import system_prompt

MAX_PROMPT_CHARS = 16_000
MAX_CONTEXT_PROMPT_CHARS = 48_000
MAX_MEMORY_CHARS = 32_000
MAX_STATE_DESCRIPTION_CHARS = 2_000
MAX_STATE_CODE_CHARS = 8_000

_SYSTEM_PROMPTS: dict[str, str] = {
    "build": system_prompt.BUILD_SYSTEM_PROMPT,
    "train": system_prompt.TRAIN_SYSTEM_PROMPT,
    "evaluate": system_prompt.EVALUATE_SYSTEM_PROMPT,
    "save": system_prompt.SAVE_SYSTEM_PROMPT,
    "load": system_prompt.LOAD_SYSTEM_PROMPT,
    "load_data": system_prompt.LOAD_DATA_SYSTEM_PROMPT,
    "preprocess_data": system_prompt.PREPROCESS_DATA_SYSTEM_PROMPT,
    "analyze_data": system_prompt.ANALYZE_DATA_SYSTEM_PROMPT,
    "visualize_data": system_prompt.VISUALIZE_DATA_SYSTEM_PROMPT,
}


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def _bound_history_messages(messages: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    """Keep the most recent conversation history messages within the character budget."""
    kept: list[dict[str, Any]] = []
    total_chars = 0
    for msg in reversed(messages):
        msg_chars = sum(len(str(part)) for part in msg.get("parts", []))
        if kept and total_chars + msg_chars > max_chars:
            break
        if not kept and msg_chars > max_chars:
            break
        kept.append(msg)
        total_chars += msg_chars
    kept.reverse()
    # Preserve user-first ordering expected by Gemini chat turns.
    while kept and kept[0].get("role") != "user":
        kept.pop(0)
    return kept


class ContextBuilder:
    @staticmethod
    def validate_prompt(user_prompt: str, max_chars: int = MAX_PROMPT_CHARS) -> None:
        """Validate that the user prompt is a non-empty string within the length bound."""
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise JaxGeminiValidationError("Prompt must be a non-empty string.")
        if len(user_prompt) > max_chars:
            raise JaxGeminiValidationError(
                f"Prompt length ({len(user_prompt)} chars) exceeds maximum allowed "
                f"({max_chars} chars)."
            )

    @staticmethod
    def build(
        intent: str,
        user_prompt: str,
        memory: ConversationMemory,
        current_model_state: ModelState | None,
        max_prompt_chars: int = MAX_PROMPT_CHARS,
    ) -> list[dict[str, Any]]:
        """Assemble the full message list sent to Gemini."""
        ContextBuilder.validate_prompt(user_prompt, max_chars=max_prompt_chars)
        sys_prompt = _SYSTEM_PROMPTS.get(intent, system_prompt.BUILD_SYSTEM_PROMPT)

        messages: list[dict[str, Any]] = []

        # System prompt as first user turn, followed by an ack from the model, per guidelines
        messages.append({"role": "user", "parts": [sys_prompt]})
        messages.append({"role": "model", "parts": ["Understood. I will generate JAX/Flax code."]})

        # Extend with bounded conversation history
        messages.extend(_bound_history_messages(memory.to_gemini_messages(), MAX_MEMORY_CHARS))

        # Include current model state so modify/train/evaluate/save steps see prior code
        if current_model_state is not None:
            desc = _truncate(current_model_state.description, MAX_STATE_DESCRIPTION_CHARS)
            code = _truncate(current_model_state.code, MAX_STATE_CODE_CHARS)
            state_context = (
                f"Current model state (produced by '{current_model_state.intent}'):\n"
                f"Description: {desc}\n"
                f"Code:\n{code}"
            )
            messages.append({"role": "user", "parts": [state_context]})
            messages.append(
                {
                    "role": "model",
                    "parts": ["Understood. I have the current model state context."],
                }
            )

        # Finally, append current user prompt
        messages.append({"role": "user", "parts": [user_prompt]})

        return messages

    @staticmethod
    def get_schema(intent: str) -> dict[str, Any]:
        """Get the expected JSON schema for the given intent."""
        # A simple permissive schema that expects at least 'intent', 'code', 'description'
        return {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "code": {"type": "string"},
                "description": {"type": "string"},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "parameters": {"type": "object"},
                "hyperparameters": {"type": "object"},
            },
            "required": ["intent", "code", "description"],
        }
