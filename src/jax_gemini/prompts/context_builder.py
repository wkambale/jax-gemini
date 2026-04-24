from jax_gemini.memory.conversation import ConversationMemory, ModelState
from jax_gemini.prompts import system_prompt


class ContextBuilder:
    @staticmethod
    def build(
        intent: str,
        user_prompt: str,
        memory: ConversationMemory,
        current_model_state: ModelState | None,
    ) -> list[dict]:
        """Assemble the full message list sent to Gemini."""
        # 1. Select the relevant system prompt
        if intent == "build":
            sys_prompt = system_prompt.BUILD_SYSTEM_PROMPT
        elif intent == "train":
            sys_prompt = system_prompt.TRAIN_SYSTEM_PROMPT
        elif intent == "evaluate":
            sys_prompt = system_prompt.EVALUATE_SYSTEM_PROMPT
        elif intent == "save":
            sys_prompt = system_prompt.SAVE_SYSTEM_PROMPT
        elif intent == "load":
            sys_prompt = system_prompt.LOAD_SYSTEM_PROMPT
        elif intent == "load_data":
            sys_prompt = system_prompt.LOAD_DATA_SYSTEM_PROMPT
        elif intent == "preprocess_data":
            sys_prompt = system_prompt.PREPROCESS_DATA_SYSTEM_PROMPT
        elif intent == "analyze_data":
            sys_prompt = system_prompt.ANALYZE_DATA_SYSTEM_PROMPT
        else:
            sys_prompt = system_prompt.BUILD_SYSTEM_PROMPT

        messages = []

        # System prompt as first user turn, followed by an ack from the model, per guidelines
        messages.append({"role": "user", "parts": [sys_prompt]})
        messages.append({"role": "model", "parts": ["Understood. I will generate JAX/Flax code."]})

        # Extend with conversation history
        messages.extend(memory.to_gemini_messages())

        # If there's a current model state and we are modifying/training/etc., we might remind it
        # but let's stick to the prompt engineering guidelines

        # Finally, append current user prompt
        messages.append({"role": "user", "parts": [user_prompt]})

        return messages

    @staticmethod
    def get_schema(intent: str) -> dict:
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
