"""Backward-compatibility shim for jax_gemini.memory.conversation."""

from jaxforge.memory.conversation import ConversationMemory, ConversationTurn, ModelState

__all__ = ["ConversationMemory", "ConversationTurn", "ModelState"]
