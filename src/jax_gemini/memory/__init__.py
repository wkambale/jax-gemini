"""Backward-compatibility shim for jax_gemini.memory."""

from jaxforge.memory import ConversationMemory, ConversationTurn, ModelState

__all__ = ["ConversationMemory", "ConversationTurn", "ModelState"]
