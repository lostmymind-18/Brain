"""
ConversationMemory: manages the messages list in OpenAI format.

Stores messages using OpenAI's canonical format:
  - user:      {"role": "user", "content": "..."}
  - assistant: {"role": "assistant", "content": "..."} or
               {"role": "assistant", "content": null, "tool_calls": [...]}
  - tool:      {"role": "tool", "tool_call_id": "...", "content": "..."}

Provider adapters translate from this format to their wire format at call time.

Week 2 implements conversation memory (turn history).
Episodic and semantic memory will be added in Week 4 (Knowledge Graph).
"""
from __future__ import annotations

import logging

from bookmind_tutor.agents.models import ToolCallResult

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Stores the alternating user/assistant turn history in OpenAI format.

    When max_turns is exceeded, the oldest user+assistant pair is dropped
    (the first turn is never dropped even if it is a system context turn).

    Args:
        max_turns: Maximum number of complete user+assistant pairs to retain.
            None means unlimited.
    """

    def __init__(self, max_turns: int | None = 50) -> None:
        self._messages: list[dict] = []
        self._max_turns = max_turns

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def add_user(self, content: str) -> None:
        """Add a user turn with plain text content."""
        self._messages.append({"role": "user", "content": content})
        self._trim_if_needed()

    def add_assistant(self, raw_message: dict) -> None:
        """
        Add an assistant turn.

        raw_message is the CompletionResponse.raw_message dict in OpenAI format,
        or any dict with role="assistant". Pass it directly from CompletionResponse.
        """
        self._messages.append(raw_message)

    def add_tool_results(self, results: list[ToolCallResult]) -> None:
        """
        Add tool results as individual role="tool" messages (OpenAI format).

        Each result becomes its own message. The Anthropic adapter merges
        consecutive tool messages into a single user turn at call time.
        """
        if not results:
            return
        for r in results:
            self._messages.append({
                "role": "tool",
                "tool_call_id": r.tool_use_id,
                "content": r.content,
            })
        self._trim_if_needed()

    def clear(self) -> None:
        self._messages = []

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def messages(self) -> list[dict]:
        """Return the current messages list (a copy to prevent external mutation)."""
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trim_if_needed(self) -> None:
        if self._max_turns is None:
            return
        limit = self._max_turns * 2
        while len(self._messages) > limit:
            self._messages.pop(0)
            if self._messages:
                self._messages.pop(0)
            logger.debug(
                "ConversationMemory: trimmed oldest turn (now %d messages)",
                len(self._messages),
            )
