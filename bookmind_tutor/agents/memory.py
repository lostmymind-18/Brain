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

    Trims history when it grows too large by evicting the oldest complete
    turn (user message + all assistant/tool messages until the next user).
    Dropping a full logical turn avoids leaving orphaned tool results at the
    start of the history, which some providers reject.

    Two independent limits apply:

    - max_turns: maximum number of complete user turns to retain.
    - max_tokens: approximate token budget for the message list, estimated as
      total character count divided by 4 (no external dependency required).
      Set comfortably below the model context window to leave room for the
      system prompt, tool definitions, and completion. Default 100_000 leaves
      ~28k headroom for gpt-4o-mini's 128k window.

    Either limit may be set to None to disable it.

    Args:
        max_turns: Maximum number of complete user turns to retain. None = unlimited.
        max_tokens: Approximate token budget for all stored messages. None = unlimited.
    """

    def __init__(
        self,
        max_turns: int | None = 50,
        max_tokens: int | None = 100_000,
    ) -> None:
        self._messages: list[dict] = []
        self._max_turns = max_turns
        self._max_tokens = max_tokens

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

    def _estimated_tokens(self) -> int:
        """
        Rough token estimate: sum of all string value lengths divided by 4.

        The /4 heuristic works for English and code; it slightly underestimates
        for very short strings and slightly overestimates for heavily tokenized
        text. Accuracy is sufficient for a trimming threshold.
        """
        total = 0
        for msg in self._messages:
            for v in msg.values():
                if isinstance(v, str):
                    total += len(v)
                elif v is not None:
                    total += len(str(v))
        return total // 4

    def _drop_oldest_turn(self) -> bool:
        """
        Drop the oldest complete user+assistant exchange from memory.

        A "turn" is everything from the first user message up to (but not
        including) the next user message: one user message, any number of
        assistant messages with tool_calls, any number of tool result messages,
        and the final assistant text message.

        Returns True if a turn was dropped, False when only one user turn
        remains (we never drop the last turn).
        """
        if len(self._messages) < 2:
            return False
        # Find the start of the second user turn = end of the first turn.
        end = 1
        while end < len(self._messages) and self._messages[end]["role"] != "user":
            end += 1
        if end >= len(self._messages):
            # Only one user turn left; preserve it.
            return False
        del self._messages[:end]
        logger.debug(
            "ConversationMemory: dropped oldest turn (%d messages removed, "
            "now %d messages, ~%d estimated tokens)",
            end,
            len(self._messages),
            self._estimated_tokens(),
        )
        return True

    def _trim_if_needed(self) -> None:
        # Turn-count guard.
        if self._max_turns is not None:
            turn_count = sum(1 for m in self._messages if m["role"] == "user")
            while turn_count > self._max_turns:
                if not self._drop_oldest_turn():
                    break
                turn_count -= 1

        # Token-budget guard: runs independently so either limit alone is enough.
        if self._max_tokens is not None:
            while self._estimated_tokens() > self._max_tokens:
                if not self._drop_oldest_turn():
                    break
