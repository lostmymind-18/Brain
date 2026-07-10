"""Tests for ConversationMemory (OpenAI format)."""
import pytest

from bookmind_tutor.agents.memory import ConversationMemory
from bookmind_tutor.agents.models import ToolCallResult


def _user(i: int) -> dict:
    return {"role": "user", "content": f"question {i}"}


def _assistant(i: int) -> dict:
    return {"role": "assistant", "content": f"answer {i}"}


def _tool_result(content: str, call_id: str = "c1") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class TestConversationMemory:
    def test_empty_initially(self):
        mem = ConversationMemory()
        assert mem.messages() == []
        assert len(mem) == 0

    def test_add_user_string(self):
        mem = ConversationMemory()
        mem.add_user("Hello")
        msgs = mem.messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"

    def test_add_assistant_dict(self):
        mem = ConversationMemory()
        raw = {"role": "assistant", "content": "Hi there"}
        mem.add_assistant(raw)
        assert mem.messages()[0]["role"] == "assistant"
        assert mem.messages()[0]["content"] == "Hi there"

    def test_add_tool_results(self):
        mem = ConversationMemory()
        mem.add_user("search for X")
        mem.add_assistant({"role": "assistant", "content": None, "tool_calls": [{"id": "tu1"}]})
        results = [ToolCallResult(tool_use_id="tu1", content="result text")]
        mem.add_tool_results(results)
        msgs = mem.messages()
        assert len(msgs) == 3
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["tool_call_id"] == "tu1"
        assert msgs[2]["content"] == "result text"

    def test_add_empty_tool_results_noop(self):
        mem = ConversationMemory()
        mem.add_tool_results([])
        assert len(mem) == 0

    def test_clear(self):
        mem = ConversationMemory()
        mem.add_user("hi")
        mem.clear()
        assert len(mem) == 0

    def test_messages_returns_copy(self):
        mem = ConversationMemory()
        mem.add_user("hi")
        msgs = mem.messages()
        msgs.pop()
        assert len(mem) == 1  # original not mutated

    # ------------------------------------------------------------------
    # max_turns behaviour
    # ------------------------------------------------------------------

    def test_max_turns_trims_oldest(self):
        mem = ConversationMemory(max_turns=2, max_tokens=None)
        for i in range(4):
            mem.add_user(f"question {i}")
            mem.add_assistant({"role": "assistant", "content": f"answer {i}"})
        # Only the last 2 user turns should survive.
        user_msgs = [m for m in mem.messages() if m["role"] == "user"]
        assert len(user_msgs) == 2
        assert user_msgs[0]["content"] == "question 2"
        assert user_msgs[1]["content"] == "question 3"

    def test_max_turns_none_no_trim(self):
        mem = ConversationMemory(max_turns=None, max_tokens=None)
        for i in range(20):
            mem.add_user(f"q{i}")
            mem.add_assistant({"role": "assistant", "content": f"a{i}"})
        assert len(mem) == 40

    # ------------------------------------------------------------------
    # _drop_oldest_turn: must remove a complete logical turn, not just 2 messages
    # ------------------------------------------------------------------

    def test_drop_oldest_turn_includes_tool_results(self):
        # Turn 0: user + assistant(tool_call) + tool_result + assistant(final)
        # Turn 1: user + assistant
        # Dropping turn 0 should remove all 4 messages, leaving turn 1 intact.
        mem = ConversationMemory(max_turns=None, max_tokens=None)
        mem._messages = [
            {"role": "user", "content": "q0"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "big tool result"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        dropped = mem._drop_oldest_turn()
        assert dropped is True
        msgs = mem.messages()
        # Only turn 1 remains; no orphaned tool messages.
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "q1"

    def test_drop_oldest_turn_returns_false_when_only_one_turn(self):
        mem = ConversationMemory(max_turns=None, max_tokens=None)
        mem._messages = [
            {"role": "user", "content": "only question"},
            {"role": "assistant", "content": "only answer"},
        ]
        assert mem._drop_oldest_turn() is False
        assert len(mem) == 2  # nothing removed

    # ------------------------------------------------------------------
    # max_tokens behaviour
    # ------------------------------------------------------------------

    def test_max_tokens_trims_when_budget_exceeded(self):
        # Each turn has a large tool result; after several turns the token
        # budget is exceeded and the oldest turn must be evicted.
        large_content = "x" * 4000  # ~1 000 tokens per tool result
        mem = ConversationMemory(max_turns=None, max_tokens=2000)

        for i in range(5):
            mem._messages.append({"role": "user", "content": f"q{i}"})
            mem._messages.append({"role": "assistant", "content": None, "tool_calls": []})
            mem._messages.append({"role": "tool", "tool_call_id": "c", "content": large_content})
            mem._messages.append({"role": "assistant", "content": f"a{i}"})
        # Manually trigger trim (normally called after add_user/add_tool_results).
        mem._trim_if_needed()

        assert mem._estimated_tokens() <= 2000
        # The surviving messages must start with a user message (no orphans).
        assert mem.messages()[0]["role"] == "user"

    def test_max_tokens_none_no_trim_by_tokens(self):
        large_content = "x" * 40_000
        mem = ConversationMemory(max_turns=None, max_tokens=None)
        mem._messages = [
            {"role": "user", "content": "q"},
            {"role": "tool", "tool_call_id": "c", "content": large_content},
        ]
        before = len(mem)
        mem._trim_if_needed()
        assert len(mem) == before  # nothing trimmed

    def test_token_trim_preserves_last_turn(self):
        # Even if the last turn alone exceeds max_tokens, it must not be dropped.
        huge_content = "y" * 400_000  # ~100k tokens, above any reasonable limit
        mem = ConversationMemory(max_turns=None, max_tokens=1000)
        mem._messages = [
            {"role": "user", "content": "only question"},
            {"role": "tool", "tool_call_id": "c", "content": huge_content},
        ]
        mem._trim_if_needed()
        # The single turn cannot be dropped; memory stays intact.
        assert mem.messages()[0]["content"] == "only question"

    def test_estimated_tokens_approximation(self):
        mem = ConversationMemory(max_turns=None, max_tokens=None)
        mem._messages = [{"role": "user", "content": "a" * 400}]
        # values: "user" (4 chars) + "a"*400 (400 chars) = 404 // 4 = 101
        assert mem._estimated_tokens() == 101
