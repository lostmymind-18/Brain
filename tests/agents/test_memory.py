"""Tests for ConversationMemory (OpenAI format)."""
import pytest

from bookmind_tutor.agents.memory import ConversationMemory
from bookmind_tutor.agents.models import ToolCallResult


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
        # user + assistant + tool result
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

    def test_max_turns_trims_oldest(self):
        mem = ConversationMemory(max_turns=2)
        for i in range(4):
            mem.add_user(f"question {i}")
            mem.add_assistant({"role": "assistant", "content": f"answer {i}"})
        # max_turns=2 means max 4 messages; older pairs should have been dropped
        assert len(mem) <= 4

    def test_max_turns_none_no_trim(self):
        mem = ConversationMemory(max_turns=None)
        for i in range(20):
            mem.add_user(f"q{i}")
            mem.add_assistant({"role": "assistant", "content": f"a{i}"})
        assert len(mem) == 40
