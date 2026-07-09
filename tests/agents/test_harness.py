"""Tests for AgentHarness - all LLM calls are mocked via LLMClient."""
from unittest.mock import MagicMock, PropertyMock

import pytest

from bookmind_tutor.agents.harness import AgentHarness
from bookmind_tutor.agents.memory import ConversationMemory
from bookmind_tutor.agents.models import ToolSpec
from bookmind_tutor.agents.tools.base import Tool
from bookmind_tutor.llm.base import CompletionResponse, LLMClient, TokenUsage, ToolCallRequest


class _EchoTool(Tool):
    """Simple tool for testing: returns its query argument."""
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="echo",
            description="Echo the query",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}},
                          "required": ["query"]},
        )

    def execute(self, query: str) -> str:
        return f"result: {query}"


def _end_turn(text: str) -> CompletionResponse:
    return CompletionResponse(
        text=text,
        stop_reason="end_turn",
        tool_calls=[],
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        raw_message={"role": "assistant", "content": text},
    )


def _tool_use(tool_use_id: str, tool_name: str, input_: dict) -> CompletionResponse:
    raw_tc = {
        "id": tool_use_id,
        "type": "function",
        "function": {"name": tool_name, "arguments": str(input_)},
    }
    return CompletionResponse(
        text="",
        stop_reason="tool_use",
        tool_calls=[ToolCallRequest(id=tool_use_id, name=tool_name, input=input_)],
        usage=TokenUsage(input_tokens=15, output_tokens=10),
        raw_message={"role": "assistant", "content": None, "tool_calls": [raw_tc]},
    )


def _mock_client(*responses: CompletionResponse) -> MagicMock:
    """Create a mock LLMClient that returns the given CompletionResponse objects in order."""
    client = MagicMock(spec=LLMClient)
    client.complete.side_effect = list(responses)
    return client


class TestAgentHarnessEndTurn:
    def test_simple_end_turn(self):
        client = _mock_client(_end_turn("Hello world"))
        harness = AgentHarness(tools=[], llm_client=client)
        result = harness.chat("hi")

        assert result == "Hello world"
        assert client.complete.call_count == 1

    def test_passes_system_prompt(self):
        client = _mock_client(_end_turn("ok"))
        harness = AgentHarness(tools=[], llm_client=client, system_prompt="Custom system")
        harness.chat("hi")

        call_kwargs = client.complete.call_args.kwargs
        assert call_kwargs["system"] == "Custom system"

    def test_no_tools_passes_none(self):
        client = _mock_client(_end_turn("ok"))
        harness = AgentHarness(tools=[], llm_client=client)
        harness.chat("hi")

        call_kwargs = client.complete.call_args.kwargs
        assert call_kwargs.get("tools") is None


class TestAgentHarnessToolUse:
    def test_tool_use_then_end_turn(self):
        client = _mock_client(
            _tool_use("tu1", "echo", {"query": "test search"}),
            _end_turn("Based on the book, the answer is X."),
        )
        harness = AgentHarness(tools=[_EchoTool()], llm_client=client)
        result = harness.chat("What does the book say about X?")

        assert result == "Based on the book, the answer is X."
        assert client.complete.call_count == 2

    def test_tool_result_added_to_messages(self):
        client = _mock_client(
            _tool_use("tu1", "echo", {"query": "hello"}),
            _end_turn("Done."),
        )
        harness = AgentHarness(tools=[_EchoTool()], llm_client=client)
        harness.chat("question")

        # Second call should include tool result in messages (OpenAI format)
        second_call_messages = client.complete.call_args_list[1].kwargs["messages"]
        roles = [m["role"] for m in second_call_messages]
        # user → assistant (with tool_calls) → tool (result)
        assert roles == ["user", "assistant", "tool"]


class TestAgentHarnessMaxIterations:
    def test_max_iterations_stops_loop(self):
        client = _mock_client(*[_tool_use("tu1", "echo", {"query": "again"})] * 10)
        harness = AgentHarness(tools=[_EchoTool()], llm_client=client, max_iterations=3)
        result = harness.chat("loop forever")

        assert client.complete.call_count == 3
        assert "maximum" in result.lower() or "incomplete" in result.lower()


class TestAgentHarnessMemory:
    def test_stateless_no_memory_passed(self):
        client = _mock_client(_end_turn("answer 1"), _end_turn("answer 2"))
        harness = AgentHarness(tools=[], llm_client=client)
        harness.chat("q1")
        harness.chat("q2")

        # Each call starts with only its own message (no shared memory)
        first_messages = client.complete.call_args_list[0].kwargs["messages"]
        second_messages = client.complete.call_args_list[1].kwargs["messages"]
        assert len(first_messages) == 1
        assert len(second_messages) == 1

    def test_stateful_memory_accumulates(self):
        client = _mock_client(_end_turn("answer 1"), _end_turn("answer 2"))
        memory = ConversationMemory()
        harness = AgentHarness(tools=[], llm_client=client)
        harness.chat("q1", memory=memory)
        harness.chat("q2", memory=memory)

        second_messages = client.complete.call_args_list[1].kwargs["messages"]
        # q1 + a1 + q2 = 3 messages
        assert len(second_messages) == 3


class TestAgentHarnessTools:
    def test_tool_specs_passed_to_api(self):
        client = _mock_client(_end_turn("done"))
        harness = AgentHarness(tools=[_EchoTool()], llm_client=client)
        harness.chat("hi")

        call_kwargs = client.complete.call_args.kwargs
        tools_passed = call_kwargs["tools"]
        assert len(tools_passed) == 1
        # OpenAI function-tool format
        assert tools_passed[0]["type"] == "function"
        assert tools_passed[0]["function"]["name"] == "echo"

    def test_failed_tool_does_not_crash_harness(self):
        """Even if a tool raises, the harness should continue (is_error=True sent to the model)."""
        class _BrokenTool(Tool):
            @property
            def spec(self):
                return ToolSpec(name="broken", description="",
                                input_schema={"type": "object", "properties": {}})
            def execute(self, **_):
                raise RuntimeError("broken!")

        client = _mock_client(
            _tool_use("tu1", "broken", {}),
            _end_turn("Recovered."),
        )
        harness = AgentHarness(tools=[_BrokenTool()], llm_client=client)
        result = harness.chat("use broken tool")
        assert result == "Recovered."
