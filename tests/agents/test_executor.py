"""Tests for ToolExecutor."""
from unittest.mock import MagicMock

import pytest

from bookmind_tutor.agents.executor import ToolExecutor
from bookmind_tutor.agents.models import ToolSpec
from bookmind_tutor.agents.tools.base import Tool
from bookmind_tutor.llm.base import ToolCallRequest


class _EchoTool(Tool):
    """Returns the query argument as-is."""
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="echo",
            description="Echo input",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        )

    def execute(self, query: str) -> str:
        return f"echo: {query}"


class _FailTool(Tool):
    """Always raises a ValueError."""
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="fail", description="Always fails",
                        input_schema={"type": "object", "properties": {}})

    def execute(self, **_) -> str:
        raise ValueError("deliberate failure")


class _TransientTool(Tool):
    """Fails with ConnectionError for first N calls, then succeeds."""
    def __init__(self, fail_times: int):
        self._calls = 0
        self._fail_times = fail_times

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="transient", description="Transient failures",
                        input_schema={"type": "object", "properties": {}})

    def execute(self) -> str:
        self._calls += 1
        if self._calls <= self._fail_times:
            raise ConnectionError("network down")
        return "finally worked"


class TestToolExecutorDispatch:
    def test_executes_registered_tool(self):
        ex = ToolExecutor([_EchoTool()])
        tc = ToolCallRequest(id="tu1", name="echo", input={"query": "hello"})
        results = ex.execute_all([tc])
        assert len(results) == 1
        assert results[0].tool_use_id == "tu1"
        assert results[0].content == "echo: hello"
        assert not results[0].is_error

    def test_unknown_tool_returns_error_result(self):
        ex = ToolExecutor([_EchoTool()])
        tc = ToolCallRequest(id="tu1", name="missing_tool", input={})
        results = ex.execute_all([tc])
        assert results[0].is_error
        assert "missing_tool" in results[0].content

    def test_executes_multiple_tools(self):
        ex = ToolExecutor([_EchoTool()])
        tcs = [
            ToolCallRequest(id="tu1", name="echo", input={"query": "a"}),
            ToolCallRequest(id="tu2", name="echo", input={"query": "b"}),
        ]
        results = ex.execute_all(tcs)
        assert len(results) == 2
        assert results[0].content == "echo: a"
        assert results[1].content == "echo: b"

    def test_empty_list_returns_empty(self):
        ex = ToolExecutor([_EchoTool()])
        assert ex.execute_all([]) == []


class TestToolExecutorErrorHandling:
    def test_non_retryable_error_returns_error_result(self):
        ex = ToolExecutor([_FailTool()], max_retries=3)
        tc = ToolCallRequest(id="tu1", name="fail", input={})
        results = ex.execute_all([tc])
        assert results[0].is_error
        assert "deliberate failure" in results[0].content

    def test_transient_error_retries_and_succeeds(self):
        tool = _TransientTool(fail_times=1)
        ex = ToolExecutor([tool], max_retries=3, retry_delay=0)
        tc = ToolCallRequest(id="tu1", name="transient", input={})
        results = ex.execute_all([tc])
        assert not results[0].is_error
        assert results[0].content == "finally worked"
        assert tool._calls == 2

    def test_transient_error_exhausts_retries(self):
        tool = _TransientTool(fail_times=10)
        ex = ToolExecutor([tool], max_retries=2, retry_delay=0)
        tc = ToolCallRequest(id="tu1", name="transient", input={})
        results = ex.execute_all([tc])
        assert results[0].is_error
        assert tool._calls == 2
