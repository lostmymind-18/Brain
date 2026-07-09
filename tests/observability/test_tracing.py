"""
Tests for the OTel tracing layer.

Uses InMemorySpanExporter so spans are captured without any real infra.
Each test resets the OTel provider to prevent state leaking between tests.
"""
from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from bookmind_tutor.observability.setup import reset_for_testing
from bookmind_tutor.observability.tracing import (
    AgentAttrs,
    GenAIAttrs,
    QAAttrs,
    ToolAttrs,
    get_tracer,
    set_error,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_tracer():
    """Each test gets a clean TracerProvider backed by InMemorySpanExporter."""
    reset_for_testing()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    reset_for_testing()
    trace.set_tracer_provider(trace.ProxyTracerProvider())


# ---------------------------------------------------------------------------
# get_tracer()
# ---------------------------------------------------------------------------

class TestGetTracer:
    def test_returns_tracer(self, isolated_tracer):
        t = get_tracer()
        assert t is not None

    def test_tracer_creates_spans(self, isolated_tracer):
        t = get_tracer()
        with t.start_as_current_span("test.span"):
            pass
        spans = isolated_tracer.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "test.span"


# ---------------------------------------------------------------------------
# set_error()
# ---------------------------------------------------------------------------

class TestSetError:
    def test_records_exception(self, isolated_tracer):
        t = get_tracer()
        exc = ValueError("boom")
        with t.start_as_current_span("err.span") as span:
            set_error(span, exc)
        spans = isolated_tracer.get_finished_spans()
        span = spans[0]
        assert span.status.status_code.name == "ERROR"
        events = span.events
        assert any(e.name == "exception" for e in events)


# ---------------------------------------------------------------------------
# Attribute constants (sanity check — strings are stable contracts)
# ---------------------------------------------------------------------------

class TestAttrConstants:
    def test_gen_ai_attrs(self):
        assert GenAIAttrs.SYSTEM == "gen_ai.system"
        assert GenAIAttrs.REQUEST_MODEL == "gen_ai.request.model"
        assert GenAIAttrs.INPUT_TOKENS == "gen_ai.usage.input_tokens"
        assert GenAIAttrs.OUTPUT_TOKENS == "gen_ai.usage.output_tokens"
        assert GenAIAttrs.STOP_REASON == "gen_ai.response.stop_reason"
        assert GenAIAttrs.OPERATION == "gen_ai.operation.name"

    def test_agent_attrs(self):
        assert AgentAttrs.STRATEGY == "agent.strategy"
        assert AgentAttrs.STEP_NUM == "agent.step_num"
        assert AgentAttrs.TOTAL_LLM_CALLS == "agent.total_llm_calls"

    def test_tool_attrs(self):
        assert ToolAttrs.NAME == "tool.name"
        assert ToolAttrs.INPUT_SUMMARY == "tool.input_summary"
        assert ToolAttrs.RESULT_CHARS == "tool.result_chars"
        assert ToolAttrs.IS_ERROR == "tool.is_error"

    def test_qa_attrs(self):
        assert QAAttrs.TURN_COUNT == "qa.turn_count"
        assert QAAttrs.STRATEGY == "qa.strategy_selected"


# ---------------------------------------------------------------------------
# Span attribute setting
# ---------------------------------------------------------------------------

class TestSpanAttributes:
    def test_attributes_set_on_span(self, isolated_tracer):
        t = get_tracer()
        with t.start_as_current_span("gen_ai.complete") as span:
            span.set_attribute(GenAIAttrs.SYSTEM, "anthropic")
            span.set_attribute(GenAIAttrs.REQUEST_MODEL, "claude-haiku-4-5-20251001")
            span.set_attribute(GenAIAttrs.INPUT_TOKENS, 100)
            span.set_attribute(GenAIAttrs.OUTPUT_TOKENS, 50)

        spans = isolated_tracer.get_finished_spans()
        attrs = spans[0].attributes
        assert attrs[GenAIAttrs.SYSTEM] == "anthropic"
        assert attrs[GenAIAttrs.REQUEST_MODEL] == "claude-haiku-4-5-20251001"
        assert attrs[GenAIAttrs.INPUT_TOKENS] == 100
        assert attrs[GenAIAttrs.OUTPUT_TOKENS] == 50

    def test_nested_spans(self, isolated_tracer):
        t = get_tracer()
        with t.start_as_current_span("parent"):
            with t.start_as_current_span("child"):
                pass

        spans = isolated_tracer.get_finished_spans()
        assert len(spans) == 2
        names = {s.name for s in spans}
        assert names == {"parent", "child"}
        child = next(s for s in spans if s.name == "child")
        parent = next(s for s in spans if s.name == "parent")
        assert child.parent.span_id == parent.context.span_id
