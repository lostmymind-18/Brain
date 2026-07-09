"""
OpenTelemetry tracing helpers.

Follows the OpenTelemetry GenAI Semantic Conventions:
https://opentelemetry.io/docs/specs/semconv/gen-ai/

Usage:
    from bookmind_tutor.observability.tracing import get_tracer, record_llm_attrs

    tracer = get_tracer()
    with tracer.start_as_current_span("gen_ai.complete") as span:
        span.set_attribute(GenAIAttrs.SYSTEM, "openai")
        ...

If configure_observability() was never called, get_tracer() returns a no-op
tracer — all span calls are silent no-ops. Safe to use in tests without setup.
"""
from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode

_TRACER_NAME = "bookmind_tutor"


def get_tracer() -> trace.Tracer:
    """Return the global tracer (no-op if not configured)."""
    return trace.get_tracer(_TRACER_NAME)


class GenAIAttrs:
    """Attribute name constants — GenAI Semantic Conventions."""
    SYSTEM           = "gen_ai.system"           # "openai" | "anthropic"
    REQUEST_MODEL    = "gen_ai.request.model"
    OPERATION        = "gen_ai.operation.name"   # "complete" | "stream"
    INPUT_TOKENS     = "gen_ai.usage.input_tokens"
    OUTPUT_TOKENS    = "gen_ai.usage.output_tokens"
    STOP_REASON      = "gen_ai.response.stop_reason"


class AgentAttrs:
    """Attribute name constants — BookMind agent spans."""
    STRATEGY         = "agent.strategy"
    STEP_NUM         = "agent.step_num"
    STEPS_TAKEN      = "agent.steps_taken"
    TOTAL_LLM_CALLS  = "agent.total_llm_calls"


class ToolAttrs:
    """Attribute name constants — tool execution spans."""
    NAME             = "tool.name"
    INPUT_SUMMARY    = "tool.input_summary"
    RESULT_CHARS     = "tool.result_chars"
    IS_ERROR         = "tool.is_error"


class QAAttrs:
    """Attribute name constants — QA turn spans."""
    BOOK_ID          = "qa.book_id"
    TURN_COUNT       = "qa.turn_count"
    STRATEGY         = "qa.strategy_selected"


def set_error(span: Span, exc: Exception) -> None:
    """Record an exception and set the span status to ERROR."""
    span.record_exception(exc)
    span.set_status(StatusCode.ERROR, str(exc))
