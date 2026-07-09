"""Observability for BookMind Tutor: OpenTelemetry traces + structlog."""
from bookmind_tutor.observability.setup import configure_observability, configure_structlog
from bookmind_tutor.observability.tracing import AgentAttrs, GenAIAttrs, QAAttrs, ToolAttrs, get_tracer

__all__ = [
    "AgentAttrs",
    "GenAIAttrs",
    "QAAttrs",
    "ToolAttrs",
    "configure_observability",
    "configure_structlog",
    "get_tracer",
]
