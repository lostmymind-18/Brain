"""
One-call observability setup for BookMind Tutor.

Call configure_observability() and configure_structlog() once at process
startup (module level in app.py / CLI main). Both are idempotent.

Observability export targets:
  "console"  — print spans to stdout (development, zero infrastructure)
  "otlp"     — gRPC export to Jaeger / Grafana Tempo / Datadog
               (set OTEL_EXPORTER_OTLP_ENDPOINT or pass otlp_endpoint)

Structured logging:
  LOG_FORMAT=pretty  → colored human-readable output (default)
  LOG_FORMAT=json    → JSON lines for log aggregation (production)
"""
from __future__ import annotations

import logging
import os

_otel_configured = False
_structlog_configured = False


def configure_observability(
    service_name: str = "bookmind-tutor",
    export_to: str | None = None,
    otlp_endpoint: str | None = None,
) -> None:
    """
    Set up the global OpenTelemetry TracerProvider.

    Args:
        service_name: Identifies this service in traces.
        export_to: "console" | "otlp". Defaults to OTEL_EXPORT env var,
                   then "console".
        otlp_endpoint: gRPC endpoint for OTLP export. Defaults to
                       OTEL_EXPORTER_OTLP_ENDPOINT, then localhost:4317.
    """
    global _otel_configured
    if _otel_configured:
        return
    _otel_configured = True

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    target = export_to or os.getenv("OTEL_EXPORT", "console")

    if target == "otlp":
        endpoint = (
            otlp_endpoint
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        )
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            # Fall back to console if OTLP exporter not installed
            _add_console_exporter(provider, SimpleSpanProcessor)
    else:
        _add_console_exporter(provider, SimpleSpanProcessor)

    trace.set_tracer_provider(provider)


def _add_console_exporter(provider, processor_cls) -> None:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    # Use ConsoleSpanExporter if available, else silent (tests)
    try:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        provider.add_span_processor(processor_cls(ConsoleSpanExporter()))
    except ImportError:
        pass


def configure_structlog(
    level: str | None = None,
    json_output: bool | None = None,
) -> None:
    """
    Configure structlog for the process.

    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING"). Defaults to
               LOG_LEVEL env var, then "INFO".
        json_output: True for JSON lines (production). Defaults to
                     LOG_FORMAT=json env var, then False (pretty).
    """
    global _structlog_configured
    if _structlog_configured:
        return
    _structlog_configured = True

    import structlog

    log_level_str = level or os.getenv("LOG_LEVEL", "INFO")
    use_json = json_output if json_output is not None else (
        os.getenv("LOG_FORMAT", "pretty").lower() == "json"
    )

    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(message)s")

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        # add_logger_name requires logging.Logger.name and is incompatible
        # with PrintLoggerFactory — omit it when not using stdlib integration.
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if use_json:
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def configure_langfuse() -> None:
    """Thin re-export so callers can import everything from one place."""
    from bookmind_tutor.observability.langfuse_tracing import configure_langfuse as _cfg
    _cfg()


def reset_for_testing() -> None:
    """
    Reset setup flags and OTel global state so tests can reconfigure from scratch.

    Uses private OTel internals (_TRACER_PROVIDER, _TRACER_PROVIDER_SET_ONCE)
    to allow multiple set_tracer_provider() calls within the same process.
    This is intentionally test-only — never call in production code.
    """
    global _otel_configured, _structlog_configured
    _otel_configured = False
    _structlog_configured = False

    # Reset OTel global so tests can install their own InMemorySpanExporter.
    import opentelemetry.trace as otel_trace
    otel_trace._TRACER_PROVIDER = None
    if hasattr(otel_trace, "_TRACER_PROVIDER_SET_ONCE"):
        otel_trace._TRACER_PROVIDER_SET_ONCE._done = False
