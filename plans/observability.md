# Plan: Observability (OpenTelemetry + Structured Logging)

## Tại sao cần

Observability = khả năng hiểu system đang làm gì **từ output của nó** mà không cần sửa code.
Ba trụ cột (OpenTelemetry standard):
- **Traces**: request đi qua những bước nào, mỗi bước mất bao lâu
- **Metrics**: aggregated numbers (latency p50/p99, token count, error rate)
- **Logs**: events có context đầy đủ (không chỉ là string)

Với LLM app, observability đặc biệt quan trọng vì:
- Latency không ổn định (streaming, tool calls, retries)
- Token usage ảnh hưởng trực tiếp đến chi phí
- Debugging agent loop không thể làm bằng mắt khi có nhiều bước

## Industry standard: OpenTelemetry

OpenTelemetry (OTel) là CNCF standard cho observability - vendor neutral, được support bởi AWS, GCP, Azure, Datadog, Grafana, New Relic, etc. Một lần instrument, export được đến mọi backend.

**Cho LLM apps:** OpenTelemetry đang phát triển [GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) - chuẩn hóa attribute names cho LLM traces.

```
┌─────────────────────────────────────────────────────┐
│  Application Code (instrumented with OTel spans)     │
└──────────────────┬──────────────────────────────────┘
                   │ OTLP protocol
        ┌──────────▼──────────┐
        │  OTel Collector     │  (optional, production)
        └──────┬──────┬───────┘
               │      │
    ┌──────────▼──┐ ┌──▼──────────┐
    │   Jaeger    │ │  Prometheus  │
    │  (traces)   │ │  (metrics)  │
    └─────────────┘ └─────────────┘
```

Cho development: Console exporter (no infra needed).
Cho production / local demo: Jaeger in Docker (1 command, free).

## Structured Logging với `structlog`

`structlog` là Python library phổ biến nhất cho structured logging trong production.
Khác với `logging` standard:

```python
# logging standard - khó query, thiếu context
logger.info("LLM call completed in 1.2s with 450 tokens")

# structlog - queryable, consistent, có thể add context
logger.info("llm_call_completed", latency_ms=1200, input_tokens=320, output_tokens=130, model="gpt-4o-mini")
```

Output dạng JSON trong production, colored readable format trong development.

**Killer feature:** `structlog.contextvars` - bind fields một lần (ví dụ `trace_id`, `session_id`), tự động xuất hiện trong mọi log line trong scope đó. Không cần pass context thủ công.

## Architecture

```
bookmind_tutor/observability/
  __init__.py
  setup.py           # configure_observability() - gọi 1 lần khi startup
  tracing.py         # get_tracer(), span context managers/decorators
  logging_setup.py   # configure_structlog() - replace logging với structlog

tests/observability/
  __init__.py
  test_tracing.py    # verify spans được tạo đúng
```

Không tạo thư mục riêng cho metrics - dùng OTel metrics API trực tiếp trong tracing.py.

## Instrumentation points

### 1. LLM calls (`llm/base.py` + từng client)

Mỗi `complete()` và `stream()` call được wrap trong một span:

```
Span: gen_ai.complete
Attributes:
  gen_ai.system            = "openai" | "anthropic"
  gen_ai.request.model     = "gpt-4o-mini"
  gen_ai.operation.name    = "complete" | "stream"
  gen_ai.usage.input_tokens   (set after response)
  gen_ai.usage.output_tokens  (set after response)
  gen_ai.response.stop_reason = "end_turn" | "tool_use" | "max_tokens"
  error.type               (set if exception)
```

Cách implement: `LLMClient` abstract base class thêm `_trace_complete()` wrapper method.
Các client cụ thể (Anthropic, OpenAI) gọi wrapper thay vì SDK trực tiếp.

### 2. Agent ReAct loop (`agents/harness.py`)

```
Span: agent.react_turn  [root span của một user turn]
  ├── Span: agent.react_step[1]
  │     ├── Span: gen_ai.complete
  │     ├── Span: tool.execute (book_search)
  │     └── Span: tool.execute (nếu nhiều tools)
  ├── Span: agent.react_step[2]
  │     └── Span: gen_ai.complete  [end_turn]
  ...
Attributes trên root span:
  agent.strategy          = "react"
  agent.max_iterations    = 10
  agent.steps_taken       (set khi done)
  agent.total_llm_calls   (set khi done)
```

### 3. Tool execution (`agents/executor.py`)

```
Span: tool.execute
Attributes:
  tool.name          = "book_search"
  tool.input_query   = "..." (first 100 chars)
  tool.result_chars  (set after)
  tool.is_error      (set after)
```

### 4. QA turn (`tutor/qa_agent.py`)

```
Span: qa.turn  [outermost span - context từ user click đến response]
Attributes:
  qa.strategy_selected = "react" | "plan_execute" | "reflexion"
  qa.book_id
  qa.turn_count
```

### Không instrument (explicit scope boundary)

- `VectorStore.search()` - add sau nếu cần
- `EntityExtractor._call_llm()` - secondary concern
- Metrics (counters, histograms) - add sau traces ổn định

## Setup API

```python
# bookmind_tutor/observability/setup.py

def configure_observability(
    service_name: str = "bookmind-tutor",
    export_to: str = "console",      # "console" | "otlp"
    otlp_endpoint: str = "http://localhost:4317",
) -> None:
    """
    Gọi một lần khi startup. Idempotent (gọi nhiều lần ok).

    console: in traces ra stdout, tốt cho development
    otlp:   export đến Jaeger / Grafana Tempo / Datadog qua OTLP gRPC
    """
    ...

def configure_structlog(level: str = "INFO", json_output: bool = False) -> None:
    """
    json_output=False: colored pretty output (development)
    json_output=True:  JSON lines (production / log aggregation)
    """
    ...
```

Gọi trong:
- `app.py` tại `main()` trước khi Streamlit render
- `evaluation/__main__.py` tại `main()`

Config qua env vars:
```
OTEL_EXPORT=console|otlp        # default: console
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
LOG_LEVEL=DEBUG|INFO|WARNING
LOG_FORMAT=pretty|json           # default: pretty
```

## Jaeger local setup (documentation)

Thêm `docs/local_observability.md` với hướng dẫn:

```bash
# Start Jaeger all-in-one (traces UI tại http://localhost:16686)
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest

# Chạy app với OTLP export
OTEL_EXPORT=otlp streamlit run bookmind_tutor/tutor/app.py
```

## structlog migration

Thay thế `logging.getLogger(__name__)` bằng `structlog.get_logger()` trong các file đang dùng logging:
- `agents/harness.py`
- `agents/executor.py`

Các file khác chưa có logging → giữ nguyên (add structlog khi cần).

Structlog processors chain (production):
```python
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,   # pick up trace_id etc.
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),        # JSON output
    ]
)
```

Development chain: thêm `structlog.dev.ConsoleRenderer()` thay `JSONRenderer`.

## Trace correlation với logs

Industry pattern: mỗi log line phải có `trace_id` và `span_id` để có thể correlate với traces.

```python
# Trong mỗi span, bind trace context vào structlog:
from opentelemetry import trace
import structlog

span = trace.get_current_span()
ctx = span.get_span_context()
structlog.contextvars.bind_contextvars(
    trace_id=format(ctx.trace_id, '032x'),
    span_id=format(ctx.span_id, '016x'),
)
```

Với pattern này, nếu user báo "turn X bị chậm" → lấy trace_id từ log → mở Jaeger → thấy ngay step nào chậm.

## Dependencies mới

```
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-otlp-proto-grpc>=1.20.0
structlog>=24.0.0
```

## Implementation order

1. `observability/logging_setup.py` - structlog config (no OTel dependency)
2. Replace `logging` trong `harness.py` và `executor.py` với structlog
3. `observability/setup.py` - OTel tracer provider setup (console exporter trước)
4. `observability/tracing.py` - helper functions (`get_tracer()`, span context managers)
5. Instrument `LLMClient.complete()` / `.stream()` trong base class
6. Instrument `AgentHarness.chat()` / `.stream_chat()`
7. Instrument `ToolExecutor._execute_one()`
8. Instrument `QAAgent.chat()` / `.stream_chat()`
9. Call `configure_observability()` trong `app.py`
10. Tests
11. `docs/local_observability.md` - Jaeger setup guide

## Tests

```python
# test_tracing.py
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

def test_llm_complete_creates_span(mock_client):
    """LLMClient.complete() phải tạo span với đúng attributes."""
    exporter = InMemorySpanExporter()
    # ... setup tracer với exporter
    # ... call complete()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gen_ai.complete"
    assert spans[0].attributes["gen_ai.system"] == "openai"
```

Dùng `InMemorySpanExporter` từ OTel SDK - export spans vào memory để assert trong tests. Không cần Jaeger hay network.
