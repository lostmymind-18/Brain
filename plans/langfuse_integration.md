# Plan: Langfuse Integration

## Langfuse là gì và tại sao cần

OTel/Jaeger đang capture được metadata (token count, latency, strategy) nhưng không thấy được
nội dung thực tế của prompt và response.
Langfuse lấp đúng chỗ thiếu đó: mỗi LLM call hiện ra đầy đủ messages input + output text
trong một UI đẹp, kèm cost ước tính và link đến conversation context.

So sánh:
- **OTel/Jaeger**: `gen_ai.stream → input_tokens=298, latency=7.2s` (metadata only)
- **Langfuse**: thấy toàn bộ system prompt, conversation history, response text, cost breakdown

Với BookMind Tutor, Langfuse giúp:
1. Debug nhanh khi agent trả lời sai (xem chính xác prompt đã gửi)
2. Hiểu tại sao agent chọn strategy này (xem router's LLM call)
3. Tối ưu prompt cost (thấy token count theo từng step)
4. Cơ sở cho DPO/RL (Tuần 10-11): Langfuse lưu pairs question → answer để sau này label

## Langfuse concepts

```
Trace (1 per QA turn)
  ├── Span: rag-search           (tool call)
  ├── Generation: llm-stream-1   (ReAct step 1 - tool decision)
  ├── Span: book_search          (tool execution)
  └── Generation: llm-stream-2   (ReAct step 2 - final answer)
```

- **Trace**: đơn vị theo dõi cao nhất, tương ứng 1 user turn (câu hỏi → câu trả lời)
- **Generation**: LLM call, có input messages + output text + token usage
- **Span**: bất kỳ bước nào khác (tool call, retrieval, ...)

## Integration approach

Dùng **Langfuse Python SDK** (low-level API, không phải decorator).
Lý do: decorator approach không hoạt động tốt với generators (streaming) và class methods.

Langfuse là **additive** - không thay thế OTel:
- OTel tiếp tục chạy (infrastructure tracing, có thể export Jaeger sau)
- Langfuse thêm LLM-specific view với full prompt/response content

**Opt-in via env vars**: nếu không set `LANGFUSE_SECRET_KEY`, toàn bộ Langfuse code
là no-op - không crash, không warning.

## Context propagation

LLMClient và QAAgent là hai class riêng biệt.
Dùng `contextvars.ContextVar` để truyền current Langfuse trace xuống:

```python
# qa_agent.py tạo trace, set vào context var
_current_trace: ContextVar[Optional[StatefulTraceClient]] = ContextVar("langfuse_trace", default=None)

# llm/base.py đọc từ context var để attach generation vào đúng trace
trace = _current_trace.get(None)
if trace:
    gen = trace.generation(name="llm.stream", ...)
```

Vì QAAgent gọi LLMClient trong cùng thread/coroutine, `ContextVar` propagate đúng.

## Architecture

```
bookmind_tutor/observability/
  langfuse_tracing.py   # LangfuseTracer singleton + context var

bookmind_tutor/llm/
  base.py               # complete()/stream() tạo Langfuse generation

bookmind_tutor/tutor/
  qa_agent.py           # chat()/stream_chat() tạo Langfuse trace

bookmind_tutor/agents/
  executor.py           # _execute_one() tạo Langfuse span (tool calls)
```

## Files thay đổi

| File | Thay đổi |
|------|----------|
| `requirements.txt` | thêm `langfuse>=2.0.0` |
| `observability/langfuse_tracing.py` | module mới - `LangfuseTracer` singleton |
| `observability/setup.py` | thêm `configure_langfuse()` |
| `llm/base.py` | `complete()`/`stream()` tạo Langfuse generation |
| `tutor/qa_agent.py` | `chat()`/`stream_chat()` tạo Langfuse trace |
| `agents/executor.py` | `_execute_one()` tạo Langfuse span |
| `tutor/app.py` | gọi `configure_langfuse()` tại startup |
| `.env` | thêm Langfuse keys (commented placeholder) |
| `tests/observability/test_langfuse.py` | unit tests với mock Langfuse |

## `LangfuseTracer` design

```python
# observability/langfuse_tracing.py

_current_trace: ContextVar[Optional[Any]] = ContextVar("langfuse_trace", default=None)

class LangfuseTracer:
    """
    Singleton wrapper. All methods are no-ops if Langfuse is not configured.
    Callers never need to check - safe to call unconditionally.
    """

    def start_trace(self, name, input, metadata=None) -> Any:
        """Returns trace handle (or None if disabled)."""

    def end_trace(self, trace, output, metadata=None) -> None:
        """Flush trace to Langfuse."""

    def start_generation(self, name, model, provider, input_messages) -> Any:
        """Reads current trace from context var. Returns generation handle."""

    def end_generation(self, gen, output_text, usage, error=None) -> None:
        """Closes generation with output + token usage."""

    def start_span(self, name, input) -> Any:
        """Creates a span under current trace."""

    def end_span(self, span, output, error=None) -> None:
        """Closes span."""

    def set_current_trace(self, trace) -> None:
        """Set context var so downstream LLM calls attach to this trace."""

    def flush(self) -> None:
        """Flush all pending events (call at process exit or after stream)."""


_tracer = LangfuseTracer()  # module-level singleton

def get_langfuse_tracer() -> LangfuseTracer:
    return _tracer
```

## Trace content (Langfuse UI sẽ hiện)

```
Trace: qa.turn
  input:  { "question": "What are architectural characteristics?" }
  output: { "answer": "The book discusses three categories..." }
  metadata: { "book_id": "...", "strategy": "react", "turn_count": 3 }

  Generation: llm.stream (step 1)
    model: gpt-4o-mini / openai
    input: [ {role: system, ...}, {role: user, "What are..."}, ... ]
    output: <tool_call: book_search(query="architectural characteristics")>
    usage: { input: 298, output: 16 }
    latency: 1.5s

  Span: tool.execute (book_search)
    input: { "query": "architectural characteristics" }
    output: { "result_chars": 12008, "chunks": 5 }
    latency: 0.4s

  Generation: llm.stream (step 2)
    model: gpt-4o-mini / openai
    input: [ ..., tool result ... ]
    output: "The book discusses three categories: Operational..."
    usage: { input: 2645, output: 516 }
    latency: 7.2s
```

## Setup: Langfuse Cloud (free tier)

1. Tạo account tại https://cloud.langfuse.com
2. Tạo project mới
3. Lấy `Public Key` và `Secret Key` từ Settings → API Keys
4. Thêm vào `.env`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # hoặc self-hosted URL
```

Langfuse Cloud free tier: 50k observations/month.
Self-hosted: `docker compose up` từ langfuse/langfuse repo.

## Tests

Dùng `unittest.mock.patch` để mock `Langfuse` SDK - không cần account thật khi test.

```python
def test_langfuse_trace_created_per_qa_turn(mock_langfuse):
    """QAAgent.chat() phải tạo đúng 1 trace với input/output."""

def test_langfuse_generation_per_llm_call(mock_langfuse):
    """LLMClient.complete() phải tạo generation với model, input_messages, usage."""

def test_langfuse_disabled_when_no_keys(monkeypatch):
    """Nếu LANGFUSE_SECRET_KEY không set, mọi call đều no-op, không exception."""
```

## Implementation order

1. `requirements.txt` - thêm dependency
2. `observability/langfuse_tracing.py` - LangfuseTracer singleton
3. `observability/setup.py` - `configure_langfuse()`
4. `llm/base.py` - generation trong `complete()` / `stream()`
5. `tutor/qa_agent.py` - trace trong `chat()` / `stream_chat()`
6. `agents/executor.py` - span trong `_execute_one()`
7. `tutor/app.py` - gọi `configure_langfuse()` tại startup
8. `.env` - placeholder keys
9. `tests/observability/test_langfuse.py` - unit tests

## Rủi ro và giới hạn

- **Streaming generator**: `stream()` trong base.py dùng `yield from` - Langfuse generation
  phải bắt đầu trước khi yield và kết thúc sau khi generator exhausted.
  Giải pháp: wrap generator trong một generator wrapper để hook on completion.
- **Flush on Streamlit rerun**: Streamlit có thể interrupt mid-stream. Cần gọi
  `langfuse.flush()` trong finally block để không mất events.
- **Context propagation qua thread boundaries**: nếu sau này dùng asyncio hay
  thread pool, ContextVar cần được propagate thủ công.
