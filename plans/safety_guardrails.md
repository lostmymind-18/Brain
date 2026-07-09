# Plan: Safety & Guardrails Layer

## Tại sao cần

Production LLM apps luôn phải có guardrails vì 3 lý do:
1. **Prompt injection** - user cố override system prompt để bypass restrictions
2. **Off-topic abuse** - user dùng chatbot không đúng mục đích (chatbot sách bị dùng làm general assistant)
3. **Input quality** - quá dài, format bất thường → tốn token, kết quả kém

Guardrails là lớp **trước và sau** LLM call - không phải trong LLM. Mục tiêu: fail fast, fail explicitly, fail gracefully.

## Industry pattern

```
User Input
   ↓
[InputGuardLayer]  ← short-circuit nếu vi phạm, trả về GuardResult.blocked
   ↓ (pass)
QAAgent.stream_chat()
   ↓
[OutputGuardLayer] ← check response trước khi render
   ↓ (pass)
UI render
```

Guards chạy theo thứ tự, short-circuit khi gặp `severity == "block"`. Guard với `severity == "warn"` log warning nhưng vẫn cho qua.

## Architecture

```
bookmind_tutor/safety/
  __init__.py
  models.py          # GuardResult, ViolationType, Severity
  input_guards.py    # LengthGuard, PromptInjectionGuard, TopicalityGuard
  output_guards.py   # OutputLengthGuard
  layer.py           # GuardrailsLayer - orchestrates input + output guards

tests/safety/
  __init__.py
  test_input_guards.py
  test_output_guards.py
  test_layer.py
```

## Data models (`models.py`)

```python
class Severity(str, Enum):
    BLOCK = "block"   # hard block, return error to user
    WARN  = "warn"    # log but allow through
    LOG   = "log"     # silent track only

class ViolationType(str, Enum):
    INPUT_TOO_LONG      = "input_too_long"
    PROMPT_INJECTION    = "prompt_injection"
    OFF_TOPIC           = "off_topic"
    OUTPUT_TOO_LONG     = "output_too_long"

@dataclass
class GuardContext:
    book_name: str = ""          # used by TopicalityGuard
    book_id: str = ""
    session_id: str = ""

@dataclass
class GuardResult:
    passed: bool
    violation_type: ViolationType | None = None
    severity: Severity | None = None
    user_message: str = ""       # shown to user when blocked
    internal_detail: str = ""    # logged internally, NOT shown to user
```

## Input Guards

### 1. `LengthGuard` (rule-based, O(1))

```python
class LengthGuard:
    def __init__(self, max_chars: int = 2000) -> None: ...
    def check(self, text: str, ctx: GuardContext) -> GuardResult: ...
```

- Lý do 2000 chars: tương đương ~500 tokens, đủ cho câu hỏi phức tạp nhất
- `severity=BLOCK`, `user_message="Câu hỏi quá dài. Hãy chia nhỏ thành nhiều câu hỏi."`

### 2. `PromptInjectionGuard` (rule-based regex)

Phát hiện các pattern phổ biến. Không cần LLM - regex đủ nhanh và đủ tốt cho common patterns.

Pattern categories:
```
ROLE_OVERRIDE:  r"ignore (all |previous |prior )?(instructions?|prompts?|rules?)"
ROLE_OVERRIDE:  r"you are now|act as|pretend (you are|to be)|your new (role|persona)"
SYSTEM_LEAK:    r"(repeat|print|show|reveal|tell me) (your |the )?(system |initial )?prompt"
JAILBREAK:      r"jailbreak|dan mode|developer mode|no restrictions"
DELIMITER_INJ:  r"(</?(system|human|assistant|user)>|<\|im_start\|>|\[INST\])"
```

- `severity=BLOCK`, `user_message="Câu hỏi không hợp lệ."`
- Log `internal_detail` với pattern nào match (để monitor)

### 3. `TopicalityGuard` (LLM-based, optional)

Gọi LLM với model rẻ để hỏi: "Câu hỏi này có liên quan đến cuốn sách không?"

```python
class TopicalityGuard:
    def __init__(
        self,
        llm_client: LLMClient,
        enabled: bool = True,    # có thể tắt trong test
    ) -> None: ...
    def check(self, text: str, ctx: GuardContext) -> GuardResult: ...
```

Prompt ngắn gọn:
```
Book: "{book_name}"
Question: "{text}"
Is this question related to the book above? Answer ONLY "yes" or "no".
```

- `max_tokens=5` - chỉ cần yes/no
- Nếu LLM fail → `passed=True` (fail open, không block user vì lỗi guard)
- `severity=WARN` (log but allow) thay vì BLOCK - vì false positive rate của rule đơn giản này cao
- Thực tế: warn mode cho phép monitor mà không break UX

**Lý do dùng WARN không phải BLOCK cho topicality:**
False positive quá nguy hiểm cho UX - user hỏi "cuốn sách này có liên quan đến distributed systems không?" bị block là rất bad experience. WARN mode cho phép thu thập data để tune sau.

## Output Guards

### `OutputLengthGuard` (rule-based)

```python
class OutputLengthGuard:
    def __init__(self, max_chars: int = 8000) -> None: ...
    def check(self, text: str, ctx: GuardContext) -> GuardResult: ...
```

- `severity=WARN` - log nếu response quá dài nhưng vẫn cho hiển thị
- 8000 chars tương đương ~2000 tokens - đủ cho response dài nhất

**Không implement:**
- `GroundingGuard` (output hallucination check) - quá đắt (1 LLM call/response), để TODO

## GuardrailsLayer (`layer.py`)

```python
class GuardrailsLayer:
    def __init__(
        self,
        input_guards: list,
        output_guards: list,
    ) -> None: ...

    def check_input(self, text: str, ctx: GuardContext) -> GuardResult:
        """Run all input guards in order. Short-circuit on BLOCK."""
        ...

    def check_output(self, text: str, ctx: GuardContext) -> GuardResult:
        """Run all output guards in order. Short-circuit on BLOCK."""
        ...
```

Short-circuit logic:
- Nếu guard trả về `severity == BLOCK` → dừng, trả về result đó
- Nếu `severity == WARN` → log warning, tiếp tục
- Tất cả pass → trả về `GuardResult(passed=True)`

## Integration trong `app.py`

```python
# _init_state(): khởi tạo guardrails
st.session_state.guardrails = GuardrailsLayer(
    input_guards=[
        LengthGuard(max_chars=2000),
        PromptInjectionGuard(),
        TopicalityGuard(llm_client=st.session_state.llm_client),
    ],
    output_guards=[
        OutputLengthGuard(max_chars=8000),
    ],
)

# _handle_question(): check input trước khi chat
ctx = GuardContext(
    book_name=st.session_state.book_names.get(bid, ""),
    book_id=bid,
)
guard_result = st.session_state.guardrails.check_input(question, ctx)
if not guard_result.passed and guard_result.severity == Severity.BLOCK:
    st.error(guard_result.user_message)
    return
```

## Logging cho guardrails

Mỗi violation phải được log với structured fields:
```json
{
  "event": "guardrail_violation",
  "violation_type": "prompt_injection",
  "severity": "block",
  "book_id": "...",
  "detail": "matched pattern: ROLE_OVERRIDE",
  "input_preview": "ignore all prev..."  // first 50 chars only, no PII
}
```

## Implementation order

1. `models.py` - data models
2. `input_guards.py` - LengthGuard, PromptInjectionGuard (no LLM dependency)
3. `output_guards.py` - OutputLengthGuard
4. `layer.py` - GuardrailsLayer
5. Tests cho 1-4
6. `TopicalityGuard` (LLM dependency - sau cùng vì cần mock)
7. Integration trong `app.py`

## Dependencies

Không cần dependency mới - tất cả rule-based guards chỉ dùng `re`. TopicalityGuard dùng `LLMClient` đã có.

## Không implement (explicit scope boundary)

- `ModerationGuard` (OpenAI Moderation API) - nice to have, để TODO
- `GroundingGuard` (hallucination check) - quá đắt per-response
- Rate limiting per session - Streamlit đã isolate session, low priority
- PII scrubbing trong input - scope riêng, để TODO
