# Plan: QAAgent

## Bối cảnh

Roadmap tuần 5-6 yêu cầu một `QAAgent` reactive - trả lời tốt câu hỏi của user, có mix
Socratic questioning và Feynman check khi phù hợp, và điều chỉnh độ sâu câu trả lời theo
cách user đặt câu hỏi (student modeling nhẹ). Không phải proactive tutor tự dẫn dắt.

## Thiết kế

### Cơ chế cốt lõi: dynamic system prompt

`AgentHarness` đã nhận `system_prompt` lúc construction. Thay vì tạo lại harness mỗi turn,
ta thêm một `system_prompt` setter để `QAAgent` cập nhật trước mỗi lần gọi.

```
QAAgent.chat(question)
  │
  ├── Infer user level from conversation history
  ├── Decide pedagogical additions (Socratic? Feynman?)
  ├── Build system prompt: base + level adaptation + pedagogical hints
  ├── harness.system_prompt = new_prompt
  └── router.chat(question, memory) → answer
```

### Student modeling nhẹ

Phân tích N message cuối của user bằng keyword heuristic — không LLM call thêm.

| Signal | Beginner | Intermediate/Advanced |
|---|---|---|
| Từ khoá | "what is", "explain", "i don't understand", "how does", "what does" | "why", "tradeoff", "compare", "edge case", "when should i", "how does X differ" |
| Độ dài câu hỏi | Ngắn (< 10 từ) | Dài, nhiều điều kiện |

Kết quả: `"beginner"` / `"intermediate"` / `"advanced"`. Cập nhật mỗi turn.

Level → thay đổi system prompt:
- **beginner**: "Use plain language and relatable analogies. Define technical terms before using them."
- **intermediate**: "Assume familiarity with basic concepts. Highlight nuance and tradeoffs."
- **advanced**: "Engage at depth. Discuss edge cases, failure modes, and technical precision."

### Socratic mixing

Không phải mặc định. Trigger: sau mỗi 3 turn, nếu câu hỏi không phải lookup đơn giản
(không có "what is" hay "define").

Khi trigger: thêm vào system prompt:
> "If it feels natural, end with one thought-provoking question to invite the user to think
> deeper — but only if it genuinely adds value. Do not force it."

Không trigger trên câu hỏi factual thuần túy ("what is X?") vì Socratic question ở đó sẽ lạc lõng.

### Feynman check

Trigger: sau 5+ turn trong session, nếu user vẫn đang hỏi về cùng topic domain
(kiểm tra bằng user KG entities xuất hiện trong history).

Khi trigger: thêm vào system prompt:
> "If the user seems to be building solid familiarity with this topic, you may (optionally)
> suggest they try explaining a key concept in their own words as a quick self-check."

Chỉ "may" và "optionally" — LLM tự quyết định có mix vào không dựa trên context.

### Không trigger nếu:
- User đang hỏi lookup nhanh ("what page is X on?")
- Câu hỏi rất ngắn (< 5 từ)
- User vừa mới bắt đầu session (< 2 turn)

## Components

### 1. `AgentHarness` — thêm `system_prompt` setter

```python
@property
def system_prompt(self) -> str:
    return self._system

@system_prompt.setter
def system_prompt(self, value: str) -> None:
    self._system = value
```

Thay đổi tối thiểu, backwards compatible.

### 2. `QAAgent` — `bookmind_tutor/tutor/qa_agent.py`

```python
class QAAgent:
    def __init__(self, harness: AgentHarness, router: StrategyRouter) -> None: ...
    def chat(self, question: str) -> tuple[str, ReasoningTrace | None]: ...
    def _infer_level(self) -> str: ...          # "beginner" / "intermediate" / "advanced"
    def _should_socratic(self, question: str) -> bool: ...
    def _should_feynman(self) -> bool: ...
    def _build_system_prompt(self, question: str) -> str: ...
```

Holds `ConversationMemory` internally (multi-turn context across the session).

### 3. `app.py` — dùng `QAAgent` thay `StrategyRouter` trực tiếp

Trong `_setup_agent()`: tạo `QAAgent(harness=harness, router=router)`, lưu vào
`session_state.qa_agent`.

Trong `_handle_question()`: thay `router.chat(question)` bằng `qa_agent.chat(question)`.

`last_trace` lấy từ `router.last_trace` vẫn như cũ.

## Tests

- `test_qa_agent.py`: unit test `_infer_level`, `_should_socratic`, `_should_feynman` với
  mock history. Mock `router.chat` để không cần LLM.
- Test: beginner signal → correct level
- Test: intermediate signals → correct level
- Test: Socratic trigger sau 3 turns với non-factual question
- Test: Socratic không trigger với factual question ("what is X?")
- Test: Feynman trigger sau 5+ turns
- Test: system prompt chứa đúng level text khi build

## Phạm vi không làm

- Không adaptive curriculum (không tự tạo lộ trình học)
- Không proactive initiation (QAAgent chỉ reply, không tự hỏi trước)
- Không lưu user level vào Neo4j / persist across sessions (lightweight in-memory đủ rồi)
- Không streaming
