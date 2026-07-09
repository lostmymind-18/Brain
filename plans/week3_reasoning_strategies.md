# Week 3: Advanced Reasoning Strategies

## Mục tiêu học

Tuần này không xây thêm feature mới cho người dùng - mục tiêu là **hiểu sâu hơn về cách agent ra quyết định** bằng cách implement nhiều reasoning pattern khác nhau trên cùng một harness cũ, rồi so sánh chúng trực tiếp.

Sau tuần này nên hiểu được:
- Tại sao ReAct không phải lúc nào cũng là pattern tốt nhất.
- Khi nào thì Plan-and-Execute có lợi thế.
- Reflexion giải quyết vấn đề gì mà ReAct không làm được.
- Dynamic routing là gì và trade-off khi thêm nó vào.

---

## Background: 3 papers cần đọc trước

| Paper | Ý tưởng cốt lõi |
|---|---|
| ReAct (Yao et al. 2022) | Đã implement ở Tuần 2 — interleave Reason + Act trong từng bước |
| Plan-and-Execute / Plan-and-Solve (Wang et al. 2023) | Tách riêng bước lập kế hoạch trước, rồi mới thực thi từng bước |
| Reflexion (Shinn et al. 2023) | Sau khi có kết quả, agent tự đánh giá và thử lại nếu chưa đủ tốt |

---

## Kiến trúc đề xuất

### Nguyên tắc thiết kế

- **Không duplicate `AgentHarness`** - mọi strategy đều dùng lại `AgentHarness` (ReAct loop) ở tầng thực thi.
  Các strategy chỉ khác nhau ở *cách tổ chức prompt và vòng lặp bên ngoài* trước/sau khi gọi harness.
- **Interface đồng nhất** - tất cả strategy đều expose `chat(question, memory)` để `StrategyRouter` và script demo có thể gọi như nhau.
- **Không thay đổi code Tuần 2** - `AgentHarness`, `ToolExecutor`, `ConversationMemory` giữ nguyên.

### Cấu trúc thư mục mới

```
bookmind_tutor/agents/
  strategies/
    __init__.py
    base.py           # ReasoningStrategy protocol
    react.py          # thin wrapper — delegates thẳng vào AgentHarness (hiện tại)
    plan_execute.py   # Plan-and-Execute strategy
    reflexion.py      # Reflexion strategy
    router.py         # StrategyRouter — chọn strategy dựa vào câu hỏi
```

---

## Chi tiết từng strategy

### 1. ReAct (đã có — bọc lại)

`react.py` chỉ là thin wrapper quanh `AgentHarness.chat()`, không thêm logic.
Mục đích: đưa ReAct vào cùng interface với các strategy khác để có thể so sánh.

```
ReActStrategy.chat(question)
  → AgentHarness.chat(question)   # y chang Tuần 2
```

---

### 2. Plan-and-Execute

**Vấn đề ReAct gặp phải:**
ReAct quyết định bước tiếp theo sau *mỗi* tool call.
Với câu hỏi phức tạp (vd: "So sánh cách DDIA và Fundamentals of Software Architecture tiếp cận vấn đề scalability"), ReAct có thể đi lạc, lặp lại tool call, hoặc quên mất sub-question đã đặt ra.

**Cách Plan-and-Execute giải quyết:**
Tách thành 2 phase rõ ràng:

```
Phase 1 — Plan:
  Gọi Claude (không dùng tool) để decompose câu hỏi thành danh sách sub-tasks.
  Output: ["1. Tìm định nghĩa scalability trong DDIA", "2. Tìm...", ...]

Phase 2 — Execute:
  Với mỗi sub-task, gọi AgentHarness.chat() để thực thi (có tool access).
  Tích lũy kết quả từng bước.

Phase 3 — Synthesize:
  Gọi Claude lần cuối để tổng hợp tất cả kết quả thành câu trả lời hoàn chỉnh.
```

**Structured output cho bước Plan:**
Claude trả về JSON list để parse dễ dàng:
```json
["sub-task 1", "sub-task 2", "sub-task 3"]
```
Dùng kỹ thuật extract JSON đã học ở Tuần 1 (tìm `[` đến `]` trước khi parse).

**Giới hạn:**
- Max 5 sub-tasks để tránh quá nhiều API call.
- Nếu parse JSON thất bại, fallback về ReAct thay vì crash.

---

### 3. Reflexion

**Vấn đề ReAct gặp phải:**
ReAct không tự đánh giá chất lượng câu trả lời của mình.
Nếu search trả về kết quả mơ hồ, Claude vẫn có thể trả lời tự tin nhưng thiếu độ sâu.

**Cách Reflexion giải quyết:**
Sau khi có câu trả lời đầu tiên, thêm một vòng lặp "đánh giá - cải thiện":

```
Round 1:
  AgentHarness.chat(question) → answer_v1

Reflection:
  Gọi Claude để tự đánh giá answer_v1 theo 3 tiêu chí:
    - Có trích dẫn cụ thể từ sách không?
    - Câu trả lời có đủ độ sâu cho câu hỏi này không?
    - Còn điểm nào quan trọng bị bỏ sót không?
  Output: {"score": 1-5, "missing": "...", "should_retry": true/false}

Round 2 (chỉ nếu should_retry=True và score < threshold):
  AgentHarness.chat(question + reflection_feedback) → answer_v2
  Trả về answer_v2.
```

**Giới hạn:**
- Max 2 rounds (1 lần thử lại) để tránh chi phí API tăng cao.
- `score_threshold` mặc định là 4/5 — chỉ retry nếu điểm dưới 4.
- Nếu parse reflection thất bại, trả về answer_v1 không retry.

---

### 4. StrategyRouter

Router nhẹ - dùng heuristic đơn giản trước, không cần thêm LLM call.

**Routing rules:**

| Điều kiện | Strategy được chọn |
|---|---|
| Câu hỏi có từ khóa "compare", "difference", "vs", "contrast", "both" | Plan-and-Execute |
| Câu hỏi có từ khóa "explain in depth", "detailed", "comprehensively" | Reflexion |
| Mặc định | ReAct |

Vì đây là heuristic thô, ghi rõ limitation trong docstring.
Nếu sau khi test thấy heuristic sai nhiều, có thể nâng cấp thành 1 LLM call nhỏ để classify — nhưng chỉ làm nếu còn thời gian (ghi vào backlog).

---

## Interface thống nhất

```python
# base.py
from typing import Protocol

class ReasoningStrategy(Protocol):
    def chat(self, question: str, memory: ConversationMemory | None = None) -> str:
        ...
```

```python
# Cách dùng từ ngoài (demo script, tests)
router = StrategyRouter(harness=AgentHarness(...))
answer = router.chat("Compare replication vs partitioning")
# router tự chọn Plan-and-Execute vì có từ "compare"
```

---

## Data model mới

Thêm vào `models.py`:

```python
@dataclass
class ReasoningTrace:
    """
    Ghi lại quá trình reasoning của một lần chat.
    Dùng để so sánh strategy trong evaluation và demo.
    """
    strategy_name: str
    question: str
    answer: str
    steps: list[str]          # mô tả từng bước (plan steps, reflection rounds...)
    num_llm_calls: int        # tổng số lần gọi LLM (để đo cost)
    elapsed_seconds: float
```

---

## File cần tạo mới

| File | Nội dung |
|---|---|
| `bookmind_tutor/agents/strategies/__init__.py` | export `ReActStrategy`, `PlanExecuteStrategy`, `ReflexionStrategy`, `StrategyRouter` |
| `bookmind_tutor/agents/strategies/base.py` | `ReasoningStrategy` Protocol + `ReasoningTrace` (hoặc import từ models) |
| `bookmind_tutor/agents/strategies/react.py` | thin wrapper |
| `bookmind_tutor/agents/strategies/plan_execute.py` | Plan-and-Execute |
| `bookmind_tutor/agents/strategies/reflexion.py` | Reflexion |
| `bookmind_tutor/agents/strategies/router.py` | `StrategyRouter` |
| `tests/agents/strategies/test_react.py` | |
| `tests/agents/strategies/test_plan_execute.py` | |
| `tests/agents/strategies/test_reflexion.py` | |
| `tests/agents/strategies/test_router.py` | |
| `scripts/demo_strategies.py` | chạy cả 3 strategy trên cùng câu hỏi, in side-by-side |

---

## File thay đổi

| File | Thay đổi |
|---|---|
| `bookmind_tutor/agents/models.py` | thêm `ReasoningTrace` dataclass |
| `bookmind_tutor/agents/__init__.py` | export thêm `strategies` subpackage |
| `scripts/demo_chat.py` | thêm flag `--strategy react|plan|reflexion|auto` |

---

## Test plan

Tất cả LLM call đều mock — không cần ANTHROPIC_API_KEY để chạy unit test.

- `test_plan_execute.py`: mock LLM trả về plan JSON hợp lệ → verify 3 phase chạy đúng thứ tự; mock JSON lỗi → verify fallback về ReAct.
- `test_reflexion.py`: mock reflection score cao (≥4) → verify không retry; mock score thấp → verify retry xảy ra đúng 1 lần.
- `test_router.py`: verify routing rules với các câu hỏi mẫu.
- `test_react.py`: verify wrapper delegate đúng vào `AgentHarness`.

---

## Evaluation thủ công (sau khi implement)

Chạy `scripts/demo_strategies.py` với 3 câu hỏi đại diện:

| Loại câu hỏi | Câu hỏi mẫu | Strategy kỳ vọng tốt hơn |
|---|---|---|
| Đơn giản | "What is eventual consistency?" | ReAct (nhanh, đủ) |
| So sánh | "Compare leader-based vs leaderless replication" | Plan-and-Execute |
| Cần độ sâu | "Explain in depth how consensus algorithms work" | Reflexion |

Ghi kết quả quan sát vào `lessons/` sau khi chạy xong.

---

## Scope và giới hạn

**Trong scope tuần này:**
- 3 strategy + router như mô tả trên.
- `ReasoningTrace` để observe từng strategy.
- Demo script so sánh side-by-side.

**Ngoài scope (không làm tuần này):**
- LLM-based router (chỉ dùng heuristic từ khóa).
- Online learning từ feedback user.
- Streaming output.
- Tree-of-Thought (phức tạp, để backlog nếu muốn).

**Rủi ro:**
- Reflexion tốn nhiều API call nhất (3 call thay vì 1). Nếu thấy chậm, giảm max_rounds xuống 1 (không retry).
- Plan-and-Execute dễ bị LLM tạo ra plan quá nhiều bước. Cap ở 5 sub-tasks là đủ an toàn.
