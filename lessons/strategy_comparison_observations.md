# Kết quả so sánh ReAct vs Plan-and-Execute vs Reflexion trên Book QA

## Setup

- Sách: Designing Data-Intensive Applications (DDIA), 550 chunks
- Model: claude-haiku-4-5-20251001
- 3 câu hỏi đại diện cho 3 loại: đơn giản, so sánh, cần độ sâu

---

## Kết quả thực đo

| Strategy | Q1 đơn giản | Q2 so sánh | Q3 in-depth |
|---|---|---|---|
| ReAct | 6.5s | 22.7s | 13.9s |
| Plan-and-Execute (parallel) | 20.2s | 24.6s | 29.9s |
| Reflexion | 8.4s | 23.2s | 18.3s |

---

## Insight 1: Plan-and-Execute overkill cho câu đơn giản

Với câu "What is eventual consistency?", ReAct mất 6.5s còn Plan-and-Execute mất 20.2s
cho kết quả tương đương về chất lượng.
Chi phí thêm đến từ bước plan (1 LLM call) và bước synthesize (1 LLM call nữa).

**Rút ra:** Strategy routing quan trọng. Dùng Plan-and-Execute cho mọi câu hỏi là lãng phí.

---

## Insight 2: Plan-and-Execute tỏa sáng ở câu so sánh đa chiều

Q2 (Compare leader-based vs leaderless): Plan tạo ra 5 sub-tasks độc lập,
mỗi cái cover đúng một khía cạnh (mechanisms, trade-offs, real-world examples...).
Kết quả có cấu trúc rõ nhất, không bỏ sót yêu cầu nào.

ReAct cũng trả lời được, nhưng dễ "quên" một phần yêu cầu khi vòng lặp kéo dài.

**Rút ra:** Plan-and-Execute đúng với lý thuyết — proactive planning giúp enumerate
đủ sub-requirements ngay từ đầu thay vì phát hiện dần qua observation.

---

## Insight 3: Parallel execution là optimization tự nhiên cho Plan-and-Execute

Trước khi fix: 5 sub-tasks sequential → 82s.
Sau khi fix: 5 sub-tasks parallel → 24.6s (~3x nhanh hơn).

Sub-tasks của Plan-and-Execute hoàn toàn độc lập nhau theo thiết kế (mỗi cái
là một câu hỏi riêng), nên ThreadPoolExecutor fit hoàn hảo mà không cần thay đổi logic.

**Rút ra:** Khi thiết kế strategy có independent sub-tasks, parallelism nên là
mặc định chứ không phải optimization sau.

---

## Insight 4: Reflexion không retry trong bối cảnh Book QA + Haiku

Cả 3 câu hỏi đều cho score 4-5/5, không lần nào retry.
Lý do: Haiku tự chấm điểm cho chính mình có xu hướng generous,
và với book QA (có sẵn chunks để search), initial answer thường đã đủ tốt.

Reflexion trong paper gốc (Shinn et al. 2023) được test trên coding tasks và
reasoning puzzles — những bài mà initial attempt thường sai hoàn toàn.
Book QA ít có dạng "sai hoàn toàn", nên trigger điều kiện retry khó hơn.

**Rút ra:** Reflexion có giá trị nhất khi task có tỷ lệ fail cao ở lần đầu.
Với retrieval-based QA, giá trị của nó hạn chế hơn so với coding/reasoning.

---

## Insight 5: System prompt cần explicit về những gì KHÔNG làm

Prompt "Do not narrate your reasoning steps" không đủ.
Cần liệt kê cụ thể các phrase cần tránh:
"Now I have enough information", "Let me compile", "Perfect!", "Based on my search"...

Nguyên tắc: LLM hiểu instruction theo nghĩa literal. Nếu muốn tắt một behavior
cụ thể, nêu tên behavior đó thay vì mô tả chung.

---

## Tổng kết: khi nào dùng strategy nào

| Tình huống | Strategy |
|---|---|
| Câu hỏi tra cứu đơn giản | ReAct |
| Câu hỏi so sánh, nhiều phần, có cấu trúc rõ | Plan-and-Execute |
| Task có tỷ lệ initial answer kém cao (coding, reasoning) | Reflexion |
| Không biết câu hỏi thuộc loại nào | StrategyRouter + heuristic keyword |

## Nguồn

- Demo thực chạy: `scripts/demo_strategies.py` trên DDIA
- Implementation: `bookmind_tutor/agents/strategies/`
- Liên quan: [[reactive_vs_proactive_planning]]
