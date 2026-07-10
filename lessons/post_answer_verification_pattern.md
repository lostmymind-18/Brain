# Bài học: Post-answer Claim Verification Pattern

**Ngữ cảnh:** Tuần 8-9 BookMind Tutor - thiết kế `CitationVerifier` để kiểm tra
factual claims trong answer của model sau khi answer đã được tạo ra.

---

## Tại sao verify sau khi answer đã có

Cách tiếp cận naive: kiểm tra retrieved chunks, chỉ trả lời nếu có nguồn.
Vấn đề: LLM không follow constraint này reliably. Ngay cả với grounding instruction
mạnh, model vẫn có thể blend retrieved facts với inferred details.

Post-answer verification khác về mặt kiến trúc: nó không can thiệp vào generation,
mà **audit output sau khi generation xong**. Như quality control cuối dây chuyền
thay vì instruction đầu dây chuyền.

Ưu điểm:
- Không affect generation latency (verify song song hoặc sau)
- Không alter answer (user thấy đúng những gì model nói, không bị filter)
- Bắt được những gì grounding instruction bỏ sót
- Dạy user calibrate trust (claim nào có nguồn, claim nào là inferred)

---

## Thiết kế một LLM call cho cả extraction lẫn verification

Bước đầu có thể tách thành hai LLM calls:
1. Call 1: extract claims từ answer
2. Call 2: verify từng claim với retrieved chunks

Nhưng hai calls tốn double latency và double cost. Prompt có thể gộp cả hai:

```python
prompt = f"""Given this answer and the retrieved book excerpts, extract up to 5
factual claims from the answer and check whether each is supported by the excerpts.

ANSWER:
{answer}

RETRIEVED EXCERPTS:
{chunks_text}

Return JSON array:
[{{"claim": "...", "grounded": true/false, "source": "supporting quote or ''"}}]

Rules:
- Only extract factual claims about the book's content (not meta-commentary)
- grounded=true only if the claim is explicitly stated in excerpts
- source: exact quote from excerpts if grounded, empty string if not
- Max 5 claims"""
```

Một LLM call, kết quả là list với cả claim + verdict + evidence.

---

## Tại sao giới hạn 5 claims

Trade-off giữa coverage và cost/latency:
- Answer dài có thể có 10-15 factual claims
- Verify tất cả → prompt dài, LLM chậm hơn, cost cao hơn
- 5 claims bắt được những claims quan trọng nhất (đầu answer thường là core content)
- User có thể tự check tiếp nếu muốn

---

## Skip conditions: không phải answer nào cũng cần verify

```python
_MIN_ANSWER_LENGTH = 80

def verify(self, answer: str, chunks: list[str]) -> VerificationResult | None:
    if len(answer) < _MIN_ANSWER_LENGTH:
        return None  # greeting, clarification, short response
    if not chunks:
        return None  # không có retrieved content để compare
```

Câu trả lời ngắn như "Bạn có câu hỏi gì về cuốn sách?" hay "Tôi không tìm thấy
thông tin này." không có factual claims cần verify. Overhead của LLM call không
worth it.

---

## Display-only là nguyên tắc thiết kế, không phải compromise

Một cách khác: nếu có unverified claims, tự động rewrite answer để remove chúng.

Tại sao không làm vậy:
1. Model verify có thể sai - marking a true claim as ungrounded khi excerpt dùng
   paraphrase thay vì exact words
2. Altered answer có thể mất coherence nếu claim là load-bearing
3. User mất đi cơ hội học: không thấy model suy diễn gì, không calibrate được

Post-answer verification đúng nhất khi là **transparency tool**, không phải
**automatic filter**.

---

## UX: Auto-expand khi có unverified claims

```python
u = sum(1 for c in checks if not c.grounded)
with st.expander(f"Fact-check: {v} verified, {u} unverified", expanded=(u > 0)):
    ...
```

Nếu mọi claim đều verified → expander collapse by default (good news, không cần chú ý).
Nếu có claim chưa verified → expander tự mở (warning cần chú ý).

Pattern tổng quát: UI thành phần chứa warning/error nên auto-expand để không bị bỏ qua.
Thành phần chứa info bổ sung (không urgent) nên collapse by default để không clutter.

---

## Nguyên tắc tổng quát

Post-answer verification là pattern phù hợp khi:
- Generation không thể bị constrained hoàn toàn (LLM không 100% reliable)
- User trust vào accuracy là critical (sách, tài liệu kỹ thuật, pháp lý)
- Output là long-form text với nhiều independent claims
- Bạn muốn giữ nguyên answer gốc nhưng vẫn cần transparency

Không nên dùng khi:
- Answer là code/structured output → có thể test mechanically, không cần LLM verify
- Answer rất ngắn → overhead không đáng
- Latency budget rất chặt → một LLM call thêm có thể không acceptable
