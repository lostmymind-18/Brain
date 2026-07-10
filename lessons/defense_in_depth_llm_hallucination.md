# Bài học: Defense-in-depth cho hallucination trong RAG

**Ngữ cảnh:** Debugging session phát hiện model hallucinate tên chương tự tin,
dẫn đến việc thiết kế hệ thống chống hallucination 4 tầng cho BookMind Tutor.

---

## Tại sao một lớp bảo vệ là không đủ

Hallucination trong RAG có nhiều nguyên nhân độc lập:

- Retrieval thất bại (chunk liên quan không vào top-k) → model thiếu ngữ liệu
- Model ignore retrieved chunks, dùng training memory thay thế
- Model nhận đủ chunks nhưng vẫn suy diễn quá đà ("fabricate plausible details")
- Không ai kiểm tra đầu ra - user không biết claim nào có nguồn, claim nào không

Không có lớp đơn nào giải quyết cả ba. Cần defense in depth.

---

## Kiến trúc 4 tầng

```
[USER QUESTION]
      │
      ▼
[TẦNG 1: Retrieval] ── Hybrid search, tăng khả năng tìm đúng chunk
      │
      ▼
[TẦNG 2: Generation] ── System prompt cấm dùng trí nhớ thay thế retrieved text
      │
      ▼
[TẦNG 3: Verification] ── Post-answer citation check, highlight claim chưa có nguồn
      │
      ▼
[TẦNG 4: Measurement] ── hallucination_score trong eval pipeline
```

Mỗi tầng bảo vệ một failure mode khác nhau. Tầng trước thất bại, tầng sau vẫn catch được.

---

## Tầng 1 - Retrieval: BM25 + Dense Hybrid

**Vấn đề:** `all-MiniLM-L6-v2` kém với proper nouns và exact terms.
"Chapter 14" hay "CQRS" có thể không được match tốt bằng cosine similarity.

**Fix:** Kết hợp BM25 (lexical exact match) + dense vector (semantic).
BM25 cực tốt ở chỗ dense kém (proper noun, technical term, chapter name).
Dense tốt ở chỗ BM25 kém (paraphrase, synonym, concept overlap).

```
hybrid_score = alpha * dense_score + (1 - alpha) * bm25_score_normalized
alpha = 0.5  # equal weight là reasonable default
```

Cách normalize BM25: chia cho max score của batch để đưa về [0, 1].

---

## Tầng 2 - Generation: Hard grounding instruction

**Vấn đề:** Default model behavior là "answer helpfully" - khi thiếu context,
nó fill gap bằng training memory mà không thông báo.

**Fix:** Thêm vào system prompt:
```
GROUNDING RULES: Every factual claim about the book must appear in the
retrieved excerpts or book_outline output. Never use training knowledge
to fill gaps. If search results don't contain enough information, say
"I could not find this in the book" rather than guessing.
```

Điểm quan trọng: instruction phải là **negative constraint** ("never fill gaps")
chứ không phải positive request ("cite your sources") - vì model vẫn có thể
cite sources lẫn thêm hallucinated details nếu chỉ yêu cầu cite.

---

## Tầng 3 - Verification: Post-answer claim check

**Vấn đề:** Ngay cả với grounding instruction, model vẫn có thể đưa ra
claims có vẻ hợp lý nhưng không có trong retrieved chunks.

**Fix:** Sau khi answer được tạo ra, chạy một LLM call riêng:
extract tối đa 5 factual claims từ answer, kiểm tra từng claim có xuất hiện
trong retrieved chunks không. Hiển thị kết quả cho user.

Nguyên tắc thiết kế:
- **Display-only, không chặn, không sửa answer.** Verification báo user biết để
  tự đánh giá, không tự ý alter nội dung model đã tạo.
- **Một LLM call cho cả extraction lẫn verification** để giảm latency.
- **Skip answer ngắn** (< 80 ký tự) - câu trả lời ngắn thường là greeting/clarification,
  không có factual claim cần verify.
- **Expand tự động khi có unverified claim** - giúp user không bỏ qua cảnh báo.

---

## Tầng 4 - Measurement: hallucination_score trong eval

**Vấn đề:** Không đo được = không cải thiện được.
Trước khi có metric, không biết các tầng bảo vệ kia có tác dụng không.

**Fix:** Thêm `hallucination_score` (0.0 = fully grounded, 1.0 = hallucinated)
vào LLM-as-judge eval prompt. Không cần LLM call thêm - reuse call đã có,
chỉ thêm một dimension vào JSON output.

Metric này cho phép A/B test: so sánh hallucination_score trước/sau khi có
hybrid search, trước/sau khi có grounding instruction.

---

## Quan sát từ thực tế

Sau khi implement, fact-check expander trên một câu hỏi về layered architecture
cho kết quả: 2 verified, 3 unverified. Điều này cho thấy:
- Model trả lời đúng ở core content (có trong chunks)
- Model suy diễn thêm ở các claims phụ ("each layer only interacts with adjacent layers",
  "changes necessitate redeploying entire application") - những điều nghe có lý
  nhưng không có trong ngữ liệu retrieved

Đây là pattern phổ biến nhất của LLM hallucination: **không bịa đặt thô thiển,
mà suy diễn plausible details ngoài ngữ liệu được cung cấp**.

---

## Nguyên tắc tổng quát

**Hallucination defense là multi-layer problem.** Với bất kỳ RAG application nào:

1. Retrieval quality giới hạn ceiling - nếu chunk đúng không vào top-k, các layer sau
   không thể bù đắp.
2. System prompt grounding là necessary but not sufficient - model có thể drift khi
   context dài hoặc câu hỏi phức tạp.
3. Post-answer verification là net trước khi user nhận output - đặc biệt quan trọng
   với domain quan trọng (y tế, pháp lý, tài chính).
4. Metric là tiền đề cải thiện - không đo không biết mình đang ở đâu.

Thứ tự implement quan trọng: **Retrieval trước, Generation sau, Verification cuối**.
Vì fix retrieval thường có tác động lớn nhất với effort ít nhất.
