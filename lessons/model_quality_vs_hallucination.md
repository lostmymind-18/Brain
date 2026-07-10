# Bài học: Chất lượng model và hallucination không phải quan hệ tuyến tính

**Ngữ cảnh:** Câu hỏi phát sinh sau khi implement 4-layer hallucination defense
cho BookMind Tutor - liệu dùng model tốt hơn có giải quyết được vấn đề không?

---

## Hai loại hallucination có đặc tính khác nhau

### Loại 1: Parametric hallucination

Model dùng kiến thức từ training thay vì retrieved context.
Ví dụ: hỏi "chương nào nói về CQRS?" → model nhớ trong training data CQRS thường
nằm ở chương nào đó trong một cuốn sách nổi tiếng → trả lời sai về cuốn sách hiện tại.

**Model tốt hơn giảm loại này đáng kể** vì chúng follow grounding instruction tốt hơn.
Layer 2 (system prompt: "never fill gaps with training knowledge") hiệu quả hơn hẳn
với Opus/GPT-4 so với Haiku/GPT-3.5.

### Loại 2: Plausible extrapolation

Model nhận đủ context, nhưng suy diễn thêm những điều nghe có lý mà không có
trong retrieved chunks.

Ví dụ thực tế từ BookMind Tutor (fact-check bắt được):
- ✓ "Layered architecture is also known as n-tier architecture" — có trong chunks
- ⚠ "Each layer only interacts with adjacent layers" — không có trong chunks,
  nhưng là điều hoàn toàn hợp lý về layered architecture

**Model càng giỏi thì extrapolation càng convincing và fluent hơn**, không nhất
thiết ít hơn. GPT-4 nói sai một cách thuyết phục hơn GPT-3.5.
Đây là loại tinh vi và nguy hiểm hơn vì khó detect bằng mắt thường.

---

## Retrieval failure là bottleneck độc lập với model quality

Khi chunk đúng không vào top-k, model giỏi nhất cũng không có ngữ liệu để trả lời.
Lúc đó nó chỉ có hai lựa chọn: từ chối hoặc hallucinate.
Model tốt hơn từ chối nhiều hơn, nhưng không giải quyết được root cause.

**Hệ quả:** Đầu tư vào retrieval quality (hybrid search, chunking strategy, metadata)
thường có ROI cao hơn là upgrade model, vì nó fix root cause thay vì chỉ giảm symptom.

---

## Mỗi layer defense phụ thuộc model theo cách khác nhau

| Layer | Phụ thuộc model chính? | Ghi chú |
|---|---|---|
| Hybrid search (BM25 + dense) | Không | Hoàn toàn independent |
| Grounding system prompt | Có | Model tốt follow instruction tốt hơn |
| Citation verifier | Có (model của verifier) | Verifier dùng LLM riêng để check claims |
| hallucination_score metric | Có (model của judge) | LLM-as-judge: judge tốt → score chính xác |

Lưu ý: verifier và judge không cần phải là model đắt tiền.
Haiku đủ tốt cho task extract-and-check claims vì task có cấu trúc rõ ràng
và output là JSON, không cần reasoning phức tạp.

---

## Nguyên tắc tổng quát

**"Model upgrade" là giải pháp cho parametric hallucination, không phải cho mọi
loại hallucination.**

Khi debug một RAG system bị hallucinate, hỏi trước:
1. Retrieval thất bại không? (chunk đúng có vào top-k không?) → nếu có, fix retrieval trước
2. Model ignore retrieved context không? → model upgrade hoặc stronger grounding prompt
3. Model extrapolate beyond context? → cần post-answer verification layer

Thứ tự này phản ánh tần suất gặp trong thực tế: retrieval failure là nguyên nhân
phổ biến nhất, extrapolation là tinh vi nhất và khó fix nhất bằng model upgrade.

---

## Corollary: Tại sao defense in depth cần thiết bất kể model nào

Ngay cả khi có model tốt nhất hiện tại (Opus 4 / GPT-4o), hệ thống vẫn cần:
- Retrieval tốt (model không thể bù đắp thiếu context)
- Post-answer verification (loại 2 - plausible extrapolation - vẫn xảy ra)
- Hallucination metric (để biết mình đang ở đâu sau mỗi upgrade)

Defense in depth không phải workaround cho model yếu — đó là kiến trúc đúng
cho production RAG system ở mọi tier model.
