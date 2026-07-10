# Bài học: Blueprint cho một hệ thống RAG đáng tin cậy

**Ngữ cảnh:** Tổng kết sau 9 tuần xây dựng BookMind Tutor.
Đây là bản đồ tổng thể, đúc kết kiến trúc thành dạng có thể tái áp dụng cho dự án RAG bất kỳ.
Các lesson chi tiết được link ở từng phần.

---

## Bức tranh tổng thể

```
[INGESTION]        PDF -> chunks với metadata giàu (chapter, section, page_range, char offsets)
     |
[RETRIEVAL]        2 chế độ bắt buộc:
     |               - semantic top-k (hybrid BM25 + dense)     <- câu hỏi cụ thể
     |               - structured complete (metadata filter)     <- tóm tắt / liệt kê / tổng hợp
     |
[GENERATION]       grounded system prompt + agent tự chọn tool
     |
[VERIFICATION]     post-answer citation check (display-only, không chặn)
     |
[MEASUREMENT]      LLM-as-judge với hallucination metric + structured logs + traces
     |
[UI]               sources trung thực (từ retrieval path thực), verification hiển thị công khai
```

Nguyên tắc xuyên suốt: **mỗi tầng giả định tầng khác sẽ thất bại**.
Không tầng nào là "đủ tốt để bỏ tầng kia" (chi tiết: [[defense_in_depth_llm_hallucination]]).

---

## 1. Metadata giàu từ ingestion là nền móng của mọi thứ

Mỗi chunk giữ: `chapter`, `section`, `page_range`, `char_offset`.
Lúc viết ingestion pipeline, những field này trông như "nice to have".
Thực tế, **mọi tính năng tạo niềm tin đều đứng trên chúng**:

| Tính năng | Cần metadata gì |
|---|---|
| Structured retrieval (tóm tắt chương) | chapter, section |
| Citation trong câu trả lời | page_range |
| Sources panel | chapter + page_range |
| Disambiguation list | heading values + counts |
| Eval truy vết ngược | tất cả |

Tiết kiệm ở tầng ingestion nghĩa là trả giá ở mọi tầng phía trên.
Đây là quyết định có ROI cao nhất toàn dự án.

---

## 2. Hai chế độ retrieval là bắt buộc, không phải tối ưu hoá

Phát hiện quan trọng nhất về mặt kiến trúc: **top-k semantic search về mặt cấu trúc không thể phục vụ câu hỏi tổng hợp**.
"Tóm tắt chương 5" cần toàn bộ ~30 chunks của chương; top-5 similarity chỉ trả về mảnh rời rạc.
"Liệt kê các chương" cần dữ liệu có thẩm quyền (TOC), không phải các chunk "nói về chương" (chi tiết: [[rag_hallucination_global_aggregation]]).

Ba tool tạo thành bộ retrieval hoàn chỉnh:

| Tool | Trả lời cho | Cơ chế |
|---|---|---|
| `book_search` | Câu hỏi cụ thể ("CQRS là gì?") | hybrid BM25 + dense, top-k ([[hybrid_bm25_dense_search]]) |
| `get_book_section` | Tóm tắt/tổng hợp một phạm vi | metadata filter, trả TOÀN BỘ chunks trong phạm vi |
| `book_outline` | Câu hỏi về cấu trúc sách | TOC deterministic, không đụng embeddings |

Router giữa các chế độ chính là LLM: nếu tool descriptions viết rõ "khi nào dùng tool nào", model tự route đúng.
Không cần intent classifier riêng.

Lưu ý ngôn ngữ: query phải cùng ngôn ngữ với corpus (embedding tiếng Việt trên sách tiếng Anh cho kết quả tệ hơn hẳn - [[query_language_mismatch_embedding]]).
Chỉ thị "always search in English" trong tool description là fix một dòng có tác động lớn.

---

## 3. Tool interface là prompt

Toàn bộ bài học nằm ở [[tool_interface_design_for_llm_agents]], tóm tắt một câu:
**mỗi parameter là một quyết định model phải tự đưa ra; nếu quyết định đòi hỏi tri thức về internals, model sẽ đoán và sai có hệ thống**.

Hệ quả thiết kế: input mơ hồ trả về danh sách lựa chọn thay vì data "có vẻ khớp"; lỗi trả về guidance string thay vì exception; matching normalize theo cách LLM viết.

---

## 4. Verification hiển thị, không chặn

Post-answer citation check ([[post_answer_verification_pattern]]) chạy sau khi answer hoàn tất:
một LLM call extract claims và đối chiếu với retrieved chunks, render thành fact-check panel.

Quyết định quan trọng: verification là **display-only**.
Nó cho user tín hiệu để hoài nghi, không âm thầm sửa hay chặn câu trả lời.
Một hệ thống tự sửa answer dựa trên verifier là chồng hai tầng LLM không đáng tin lên nhau.

Bài học từ debugging: khi fact-check báo "not found" hàng loạt, nghi ngờ **retrieval sai chunks trước**, đừng vội đổ cho verifier.
Trong case Chương 1, verifier bị mang tiếng "không handle cross-language" nhưng thực ra nó đang check claims với trang quiz appendix.
Retrieve đúng chunks xong, verifier đối chiếu tiếng Việt với tiếng Anh tốt luôn.

---

## 5. UI trung thực là trust boundary cuối cùng

Bug tinh vi nhất toàn dự án: Sources panel chạy một semantic search **độc lập** trên câu hỏi và hiển thị top-5 đó, không liên quan gì đến chunks agent thực sự đọc.
Panel trông chuyên nghiệp, hoạt động "ổn định", và sai từ ngày đầu tiên.

Nguyên tắc: **mọi thông tin provenance trên UI phải lấy từ chính retrieval path đã tạo ra câu trả lời**.
Implement bằng contract nhỏ: retrieval tools ghi `SourceRef` (chapter, page_range) vào `last_sources`; UI clear trước mỗi turn, đọc sau mỗi turn.

Hệ quả đáng giá: khi agent trả lời mà không retrieve gì, Sources rỗng.
"Không có nguồn" cũng là thông tin trung thực - nó báo user rằng câu trả lời này không được grounding.

---

## 6. Observability là công cụ debug số một, không phải trang trí

Ba tầng đã chứng minh giá trị trong thực chiến:

1. **Structured logs với full tool args** (`structlog`): root cause của bug Chương 1 được tìm ra bằng `grep tool_call`, sau khi trace UI (đã rút gọn args) dẫn debug đi sai hướng.
2. **Trace trong UI** (strategy, tool calls, tokens): giúp user và developer thấy agent "nghĩ" gì, phát hiện duplicate calls, đo chi phí từng câu hỏi.
3. **Eval có hallucination metric** (LLM-as-judge): không đo được thì không biết fix có tác dụng không.

Quy trình debug lặp lại được: đọc full args từ logs, inspect data thật (distinct metadata values), tái hiện exact call offline, rồi mới sửa.

---

## 7. Model tier quan trọng, nhưng kiến trúc quan trọng hơn

Chi tiết ở [[model_quality_vs_hallucination]], các mốc thực nghiệm:
model ≤7B skip tool hoàn toàn (nguy hiểm nhất vì không có signal); ~14B+ mới tool-use ổn định; gpt-4o-mini đủ tốt khi tool interface được thiết kế đúng.

Case Chương 1 là minh chứng: cùng một model gpt-4o-mini, chỉ sửa kiến trúc tool, kết quả từ "tóm tắt từ trang quiz, 7 calls, 63k tokens" thành "đúng nội dung chương, 2 calls, 23k tokens".
Nâng model không sửa được interface tồi.

---

## Thứ tự đầu tư khi bắt đầu dự án RAG mới

1. **Ingestion với metadata giàu** - nền móng, sửa sau rất đắt.
2. **Hai chế độ retrieval** (semantic top-k + structured complete) - quyết định trần chất lượng.
3. **Tool interface đúng nguyên tắc** - quyết định model tier nào dùng được.
4. **Grounding system prompt** - rẻ, làm ngay từ đầu.
5. **Structured logging full tool args** - trả giá bằng vài dòng code, cứu hàng giờ debug.
6. **Post-answer verification + Sources trung thực** - tầng tạo niềm tin với user.
7. **Eval với hallucination metric** - làm trước khi tối ưu bất kỳ thứ gì ở trên.

Ba mục đầu là kiến trúc (khó sửa sau), bốn mục sau là tính năng (thêm dần được).

Liên quan: [[tool_interface_design_for_llm_agents]], [[defense_in_depth_llm_hallucination]], [[hybrid_bm25_dense_search]], [[rag_hallucination_global_aggregation]], [[post_answer_verification_pattern]], [[model_quality_vs_hallucination]], [[query_language_mismatch_embedding]]
