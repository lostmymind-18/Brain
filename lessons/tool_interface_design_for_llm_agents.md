# Bài học: Tool interface là prompt - thiết kế tool cho LLM agent

**Ngữ cảnh:** gpt-4o-mini liên tục thất bại khi được hỏi "Chương 1 nói về chủ đề gì?".
Sau nhiều vòng vá lỗi bề mặt (chống duplicate call, guard param rỗng), root cause thật sự nằm ở chính thiết kế interface của tool `get_book_section`.

---

## Chuỗi failure thực tế: mỗi bước đều "hợp lý", kết quả sai có hệ thống

1. Interface cũ đòi model chọn `field` (metadata field) và `contains` (substring):
   `field='chapter'` nghĩa là part level, `field='section'` nghĩa là chapter level.
   Naming ngược đời này là di sản của structure detector, model không có cách nào biết.
2. Sách thật có anomaly: Chapter 1 đứng **trước** Part I, nên 17 chunks của nó nằm ở part level (field `chapter`).
   Cấu trúc này model càng không thể nhìn thấy.
3. Model đoán điều tự nhiên nhất: `field='section', contains='Chapter 1'`.
   Substring match trúng Chapter 10-19 (73k chars nội dung **sai hoàn toàn**), còn Chapter 1 thật thì không bao giờ match.
4. Khi model thử `contains='Introduction'`, match duy nhất là heading `'Chapter\xa01: Introduction'` của trang câu hỏi trắc nghiệm trong appendix (trang 393).
   Câu trả lời cuối cùng được "tóm tắt" từ trang quiz.
5. **Không có bước nào báo lỗi.** Mọi tool call đều trả về status OK.
   Đây là dạng failure nguy hiểm nhất: sai âm thầm, trông như thành công.

---

## Nguyên tắc 1: Tool schema là một phần của prompt

Mỗi parameter trong input schema là một **quyết định model phải tự đưa ra**.
Nếu quyết định đó đòi hỏi tri thức model không có (schema metadata nội bộ, cấu trúc dữ liệu cụ thể của từng cuốn sách), model sẽ đoán.
Đoán ở quy mô lớn nghĩa là sai có hệ thống, không phải sai ngẫu nhiên.

**Cách kiểm tra khi thiết kế tool:** với từng parameter, hỏi "model lấy giá trị này từ đâu?".
Nếu câu trả lời là "từ câu hỏi của user" hoặc "từ output của tool trước" thì ổn.
Nếu câu trả lời là "phải hiểu cách mình lưu dữ liệu" thì parameter đó phải biến mất, logic chuyển vào trong tool.

Fix cụ thể: interface mới chỉ còn `section_name` (thứ user nhắc đến trong câu hỏi).
Tool tự match trên **cả hai** metadata field, tự xử lý chuyện Chapter 1 nằm nhầm tầng.

---

## Nguyên tắc 2: Fail với guidance, đừng "fail" với data

Khi input mơ hồ (`'Chapter 1'` khớp 13 headings khác nhau), có hai lựa chọn:

| Lựa chọn | Chi phí | Hệ quả |
|---|---|---|
| Trả về mọi thứ khớp substring | 73k chars, ~20k tokens | Model tự tin tóm tắt nội dung sai, không ai phát hiện |
| Trả về danh sách headings khớp + số chunks + page range | ~400 tokens | Model chọn đúng tên trong call kế tiếp |

Nội dung sai là **invisible failure**: model không có cách phân biệt "context đúng" với "context trông có vẻ đúng".
Một error message có thể hành động được (danh sách giá trị hợp lệ, ví dụ call đúng) biến failure thành one-step recovery.

Kết quả đo được trên cùng một câu hỏi: từ 7 LLM calls / 63k input tokens xuống 2 calls / 23k tokens, và lần đầu tiên câu trả lời dựa trên đúng nội dung Chapter 1.

---

## Nguyên tắc 3: Normalize theo cách LLM viết, không phải theo cách data lưu

Model đoán `'Chapter 1: Introduction'` (dấu hai chấm).
Heading trong sách là `'Chapter 1. Introduction'` (dấu chấm).
Heading trong TOC/appendix là `'Chapter\xa01: Introduction'` (non-breaking space).

Ba biến thể của cùng một chương, và exact match theo string thô sẽ trúng **đúng cái tệ nhất** (trang quiz).
Matching phải bỏ punctuation và collapse whitespace để mọi biến thể gộp về một nhóm.

Tổng quát: khi LLM sinh ra giá trị để match với dữ liệu, luôn có một tầng normalize ở giữa.
LLM viết theo thói quen ngôn ngữ tự nhiên; data lưu theo tai nạn của pipeline extraction.

---

## Nguyên tắc 4: Mọi lỗi phải là string mà model đọc và học được

Phiên bản đầu, khi model gọi thiếu param, Python raise `TypeError: unexpected keyword argument 'section'`.
Executor catch và trả message đó cho model.
Model học được gì từ message này? Gần như không gì, nên nó thử lại một biến thể sai khác.

Fix: mọi param có default value, và validate ở đầu `execute()` trả về guidance kèm ví dụ call đúng:
`'The section_name parameter must not be empty. Example: section_name="Chapter 1. Introduction". Call book_outline to see the available names.'`

Tool cho LLM agent không được phép raise exception vì input sai format - input sai format là chuyện **chắc chắn xảy ra**, phải nằm trong happy path của thiết kế.

---

## Nguyên tắc 5 (debugging): Đừng tin trace đã bị rút gọn

Trace UI hiển thị `get_book_section(field='section')` làm mình tưởng model gửi `contains` rỗng, và mất một vòng fix nhầm hướng (thêm guard cho contains rỗng).
Structured logs mới cho sự thật: `args='{"field": "section", "contains": "Chapter 1"}'` - contains không rỗng, nó chỉ bị display summary cắt mất.

Quy trình debug đúng đắn đã tìm ra root cause:
1. Đọc **full args** từ structured logs (`grep tool_call`), không đọc từ UI.
2. Inspect metadata thật trong vector store (distinct values + counts), không suy đoán từ code.
3. Tái hiện exact failing call offline (script gọi thẳng tool với args từ log) trước khi sửa.

---

## Checklist khi thiết kế tool mới cho agent

- [ ] Từng parameter: model lấy giá trị từ đâu? Nếu từ "hiểu biết về internals" thì bỏ param, chuyển logic vào tool.
- [ ] Input mơ hồ trả về lựa chọn để model chọn, không trả về đống data "có vẻ khớp".
- [ ] Input không khớp gì trả về danh sách giá trị hợp lệ.
- [ ] Matching có tầng normalize (case, punctuation, whitespace, unicode variants).
- [ ] Không raise exception vì input sai; trả guidance string kèm ví dụ call đúng.
- [ ] Truncation note nói rõ "gọi lại cùng params sẽ ra kết quả y hệt" và chỉ cách lấy phần còn lại.
- [ ] Log full args của mọi tool call; trace hiển thị cho người xem có thể rút gọn nhưng log thì không.

Liên quan: [[trustworthy_rag_blueprint]], [[rag_hallucination_global_aggregation]], [[model_quality_vs_hallucination]]
