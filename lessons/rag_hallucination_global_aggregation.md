# Bài học: RAG hallucinate khi bị hỏi "liệt kê toàn bộ X"

**Ngữ cảnh:** Debugging câu hỏi "có những chương nào trong mỗi phần?" trên BookMind Tutor.

---

## Triệu chứng

Model trả lời tên chương bằng tiếng Việt, sai tên, sai số chương - nhưng trả lời rất
tự tin, không có dấu hiệu nào cho thấy nó đang đoán mò.
Cụ thể: "Chương 13" và "Chương 14" có cùng tên "Kiến Trúc Dựa Trên Sự Kiện".

---

## Root cause chain (3 tầng)

### Tầng 1: Dữ liệu hoàn toàn có sẵn

PDF có embedded outline 287 entries (`doc.get_toc()`), chính xác 100%.
Mục lục cũng đã được ingest vào ChromaDB (5 chunks, label "Table of Contents").
Vấn đề không phải thiếu dữ liệu.

### Tầng 2: Embedding của mục lục không đại diện cho khái niệm "mục lục"

Chunk mục lục chứa nội dung như:
```
Chapter 10. Layered Architecture Style . . . . . . . . 153
Chapter 11. Pipeline Architecture Style . . . . . . .  163
```

Embedding của đoạn này đại diện cho **chủ đề kiến trúc phần mềm** - không phải
khái niệm "table of contents" hay "danh sách chương".
Khi query `"table of contents"`, ChromaDB trả về các chunks về Space-Based
Architecture và ORM frameworks, vì chúng cũng nói về kiến trúc.
Chunk mục lục thực sự xếp hạng **#37 trong 317** - nằm ngoài top-5.

Dấu chấm lửng `. . . . .` và số trang càng làm loãng embedding thêm.

### Tầng 3: Không có relevance threshold - model không biết mình nhận rác

`BookSearchTool` không kiểm tra chất lượng kết quả.
Dù top-5 có score cực thấp (cosine similarity ~0.22), tool vẫn trả về đầy đủ 5 kết quả.
Model nhận 5 đoạn không liên quan, nhưng câu hỏi trông "trả lời được" (nó biết cuốn
sách nổi tiếng này từ training data) - nên nó tự tổng hợp từ trí nhớ, dịch sang
tiếng Việt, và trả lời tự tin.

---

## Insight cốt lõi

**RAG chunk-level không phù hợp cho câu hỏi tổng hợp toàn cục ("liệt kê tất cả X").**

RAG mạnh ở: "tìm thông tin cụ thể về X trong sách".
RAG yếu ở: "cho tôi xem toàn bộ danh sách/cấu trúc của sách".

Lý do: câu hỏi loại 2 yêu cầu **global view** - tổng hợp từ nhiều chunk rải rác,
không có chunk đơn lẻ nào chứa đủ thông tin. Retrieval by similarity chỉ tốt khi
câu trả lời nằm gọn trong 1-2 chunks gần nhau.

---

## Fix được áp dụng

### Fix chính: `BookOutlineTool` - bypass embedding hoàn toàn

Thay vì tìm mục lục qua vector search, derive trực tiếp từ ChromaDB metadata:
- Field `chapter` = "Part I. Foundations", "Part II. Architecture Styles"...
- Field `section` = "Chapter 9. Foundations", "Chapter 10. Layered..."...
- Sort theo `page_start` để giữ thứ tự sách

Kết quả: 100% chính xác, deterministic, không tốn embedding call.
Model học cách dùng tool này cho câu hỏi cấu trúc nhờ tool description rõ ràng.

### Fix phụ: Relevance threshold

Thêm warning khi `best_score < 0.25`:
```
[Warning: low relevance — best match score 0.22.
The book may not contain specific information about this query.
Do not invent details not present in the excerpts below.]
```

Làm model degrade trung thực thay vì hallucinate tự tin.
Threshold 0.25 chọn dựa trên quan sát thực tế:
- Câu hỏi không liên quan: score ~0.22
- Câu hỏi bình thường về nội dung sách: score 0.35+

---

## Nguyên tắc tổng quát

> Với bất kỳ thông tin có cấu trúc, deterministic nào (mục lục, danh sách nhân
> vật, timeline sự kiện...) - **đừng để RAG tự tìm**. Persist và expose qua tool
> riêng. Vector search dành cho semantic lookup, không phải structural lookup.

Corollary: khi thiết kế tool, hỏi "thông tin này có thể derive deterministically
không?" trước khi nghĩ đến embedding.
