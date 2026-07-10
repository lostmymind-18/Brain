# Bài học: Query language mismatch làm hỏng embedding retrieval

**Ngữ cảnh:** Debugging session BookMind Tutor - model search bằng tiếng Việt
trên index tiếng Anh.

---

## Vấn đề

Model nhận câu hỏi tiếng Việt từ user, tự động viết search query cũng bằng tiếng Việt:
```
"phần I chương", "phần II chương", "Phần III: Chương"
```

Nhưng toàn bộ content trong ChromaDB là tiếng Anh (sách gốc).
Kết quả retrieval rất kém - model hallucinate để bù đắp.

---

## Tại sao embedding không cross-language tốt

`all-MiniLM-L6-v2` (ChromaDB default) là model **monolingual English**.
Embedding của "phần I chương" và "Part I Foundations" nằm ở **hai vùng khác nhau**
trong không gian vector - dù nghĩa tương đương.

Có các model multilingual (e.g. `paraphrase-multilingual-MiniLM-L12-v2`) cross-lingual
tốt hơn nhiều, nhưng không phải default.

---

## Fix

Thêm vào tool description:
```
IMPORTANT: Always write the query in English, regardless of the language the
user is speaking. The book content is in English and the search index only
matches English text.
```

Và trong input schema field description:
```
"The question or topic to search for, written in English."
```

Model sau đó tự translate trước khi search:
- Trước: `"phần I chương"` → kết quả tệ
- Sau: `"table of contents"`, `"Part I chapters"` → kết quả tốt hơn nhiều

---

## Nguyên tắc tổng quát

**Embedding model và query phải cùng ngôn ngữ - hoặc dùng multilingual model.**

Khi build RAG system phục vụ user đa ngôn ngữ với corpus đơn ngữ, có 3 options:

1. **Hướng dẫn model translate query** (đã làm) - đơn giản, đủ tốt cho LLM mạnh.
2. **Dùng multilingual embedding model** - tốt hơn về coverage, tốn RAM hơn.
3. **Translate corpus sang ngôn ngữ user** - không thực tế với sách lớn.

Option 1 là đủ khi:
- LLM đủ mạnh để follow instruction
- Người dùng chấp nhận một chút latency (LLM cần "hiểu" câu hỏi trước khi search)
- Corpus là sách/tài liệu chuyên ngành tiếng Anh

---

## Lưu ý khi debug retrieval

Khi thấy RAG trả về kết quả không liên quan, check ngay:
1. Query language vs corpus language có khớp không?
2. Log `args` của tool call để xem model đang search gì.

Trong project này: `grep "tool_call" /tmp/streamlit_ollama.log` hiện ngay query.
