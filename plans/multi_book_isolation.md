# Plan: Per-Book Vector Store Isolation

## Vấn đề

Hiện tại toàn bộ session dùng một `VectorStore` duy nhất (một ChromaDB collection).
Khi user upload nhiều sách, chunks của tất cả sách bị trộn chung vào một collection.
`search()` trả về kết quả từ bất kỳ cuốn nào — không có sự phân tách.

User KG (`UserEntity` trong Neo4j) đúng là nên dùng chung giữa các sách — đây là
"personal brain" của user, không gắn với cuốn sách cụ thể nào. Chỉ có vector store
cần được tách riêng theo sách.

## Thiết kế tổng thể

```
session_state
├── book_stores: dict[book_id → VectorStore]   # mỗi sách 1 ChromaDB collection
├── book_names:  dict[book_id → str]            # book_id → tên hiển thị
├── book_histories: dict[book_id → list[msg]]   # lịch sử chat per-book
├── current_book_id: str | None                 # sách đang active
├── graph_store: GraphStore                     # dùng chung (user KG)
└── llm_client: Anthropic                       # dùng chung
```

Các key cũ bị bỏ: `indexed` (bool) → thay bằng `current_book_id is not None`,
`vector_store` (single) → thay bằng `book_stores[current_book_id]`,
`messages` (flat list) → thay bằng `book_histories[current_book_id]`.

## Book ID

```python
import re

def _book_id(filename: str) -> str:
    """Derive a stable, readable ID from a PDF filename."""
    name = filename.removesuffix(".pdf").lower()
    return re.sub(r"[^a-z0-9]+", "_", name)[:40]
```

Dùng tên file normalize thay vì hash — dễ debug, dễ hiểu.
Collision: nếu hai file khác nhau cho cùng ID thì VectorStore sẽ có cùng collection
name → chunks của hai sách bị trộn. Khả năng xảy ra rất thấp trong thực tế (demo/POC).
Ghi note trong code, không cần handle ở giai đoạn này.

## VectorStore per book

`VectorStore.__init__` đã nhận `collection_name`. Không cần thay đổi gì ở `VectorStore`.

Khi index sách mới:
```python
book_id = _book_id(uploaded.name)
vs = VectorStore(collection_name=f"book_{book_id}")   # ephemeral, isolated
st.session_state.book_stores[book_id] = vs
```

Khi search (chat, suggestions):
```python
vs = st.session_state.book_stores[st.session_state.current_book_id]
```

**Note về persistence:** VectorStore hiện tại là ephemeral (in-memory, mất khi restart).
Tức là mỗi session phải re-index lại. Đây là hạn chế đã biết, không thay đổi với plan này.
Persistent storage (ChromaDB on disk) là stretch goal — ghi vào backlog.

## Backend changes

### `app.py` — `_init_state()`

```python
defaults = {
    "book_stores": {},           # dict[book_id, VectorStore]
    "book_names": {},            # dict[book_id, str]  (display name)
    "book_histories": {},        # dict[book_id, list[msg]]
    "current_book_id": None,
    "approved_msg_ids": set(),
    "next_msg_id": 0,
}
```

Bỏ: `"indexed"`, `"messages"`, `"book_name"`, `"vector_store"`.

### `app.py` — `_upload_page()`

Tách làm 2 phần:
1. File uploader như hiện tại.
2. Sau khi index xong: tạo `VectorStore` với `collection_name`, lưu vào `book_stores`,
   set `current_book_id`, tạo entry rỗng trong `book_histories`.

Upload sách đã indexed (cùng book_id): upsert lại vào collection đó — idempotent như cũ.

### `app.py` — `_setup_agent()`

Nhận VectorStore từ `book_stores[current_book_id]` thay vì `session_state.vector_store`.
Không cần thêm tham số — function đọc trực tiếp từ session_state.

### `app.py` — `_handle_question()`

```python
vs = st.session_state.book_stores[st.session_state.current_book_id]
search_results = vs.search(question, k=5)   # dùng đúng book store
```

### `app.py` — property helpers

```python
def _current_vs() -> VectorStore:
    return st.session_state.book_stores[st.session_state.current_book_id]

def _current_messages() -> list:
    return st.session_state.book_histories[st.session_state.current_book_id]
```

## Frontend changes

### Routing logic (thay thế `indexed` bool)

```python
# Cũ
if not st.session_state.indexed:
    _upload_page()

# Mới
if st.session_state.current_book_id is None:
    _upload_page()
else:
    _chat_page()
```

### Upload page — hiển thị sách đã indexed

Nếu `book_stores` không rỗng (có ít nhất 1 sách đã index trong session), hiển thị
danh sách sách đã index với nút "Chat" bên cạnh mỗi sách. Điều này cho phép:
- Upload sách thứ 2 trong khi đã có sách thứ 1.
- Chuyển về sách đã index trước đó mà không cần re-upload.

```
┌─────────────────────────────────────────────────────┐
│ 📚 BookMind Tutor                                   │
│                                                     │
│ Already indexed:                                    │
│  • Fundamental of software architecture  [Chat →]  │
│                                                     │
│ ─────────── Upload another book ──────────          │
│ [file uploader]                                     │
│ [Index Book]                                        │
└─────────────────────────────────────────────────────┘
```

Khi click "Chat →": set `current_book_id`, gọi `_setup_agent()`, `st.rerun()`.

### Chat page title

Giữ như hiện tại: `st.title(f"📚 {book_names[current_book_id]}")`.

### Sidebar — thêm Book Library section

```
┌─────────────────────────────┐
│ 📖 Library                  │
│ ● Fundamental of SA  [active]│   ← đang active
│   DDIA                [→]   │   ← click để switch
│ [+ Upload another]          │   ← quay về upload page
├─────────────────────────────┤
│ My Knowledge Graph          │
│ 5 concept(s) known:         │
│ ...                         │
└─────────────────────────────┘
```

Khi click một sách khác:
1. Lưu `messages` hiện tại vào `book_histories[current_book_id]` (đã là reference, tự động).
2. Set `current_book_id = new_book_id`.
3. Gọi `_setup_agent()` để build router mới với VectorStore của sách mới.
4. `st.rerun()`.

Khi click "+ Upload another": set `current_book_id = None`, `st.rerun()` → về upload page.

### Conversation history per book

`book_histories[book_id]` là list msg dict như hiện tại.
Khi switch sang sách khác, conversation history của sách cũ được preserve.
Khi quay lại sách cũ, history được restore.

`approved_msg_ids` hiện tại dùng chung — vẫn để chung, không ảnh hưởng gì
vì msg_id là unique per message across all books.

## Phạm vi không thay đổi

- `VectorStore` class: không đổi gì.
- `ChunkIndexer`: không đổi gì.
- `GraphRAGRetriever`: không đổi gì.
- `GraphStore` / user KG: không đổi gì.
- `KGUpdateSuggester`: không đổi gì.
- `StrategyRouter` / agent harness: không đổi gì.
- Tests hiện có: không ảnh hưởng.

## Rủi ro và giới hạn

- **Re-index mỗi session:** VectorStore ephemeral → restart app = mất hết, cần upload lại.
  Noted trong backlog. Không fix ở plan này.
- **book_id collision:** hai filename khác nhau → cùng book_id → chunks bị trộn.
  Khả năng thấp với normalized filename, acceptable cho POC.
- **`_setup_agent()` được gọi mỗi lần switch sách:** tạo lại router, harness, retriever.
  Không tốn thêm LLM call. Overhead nhỏ, chấp nhận được.
- **`approved_msg_ids` dùng chung:** msg_id là unique (incrementing int per session)
  nên không có collision cross-book. Ổn.

## Checklist implement

- [ ] Thêm `_book_id()` helper function
- [ ] Update `_init_state()`: session state mới
- [ ] Update `_upload_page()`: tạo VectorStore per book, hiển thị sách đã index
- [ ] Update `_setup_agent()`: dùng `book_stores[current_book_id]`
- [ ] Update `_handle_question()`: dùng `_current_vs()`
- [ ] Update routing: `current_book_id is None` thay `not indexed`
- [ ] Update `_chat_page()`: đọc messages từ `book_histories[current_book_id]`
- [ ] Update `_render_sidebar()`: thêm Library section, book switching
- [ ] Playwright test: upload 2 sách, verify switch hoạt động, verify không cross-contaminate
