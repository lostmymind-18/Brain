# Bài học: Hybrid BM25 + Dense Vector Search

**Ngữ cảnh:** Tuần 8-9 BookMind Tutor - dense embedding search kém với proper nouns
và technical terms, dẫn đến việc thêm BM25 vào bên cạnh ChromaDB.

---

## Vấn đề của pure dense search

`all-MiniLM-L6-v2` (ChromaDB default) là model được train để capture **semantic meaning**.
Khi search "Chapter 14 Event-Driven Architecture":

- Dense: embed toàn bộ phrase thành một vector. Kết quả trả về chunks nói về
  event-driven concepts nói chung - không nhất thiết đúng Chapter 14.
- BM25: tìm documents chứa chính xác các từ "Chapter", "14", "Event-Driven" -
  trả về đúng chunk của Chapter 14.

Dense giỏi semantic paraphrase. BM25 giỏi exact match.
Proper nouns, chapter numbers, tên kỹ thuật (CQRS, SAGA, DDD) → BM25 thắng.
"Kiến trúc nào phù hợp cho high throughput?" (không có exact term) → Dense thắng.

---

## Cơ chế fusion: Score Normalization + Alpha Blend

BM25 và cosine similarity có range khác nhau hoàn toàn. Cần normalize trước khi blend.

```python
# Dense: ChromaDB trả về distance (0 = identical, 2 = opposite)
dense_score = 1.0 - distance  # → [−1, 1], thực tế [0, 1] với cosine

# BM25: raw score dương, không bounded
bm25_score_normalized = raw_score / max(all_scores)  # → [0, 1]

# Fusion
hybrid = alpha * dense_score + (1 - alpha) * bm25_score_normalized
# alpha = 0.5 → equal weight
```

**Tại sao chia max thay vì min-max normalization?**
Min-max: `(x - min) / (max - min)` làm cho candidate thấp nhất luôn = 0.
Điều này có thể overweight candidate BM25 kém nếu min rất thấp.
Chia max giữ được relative ordering và tỉ lệ thực, chỉ scale về [0, 1].

---

## Kỹ thuật: Lazy Build + Fetch Candidates Rộng Hơn

### Lazy BM25 index

BM25 index phải build từ toàn bộ corpus - không thể query incremental như vector DB.
Với 300+ chunks, build mỗi khi search sẽ chậm.

Pattern đã dùng:
```python
self._bm25 = None  # cache

def _get_bm25(self):
    if self._bm25 is not None:
        return self._bm25  # cache hit
    # build once from collection.get(include=["documents"])
    self._bm25 = BM25Okapi(tokenized_docs)
    return self._bm25
```

Invalidate cache khi corpus thay đổi (`index_chunks()`, `clear()`, `drop()`).

### Fetch k*3 candidates trước khi fuse

Nếu chỉ lấy top-k từ mỗi nguồn rồi fuse, có thể bỏ sót candidates tốt.
Ví dụ: một chunk xếp #8 ở dense nhưng #1 ở BM25 sẽ không vào fused result
nếu chỉ lấy top-5 từ mỗi nguồn.

Fix: lấy top `k*3` từ mỗi nguồn, fuse toàn bộ pool, lấy top-k sau fusion.

---

## ChromaDB API quirk: `get()` không accept "ids" trong include

```python
# Sai - ChromaDB throw error
self._collection.get(include=["documents", "ids"])

# Đúng - ids luôn được trả về by default, không cần khai báo
self._collection.get(include=["documents"])
```

Lý do: `ids` không phải "include field" trong ChromaDB model - chúng luôn
được trả về, không thể opt-out. Khác với `documents`, `metadatas`, `embeddings`,
`distances` là optional và phải khai báo.

---

## Khi nào nên dùng hybrid search

**Nên dùng khi corpus có:**
- Proper nouns, tên riêng (tên tác giả, tên chương, sản phẩm)
- Technical terms, acronyms (CQRS, DDD, REST)
- Numbers (page numbers, version numbers, statistics)
- Nội dung đa ngôn ngữ với single-language embedding model

**Pure dense đủ khi:**
- Corpus là natural language thuần, ít exact-term lookup
- Query thường là conceptual ("phong cách kiến trúc nào scale tốt?")
- Performance critical và BM25 build time là bottleneck

---

## Fallback graceful khi không có rank_bm25

BM25 là optional dependency. Nếu không install:
```python
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    return None  # search() falls back to pure dense
```

Caller (`search()`) check `bm25 is None` và dùng `_search_dense()` thay thế.
Không cần user change code - chỉ cần `pip install rank-bm25` để upgrade tự động.

Pattern này hữu ích cho optional performance improvements: feature on nếu có dependency,
off gracefully nếu không - không crash.
