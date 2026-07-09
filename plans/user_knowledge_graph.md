# Plan: User Knowledge Graph (Personal Brain)

## Bối cảnh và động lực

Week 4 đã build một Knowledge Graph từ book chunks - về bản chất đây là một *book index* được lưu trong Neo4j.
Qua thảo luận, chúng ta xác định một hướng khác có giá trị hơn:

**KG không phải là index của sách. KG là "brain" của user - mô hình tri thức cá nhân tiến hoá theo thời gian tương tác.**

Điều này có nghĩa:
- Entities trong KG = những khái niệm user đã chủ động khám phá (không phải toàn bộ nội dung sách)
- KG tồn tại xuyên suốt nhiều session, nhiều cuốn sách
- User có quyền kiểm soát: agent gợi ý, user duyệt, KG mới được cập nhật
- GraphRAG dùng user KG để personalize retrieval: tìm nội dung sách liên quan đến những gì user đã biết

## Thiết kế tổng thể

```
User hỏi câu hỏi
       │
       ▼
Vector DB search (luôn chạy, không phụ thuộc KG)
       │
       ├── User KG có entities? → GraphRAG expand query → tìm thêm nội dung liên quan
       │
       ▼
AgentHarness trả lời (dựa trên retrieved chunks)
       │
       ▼
[Post-processing] KGUpdateSuggester phân tích chunks vừa retrieved
       │
       ├── Extract entities từ chunks + câu hỏi
       ├── So sánh với user KG hiện tại
       └── Nếu có entities mới → trả về suggestions
              │
              ▼
       [Streamlit UI] Hiển thị suggestions dưới câu trả lời
       User tick chọn → "Add to my knowledge graph"
              │
              ▼
       GraphStore.upsert(selected entities)
       User KG được cập nhật
```

**Cold start:** KG trống = GraphRAG không expand gì = retrieval giống plain vector search. Hoàn toàn ổn.
Vector DB luôn là backbone, KG chỉ là enhancement layer tích luỹ theo thời gian.

## Thay đổi so với Week 4

| | Week 4 (book KG) | Design mới (user KG) |
|---|---|---|
| Entities đến từ đâu | LLM extract từ tất cả book chunks | Từ chunks được retrieve trong cuộc trò chuyện |
| Khi nào build | Offline, trước khi chat | Lazily, sau mỗi conversation turn |
| Ai quyết định thêm | Script tự động | User approve |
| Scope | Một cuốn sách | Xuyên suốt nhiều sách/session |
| Cold start | Graph đầy từ đầu | Graph trống, phát triển dần |

Book KG (`scripts/build_kg.py`) vẫn giữ lại vì có giá trị học tập (Week 4 objective),
nhưng không còn là cơ chế chính trong production flow.

## Data model

### Neo4j labels

Dùng label riêng để tách user knowledge khỏi book index (nếu cần chạy song song):

```cypher
(:UserEntity {name, type, description, added_at, source_chunk_id})
(:UserRelation) -- dùng relationship type như hiện tại
```

Hoặc đơn giản hơn: giữ `:Entity` label nhưng thêm property `scope: "user"`.
Decision: **dùng label riêng `UserEntity`** để query rõ ràng hơn, tránh nhầm lẫn với book entities.

### Python dataclasses mới

```python
@dataclass
class KGUpdateSuggestion:
    entity: Entity          # entity được đề xuất thêm
    reason: str             # tại sao đề xuất (vd: "appeared in retrieved content about Raft")
    source_chunk_id: str    # traceability

@dataclass  
class KGUpdateResult:
    suggestions: list[KGUpdateSuggestion]
    already_known: list[str]   # entity names đã có trong user KG (để debug/display)
```

## Components cần implement

### 1. `KGUpdateSuggester` - `bookmind_tutor/knowledge_graph/suggester.py`

Chạy sau mỗi conversation turn.

```python
class KGUpdateSuggester:
    def suggest(
        self,
        question: str,
        retrieved_chunks: list[Chunk],
        user_graph: GraphStore,
    ) -> KGUpdateResult:
        ...
```

Logic:
1. Gọi `EntityExtractor.extract()` trên từng retrieved chunk (reuse existing)
2. Lấy tất cả entity names từ user KG (`graph_store.get_all_entity_names()`)
3. Lọc ra entities chưa có trong user KG
4. Trả về `KGUpdateResult` với danh sách suggestions

Lưu ý: không cần LLM call riêng cho bước này vì `EntityExtractor` đã là LLM call.
Để tiết kiệm cost, có thể cache extraction result per chunk_id.

### 2. `GraphStore` - thêm support cho `UserEntity`

Thêm methods vào `graph_store.py`:
- `upsert_user_entity(entity: Entity)` - MERGE vào `UserEntity` label
- `get_user_entity_names() -> list[str]` - lấy tất cả names trong user KG
- `get_user_related_entities(name, depth=1) -> list[Entity]` - traverse user KG

### 3. `GraphRAGRetriever` - dùng user KG

Update `graph_rag.py` để nhận `user_graph: GraphStore | None`:
- Nếu có user graph: match entity names trong query với user KG → expand
- Fallback về plain vector search nếu user KG trống hoặc không có match

### 4. Streamlit integration - `bookmind_tutor/tutor/app.py`

Sau khi agent trả lời, nếu có suggestions:

```
┌─────────────────────────────────────────────┐
│ Assistant: [answer...]                       │
│                                             │
│ 💡 New concepts in this answer:             │
│  ☐ eventual consistency (CONCEPT)           │
│  ☐ replication lag (CONCEPT)                │
│  ☐ raft (SYSTEM)                            │
│                                             │
│  [Add selected to my knowledge graph]       │
└─────────────────────────────────────────────┘
```

Dùng `st.expander` + `st.checkbox` + `st.button`.

## Phân chia theo tuần

### Tuần 5 - Core pipeline + basic UI
- `KGUpdateSuggester` + tests
- `GraphStore` user entity methods + tests
- Streamlit basic: upload → chat → suggestions appear (no approve yet, just display)

### Tuần 6 - Approve loop + GraphRAG personalization
- Streamlit approve loop: user tick → entities added to KG
- Update `GraphRAGRetriever` để dùng user KG
- KG visualization nhẹ trong sidebar (danh sách entities đã biết)

## Rủi ro và giới hạn

- **LLM cost per turn**: `EntityExtractor` tốn 1 LLM call/chunk. Nếu mỗi turn retrieve 5 chunks → 5 calls thêm mỗi turn. Cần cache theo `chunk_id` để không re-extract chunk đã xử lý trước đó.
- **Noise trong suggestions**: LLM đôi khi extract entities không liên quan. User approve là safety net, nhưng UX sẽ kém nếu có quá nhiều gợi ý mỗi turn. Cần filter: chỉ suggest entities xuất hiện trong ≥2 chunks hoặc được nhắc đến trong câu hỏi.
- **Entity name normalization**: "Raft", "raft", "the Raft algorithm" cần được treat là cùng một entity. Hiện tại `Entity.__post_init__` đã lowercase, nhưng cần kiểm tra kỹ khi merge.
- **Scope của "user brain"**: Hiện tại entities vẫn đến từ book chunks. Nếu muốn user thêm kiến thức từ nguồn ngoài sách (vd: "tôi đã biết về Paxos từ paper"), cần một cơ chế khác - để TODO cho sau.
