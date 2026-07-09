# Week 4: Knowledge Graph & Long-term Memory

## Mục tiêu học

- Hiểu entity & relation extraction là gì và tại sao nó khó
- Biết cách model hóa knowledge dưới dạng graph (node/edge schema)
- Hiểu Graph RAG: tại sao graph traversal bổ sung gì mà vector search không có
- Thực hành thao tác với Neo4j qua Python driver

---

## Bối cảnh: tại sao cần KG khi đã có vector search?

Vector search tìm chunk **gần về mặt ngữ nghĩa** với câu hỏi.
Nhưng nó không biết:
- "Raft" và "Paxos" đều là consensus algorithms (relation IS_A)
- "eventual consistency" mâu thuẫn với "strong consistency" (relation CONTRADICTS)
- "write-ahead log" được định nghĩa trong Chapter 3 (relation DEFINED_IN)

Khi user hỏi "so sánh các consensus algorithms", vector search tìm chunk có từ
"consensus" — nhưng KG biết ngay Raft, Paxos, Zab là các instance của concept đó,
và có thể expand query để tìm thêm.

---

## Scope tuần này

**Trong scope:**
- Entity extraction từ chunks (LLM-based, fixed entity types)
- Lưu entities + relations vào Neo4j
- GraphRAGRetriever: kết hợp vector search + graph traversal
- Unit tests với mock Neo4j driver

**Ngoài scope (không làm tuần này):**
- Entity extraction từ conversation history (để Tuần 5-6)
- Dynamic knowledge updating khi user chat
- Graph visualization UI

---

## Fixed Entity Types (risk mitigation)

Roadmap cảnh báo entity extraction là rủi ro cao nhất. Dùng fixed types thay vì
để LLM tự quyết — dễ validate, dễ test, ít hallucination hơn.

| Type | Ví dụ trong DDIA |
|---|---|
| `CONCEPT` | eventual consistency, replication lag, write skew |
| `SYSTEM` | Kafka, Cassandra, Raft, ZooKeeper, Dynamo |
| `PRINCIPLE` | CAP theorem, linearizability, serializability |
| `DEFINITION` | write-ahead log, quorum, epoch number |
| `PERSON` | Leslie Lamport, Martin Kleppmann |

Fixed relation types:

| Type | Ý nghĩa |
|---|---|
| `IS_A` | Raft IS_A consensus algorithm |
| `RELATED_TO` | replication RELATED_TO fault tolerance |
| `CONTRADICTS` | eventual consistency CONTRADICTS linearizability |
| `PART_OF` | leader election PART_OF Raft |
| `DEFINED_IN` | WAL DEFINED_IN Chapter 3 |

---

## Graph Schema trong Neo4j

```
(:Entity {
    name: str,          # canonical name, lowercase normalized
    type: str,          # CONCEPT | SYSTEM | PRINCIPLE | DEFINITION | PERSON
    description: str,   # 1-sentence description từ sách
    chunk_id: str       # chunk nơi entity này lần đầu xuất hiện
})

(:Chunk {
    chunk_id: str,
    chapter: str,
    section: str,
    page_start: int
})

(:Entity)-[:IS_A]->(:Entity)
(:Entity)-[:RELATED_TO {description: str}]->(:Entity)
(:Entity)-[:CONTRADICTS]->(:Entity)
(:Entity)-[:PART_OF]->(:Entity)
(:Entity)-[:DEFINED_IN]->(:Chunk)
(:Entity)-[:MENTIONED_IN]->(:Chunk)
```

Mỗi Entity có unique constraint trên `name` để tránh duplicate khi nhiều chunks
nhắc đến cùng một concept.

---

## Kiến trúc module

```
bookmind_tutor/knowledge_graph/
  __init__.py
  models.py        # Entity, Relation dataclasses
  extractor.py     # EntityExtractor: chunk → [Entity] + [Relation]
  graph_store.py   # GraphStore: CRUD với Neo4j
  graph_rag.py     # GraphRAGRetriever: vector + graph combined search
  builder.py       # KnowledgeGraphBuilder: orchestrate extraction → indexing
```

### `models.py`

```python
@dataclass
class Entity:
    name: str          # lowercase, e.g. "raft"
    type: str          # CONCEPT / SYSTEM / PRINCIPLE / DEFINITION / PERSON
    description: str
    chunk_id: str

@dataclass
class Relation:
    source: str        # entity name
    relation_type: str # IS_A / RELATED_TO / CONTRADICTS / PART_OF / DEFINED_IN
    target: str        # entity name hoặc chunk_id (cho DEFINED_IN)
    description: str   # optional, dùng cho RELATED_TO
```

### `extractor.py` - EntityExtractor

Nhận 1 Chunk, gọi LLM để extract entities và relations.

**Prompt design:**
```
Given this book excerpt, extract technical entities and their relationships.

Entity types to find: CONCEPT, SYSTEM, PRINCIPLE, DEFINITION, PERSON
Relation types: IS_A, RELATED_TO, CONTRADICTS, PART_OF, DEFINED_IN

Excerpt (Chapter X, p.Y):
{chunk_text}

Return JSON:
{
  "entities": [
    {"name": "raft", "type": "SYSTEM", "description": "..."}
  ],
  "relations": [
    {"source": "raft", "type": "IS_A", "target": "consensus algorithm", "description": ""}
  ]
}
```

**Fallback:** nếu JSON parse fail → trả về ([], []) thay vì crash.
**Batching:** extract từng chunk một, không batch nhiều chunks vào 1 prompt
(context quá lớn làm giảm precision).

### `graph_store.py` - GraphStore

Thin wrapper quanh Neo4j driver.

```python
class GraphStore:
    def __init__(self, uri, user, password): ...
    def upsert_entity(self, entity: Entity) -> None: ...
    def upsert_relation(self, relation: Relation) -> None: ...
    def get_related_entities(self, entity_name: str, depth: int = 2) -> list[Entity]: ...
    def find_entities_in_chunk(self, chunk_id: str) -> list[Entity]: ...
    def clear(self) -> None: ...
    def close(self) -> None: ...
```

Upsert dùng `MERGE` thay vì `CREATE` để idempotent (re-run không tạo duplicate).

Unique constraint tạo khi init:
```cypher
CREATE CONSTRAINT entity_name IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS UNIQUE
```

### `graph_rag.py` - GraphRAGRetriever

Kết hợp 2 nguồn:

```
1. Vector search (existing VectorStore.search)
   → top-k chunks by semantic similarity

2. Graph expansion:
   a. Extract entity names từ query (dùng LLM hoặc keyword match)
   b. Tìm related entities trong KG (depth=1-2 hops)
   c. Lấy chunk_ids từ MENTIONED_IN relations của các related entities
   d. Fetch những chunks đó từ VectorStore

3. Merge và deduplicate kết quả
   → rank: vector score + bonus nếu chunk có nhiều related entities
```

Lý do không bỏ vector search: graph traversal tốt cho **known entities**,
vector search tốt cho **paraphrase / mơ hồ**. Hai cái bổ sung nhau.

### `builder.py` - KnowledgeGraphBuilder

Orchestrate toàn bộ pipeline:

```python
class KnowledgeGraphBuilder:
    def build_from_chunks(self, chunks: list[Chunk]) -> BuildStats: ...
    # PdfExtractor → chunks → EntityExtractor → GraphStore
    # Trả về stats: total_entities, total_relations, failed_chunks
```

---

## File cần tạo

| File | Nội dung |
|---|---|
| `bookmind_tutor/knowledge_graph/__init__.py` | exports |
| `bookmind_tutor/knowledge_graph/models.py` | Entity, Relation, BuildStats |
| `bookmind_tutor/knowledge_graph/extractor.py` | EntityExtractor |
| `bookmind_tutor/knowledge_graph/graph_store.py` | GraphStore (Neo4j) |
| `bookmind_tutor/knowledge_graph/graph_rag.py` | GraphRAGRetriever |
| `bookmind_tutor/knowledge_graph/builder.py` | KnowledgeGraphBuilder |
| `tests/knowledge_graph/test_models.py` | |
| `tests/knowledge_graph/test_extractor.py` | mock LLM |
| `tests/knowledge_graph/test_graph_store.py` | mock Neo4j driver |
| `tests/knowledge_graph/test_graph_rag.py` | mock cả hai |
| `tests/knowledge_graph/test_builder.py` | mock extractor + store |
| `scripts/build_kg.py` | CLI: build KG từ PDF, print stats |

---

## Test plan

Tất cả unit tests mock external dependencies (LLM + Neo4j driver).

- `test_extractor.py`: mock LLM trả về valid JSON → verify parse đúng;
  mock JSON lỗi → verify trả về ([], []).
- `test_graph_store.py`: mock `driver.session()` → verify Cypher queries đúng;
  verify upsert gọi MERGE không CREATE.
- `test_graph_rag.py`: mock VectorStore + GraphStore → verify merged results
  đúng và deduplication hoạt động.
- `test_builder.py`: mock EntityExtractor + GraphStore → verify stats đúng,
  failed chunks không crash toàn bộ pipeline.

---

## Thứ tự implement

1. `models.py` — dataclasses, không dependency
2. `graph_store.py` — Neo4j CRUD, test với mock driver
3. `extractor.py` — LLM extraction, test với mock client
4. `builder.py` — orchestration, test với mocks
5. `graph_rag.py` — combined retrieval, test với mocks
6. `scripts/build_kg.py` — chạy thử trên DDIA subset (5 chapters)
7. End-to-end test: build KG → query GraphRAGRetriever → so sánh với plain vector search

---

## Risk mitigation

| Rủi ro | Biện pháp |
|---|---|
| LLM extract sai entity / hallucinate | Fixed types, validate type trong parser, log warnings |
| Quá nhiều duplicate nodes | MERGE + unique constraint trên name |
| Extraction chậm (550 chunks × 1 LLM call) | Chỉ extract 1 chapter trước để calibrate, sau đó chạy full |
| JSON parse fail | Fallback ([], []), log chunk_id để debug |
| Graph traversal trả về quá nhiều nodes | Cap depth=2, max 20 related entities |

---

## Quyết định kỹ thuật cần ghi vào decisions.md

- **LLM-based extraction vs spaCy NER:** LLM vì technical terms trong DDIA
  cần hiểu ngữ nghĩa, không chỉ pattern matching.
- **Fixed types vs open-ended:** Fixed types vì dễ validate, ít hallucination,
  roadmap khuyến nghị nếu rủi ro cao.
- **Graph RAG approach:** Expand query entity → graph traversal → merge với vector,
  không replace vector search vì hai cái bổ sung nhau.
