# Plan: User KG Edges via Lazy Book KG

## Vấn đề

User KG hiện tại chỉ có `UserEntity` nodes, không có edges. `KGUpdateSuggester`
đang extract cả entities lẫn relations từ retrieved chunks nhưng đang vứt relations đi.

## Thiết kế

### Nguồn sự thật cho edges: book KG (Entity nodes)

Book KG (`Entity` + typed `Relation` edges) được xây **lazily** trong quá trình chat:
mỗi khi `KGUpdateSuggester` extract entities từ retrieved chunks, relations được
persist vào book KG luôn. User chỉ thấy subset: những relations nào mà cả hai đầu
đều là concepts user đã approve (UserEntity).

### Tại sao derive thay vì copy edges vào UserEntity?

Nếu copy edge lúc approve: user approve A và B riêng lẻ (khác session), ta không
biết khi nào cả hai đều đã có → edge bị miss. Với derive, mỗi lần đọc graph ta
query book KG và tự động thấy edge khi cả hai đầu đã tồn tại - không cần sync logic.

### Tại sao build book KG lazily (không build toàn bộ lúc upload)?

- Upload 300+ chunks × 1 LLM call = 5-10 phút thêm vào upload time
- Lazy: chỉ extract chunk nào thực sự được retrieve → không tốn LLM call cho
  phần user không đọc tới
- Phù hợp với docstring của `KGUpdateSuggester`: "grows lazily, one turn at a time"
- Bonus: `GraphRAGRetriever` sẽ tự nhiên có book entities để traverse sau vài turns

## Các thay đổi

### 1. `bookmind_tutor/knowledge_graph/suggester.py`

**Cache**: đổi từ `dict[str, list[Entity]]` sang `dict[str, tuple[list[Entity], list[Relation]]]`

**`_extract_cached()`**: trả về `(entities, relations)` thay vì chỉ entities.

**`suggest()`**: sau khi extract từ mỗi chunk, upsert entities và relations vào book KG:
```python
entities, relations = self._extract_cached(result)

for entity in entities:
    user_graph.upsert_entity(entity)          # book KG (Entity label)
for relation in relations:
    user_graph.upsert_relation(relation)      # book KG relations

# rồi mới filter ra suggestions cho user KG
```

### 2. `bookmind_tutor/knowledge_graph/graph_store.py`

**`get_user_graph_data()`**: đổi edge query từ `UserEntity` relations sang derive từ
`Entity` relations giữa user-known names:
```cypher
MATCH (s:Entity)-[r]->(t:Entity)
WHERE s.name IN $user_names AND t.name IN $user_names
RETURN s.name, type(r), t.name
```

**`get_user_related_entities()`**: đổi traversal từ `UserEntity` graph sang `Entity`
graph, filter kết quả về UserEntity names:
```cypher
MATCH (center:Entity {name: $name})-[*1..{depth}]-(related:Entity)
WHERE related.name IN $user_names AND related.name <> $name
RETURN DISTINCT related LIMIT 20
-- sau đó fetch UserEntity data cho matched names
```

### 3. `tests/knowledge_graph/test_suggester.py`

- `test_caps_suggestions_at_max`: patch `_extract_cached` đổi thành `lambda r: (entities, [])`
- Thêm `test_upserts_entities_and_relations_to_book_kg`: verify `graph.upsert_entity`
  và `graph.upsert_relation` được gọi khi suggest

### 4. `tests/knowledge_graph/test_graph_store.py`

Không cần test integration cho query mới (cần real Neo4j). Các tests hiện tại
đều mock Neo4j - không bị ảnh hưởng.

## Không làm

- Không thêm `ExtractedChunk` marker vào Neo4j (để backlog — session cache đủ tốt)
- Không thay đổi `GraphRAGRetriever` (nó đã dùng book KG entities cho expansion)
- Không thay đổi `KGUpdateSuggester.suggest()` signature
