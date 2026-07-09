# Bài học: Lazy KG Build + Derive-not-Copy cho User KG Edges

**Ngữ cảnh:** Tuần 5-6 - thiết kế edges cho Knowledge Graph cá nhân của user.

---

## Câu hỏi ban đầu

User KG chỉ có nodes, không có edges. Làm thế nào để tạo edges có ý nghĩa giữa
các concepts user đã approve?

---

## Sai lầm đầu tiên: giả định dữ liệu tồn tại

Bước đầu tôi đề xuất dùng book KG (Entity nodes trong Neo4j) làm nguồn sự thật cho
relations. Lý luận đúng về mặt thiết kế, nhưng có một vấn đề: **book KG không tồn tại
trong app flow**.

`KnowledgeGraphBuilder` chỉ được gọi từ `scripts/build_kg.py` và `scripts/demo_e2e.py`.
Khi user upload PDF qua Streamlit, chỉ có `ChunkIndexer.index_pdf()` chạy - chunks vào
ChromaDB, không có Entity nodes nào được tạo trong Neo4j.

**Bài học:** Trước khi thiết kế dựa trên một nguồn dữ liệu, hãy trace ngược xem nó có
thực sự được build trong production flow không. Đọc code > đọc tên class.

---

## Sai lầm thứ hai: co-occurrence nghe đơn giản nhưng vô nghĩa

Một đề xuất nhanh: khi user approve A, B, C từ cùng một batch → tạo `RELATED_TO` edge
giữa tất cả. Đơn giản, không cần LLM.

Nhưng co-occurrence không phải relation. Nếu một answer về "event sourcing" đề cập
đến "Kafka" và "TCP protocol", điều đó không có nghĩa là Kafka IS_A TCP hay Kafka
PART_OF TCP. Edge `RELATED_TO` không có semantic content - nó chỉ nói "hai thứ này
từng xuất hiện gần nhau". Graph nhìn có vẻ connected nhưng không giúp ích gì cho
traversal hay visualization có ý nghĩa.

**Bài học:** Một edge "nào đó" không tốt hơn không có edge, nếu edge đó không phản
ánh quan hệ thực.

---

## Phát hiện: dữ liệu tốt đã có, đang bị bỏ đi

Nhìn lại `KGUpdateSuggester._extract_cached()`:

```python
entities, _ = self._extractor.extract(chunk)
#             ^^^
#             relations đang bị bỏ đi hoàn toàn
```

`EntityExtractor.extract()` đã trả về cả entities lẫn typed relations (IS_A, PART_OF,
CONTRADICTS...). Chúng đang bị discard bằng một dấu `_`.

**Bài học:** Khi cần dữ liệu mới, đọc lại những gì code đang làm với dữ liệu hiện có.
Đôi khi giải pháp là bỏ dấu `_` đi, không phải viết thêm code.

---

## Thiết kế cuối cùng: Lazy Build + Derive at Read Time

### Lazy Build

Thay vì build toàn bộ book KG lúc upload (300+ LLM calls, 5-10 phút), build lazily:
mỗi khi `KGUpdateSuggester` extract một chunk để suggest entities, persist cả
entities + relations vào book KG (`Entity` label trong Neo4j) ngay lúc đó.

Book KG lớn dần theo đúng những phần user thực sự đọc. Không trả tiền cho phần user
không bao giờ hỏi tới.

### Derive Edges at Read Time, không copy

Khi cần hiển thị edges trong user KG, không lưu edge vào `UserEntity` nodes.
Thay vào đó, derive lúc đọc:

```cypher
MATCH (s:Entity)-[r]->(t:Entity)
WHERE s.name IN $user_entity_names AND t.name IN $user_entity_names
RETURN s.name, type(r), t.name
```

Tại sao derive thay vì copy?

**Copy edge lúc approve** có bug tinh vi: nếu user approve A ở session 1 và B ở session
2, và relation A→B chỉ được discovered ở session 3 (trong một chunk khác) - ta không
có approve event nào để trigger copy nữa. A và B đều đã known, không có suggestion,
không có copy. Edge bị miss vĩnh viễn.

**Derive at read time**: edge tự xuất hiện ngay khi book KG có relation đó, bất kể
khi nào nó được extracted. Một nguồn sự thật, không cần sync logic, không có class
của bug đồng bộ.

---

## Pattern tổng quát

**Lazy Build + Derive** là pattern hữu ích khi:
- Data expensive để build (LLM calls, API calls) → build lazily khi có demand
- Derived view cần tự cập nhật khi source thay đổi → compute at read time
- Source of truth và view sống ở different lifecycles → tách biệt, đừng copy

Pattern này thường gặp trong database design (materialized view vs live query),
event sourcing (projection từ events), và cache invalidation.

---

## Tóm tắt thay đổi thực tế

- `suggester.py`: cache giờ giữ `(entities, relations)`, persist cả hai vào book KG
- `graph_store.py`: `get_user_graph_data()` derive edges từ Entity relations;
  `get_user_related_entities()` traverse Entity graph, filter về UserEntity names
- Không thay đổi gì ở UI hay approval flow - edges tự xuất hiện sau vài turns chat
