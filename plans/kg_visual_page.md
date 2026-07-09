# Plan: Knowledge Graph Visual Page

## Mục tiêu

Thêm một trang riêng hiển thị user KG dưới dạng đồ thị tương tác (nodes + edges)
và cho phép user truy vấn: lọc theo tên/type, khám phá neighborhood của một node.

## Library

`streamlit-agraph` - Streamlit-native component dùng vis.js.
- API sạch: `agraph(nodes, edges, config)` trả về ID node vừa click → dùng để hiện detail panel
- Hỗ trợ: drag, zoom, physics simulation, màu theo type, label trên edge
- Không cần viết HTML/JS tay

## Components

### 1. `GraphStore.get_user_graph_data()` - query mới

```cypher
MATCH (e:UserEntity)
RETURN e.name, e.type, e.description

MATCH (s:UserEntity)-[r]->(t:UserEntity)
RETURN s.name, type(r), t.name
```

Returns `{"nodes": [...], "edges": [...]}` — tất cả node và edge trong user KG.

### 2. Streamlit multi-page app

Streamlit tự động tìm `pages/` directory cạnh file entry point.
- Main: `bookmind_tutor/tutor/app.py` (chat, không đổi)
- Pages dir: `bookmind_tutor/tutor/pages/`
- KG page: `bookmind_tutor/tutor/pages/1_Knowledge_Graph.py`

Session state được share giữa các pages — `graph_store` khởi tạo ở `app.py`
có thể dùng được trên KG page. KG page tự init `graph_store` nếu chưa có.

### 3. UI của KG page

```
Header: "🧠 Knowledge Graph"
Subtitle: số nodes + edges hiện tại

[Search concepts...]    [Filter by type: All / CONCEPT / SYSTEM / ...]

┌──────────────────────────────────┐  ┌─────────────────────┐
│                                  │  │ Node Detail         │
│    Force-directed graph          │  │                     │
│    (agraph component)            │  │ Name: ...           │
│                                  │  │ Type: ...           │
│    Node colors by type:          │  │ Desc: ...           │
│    CONCEPT=blue, SYSTEM=green    │  │                     │
│    PRINCIPLE=orange              │  │ Connected to: ...   │
│    DEFINITION=purple             │  └─────────────────────┘
│    PERSON=red                    │
└──────────────────────────────────┘

─── Neighborhood Explorer ─────────────────────────────────
Select concept: [dropdown]  Depth: [1 / 2]  [Show]
→ Re-renders graph showing only the selected subgraph
```

### 4. Query modes

**Mode 1 — Full graph (default)**:
Hiển thị tất cả UserEntity nodes và edges. Text search lọc nodes,
unmatched nodes mờ đi (opacity thấp hơn). Type filter ẩn/hiện nodes theo type.

**Mode 2 — Neighborhood**:
Chọn 1 concept từ dropdown + depth (1 hoặc 2) → gọi
`graph_store.get_user_related_entities(name, depth)` → render subgraph gồm
concept đó và neighbors của nó.

### 5. Node detail panel

`agraph()` returns ID của node vừa click (string hoặc None).
Nếu có click → hiển thị name, type, description + danh sách connected nodes
trong col bên phải.

## Node styling

| Type | Color | Size |
|---|---|---|
| CONCEPT | #4A90D9 (blue) | 20 |
| SYSTEM | #27AE60 (green) | 25 |
| PRINCIPLE | #E67E22 (orange) | 20 |
| DEFINITION | #8E44AD (purple) | 18 |
| PERSON | #E74C3C (red) | 18 |
| (unknown) | #95A5A6 (grey) | 15 |

Edge: label = relation type, color = #BDC3C7

## Edge types hiện có trong user KG

Hiện tại suggester chỉ upsert nodes, chưa tạo edges. Graph sẽ là disconnected nodes.
Đây là bình thường ở giai đoạn này — graph layout vẫn useful để xem "bộ nhớ" của user.
TODO future: khi user confirm relation giữa 2 nodes, lưu edge.

## Files thay đổi

1. `bookmind_tutor/knowledge_graph/graph_store.py` — thêm `get_user_graph_data()`
2. `requirements.txt` — thêm `streamlit-agraph`
3. `bookmind_tutor/tutor/pages/1_Knowledge_Graph.py` — trang mới (tạo mới)
4. `bookmind_tutor/tutor/pages/__init__.py` — file rỗng để Python nhận diện package

## Phạm vi không làm

- Không tạo auto-edges dựa trên co-occurrence (phức tạp, ngoài scope)
- Không natural language query vào KG (để sau)
- Không edit/delete node trực tiếp từ graph (có "Clear" button trên chat page là đủ)
