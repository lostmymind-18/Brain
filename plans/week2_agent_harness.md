# Week 2 Plan: Agent Harness Foundations

## Mục tiêu

Tuần này xây hai module mới:

1. **`bookmind_tutor/retrieval/`** - kết nối ingestion pipeline (chunks) với vector store để
   phục vụ RAG.
   Đây là phần còn thiếu từ Tuần 1 ("Basic RAG chat") và là prerequisite cho `BookSearchTool`.
2. **`bookmind_tutor/agents/`** - AgentHarness với ReAct loop, tool system, conversation memory,
   logging, và error handling.

Học được từ tuần này: cách một agent loop thực sự hoạt động từ bên trong - không dùng LangChain
hay LangGraph, mà dùng thẳng Anthropic SDK để hiểu rõ từng bước.

---

## Cấu trúc thư mục mới

```
bookmind_tutor/
  retrieval/
    __init__.py           # export: VectorStore, ChunkIndexer, SearchResult
    vector_store.py       # ChromaDB wrapper
    indexer.py            # ChunkIndexer: PDF → chunks → vector store
    models.py             # SearchResult dataclass

  agents/
    __init__.py           # export: AgentHarness, BookSearchTool
    models.py             # AgentMessage, ToolSpec, ToolCallResult
    harness.py            # AgentHarness: orchestration + ReAct loop
    executor.py           # ToolExecutor: retry + error handling
    memory.py             # ConversationMemory
    tools/
      __init__.py
      base.py             # Tool abstract base class
      book_search.py      # BookSearchTool

tests/
  retrieval/
    test_vector_store.py
    test_indexer.py
  agents/
    test_harness.py
    test_executor.py
    test_book_search.py
```

---

## Module 1: `bookmind_tutor/retrieval/`

### `models.py`

```python
@dataclass
class SearchResult:
    chunk_id: str
    text: str
    chapter: str | None
    section: str | None
    page_range: tuple[int, int]
    score: float           # cosine similarity from ChromaDB (higher = better)
```

### `vector_store.py` - VectorStore

Wraps ChromaDB.
Dùng ChromaDB's built-in `DefaultEmbeddingFunction` (sentence-transformers/all-MiniLM-L6-v2)
- không cần API key, chạy local.

```python
class VectorStore:
    def __init__(self, persist_dir: str | None = None, collection_name: str = "chunks")
    def index_chunks(self, chunks: list[Chunk]) -> None
    def search(self, query: str, k: int = 5) -> list[SearchResult]
    def count(self) -> int
    def clear(self) -> None
```

- `persist_dir=None` → in-memory collection (cho tests).
- `persist_dir="path"` → ChromaDB PersistentClient, data sống qua sessions.
- `index_chunks` dùng `upsert` (idempotent theo `chunk_id`) để có thể re-index không bị duplicate.
- `search` trả về list SearchResult sorted by score descending.

### `indexer.py` - ChunkIndexer

Kết nối ingestion pipeline với vector store.

```python
class ChunkIndexer:
    def __init__(self, vector_store: VectorStore)
    def index_pdf(self, pdf_path: str, llm_client=None) -> int
        # Returns number of chunks indexed
```

Flow: `PdfExtractor.extract()` → `StructureDetector.detect()` → `HierarchicalChunker.chunk()` →
`VectorStore.index_chunks()`.

---

## Module 2: `bookmind_tutor/agents/`

### `models.py`

```python
@dataclass
class ToolSpec:
    """Describes a tool to Claude - maps directly to Anthropic tools format."""
    name: str
    description: str
    input_schema: dict   # JSON Schema object

    def to_anthropic_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": self.input_schema}


@dataclass
class ToolCallResult:
    tool_use_id: str
    content: str
    is_error: bool = False

    def to_anthropic_dict(self) -> dict:
        d = {"type": "tool_result", "tool_use_id": self.tool_use_id, "content": self.content}
        if self.is_error:
            d["is_error"] = True
        return d
```

### `tools/base.py` - Tool

```python
from abc import ABC, abstractmethod

class Tool(ABC):
    @property
    @abstractmethod
    def spec(self) -> ToolSpec: ...

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Return a string result. Raise an exception on failure."""
```

### `tools/book_search.py` - BookSearchTool

```python
class BookSearchTool(Tool):
    """Search the indexed book for passages relevant to a query."""

    def __init__(self, vector_store: VectorStore, k: int = 5)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="book_search",
            description="Search the book for passages relevant to a question.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            }
        )

    def execute(self, query: str) -> str:
        results = self._vector_store.search(query, k=self._k)
        if not results:
            return "No relevant passages found."
        parts = []
        for i, r in enumerate(results, 1):
            loc = f"[{r.chapter or 'unknown'} / {r.section or '–'}, p.{r.page_range[0]}]"
            parts.append(f"{i}. {loc}\n{r.text}")
        return "\n\n".join(parts)
```

### `executor.py` - ToolExecutor

Nhận tool_use blocks từ API response, dispatch sang đúng Tool, handle errors.

```python
class ToolExecutor:
    def __init__(self, tools: list[Tool], max_retries: int = 3)
    def execute_all(self, content_blocks: list) -> list[ToolCallResult]

    def _execute_one(self, tool_use_block) -> ToolCallResult:
        # Retry up to max_retries for transient errors
        # Return ToolCallResult(is_error=True) on permanent failure
```

- Retry policy: chỉ retry nếu exception là transient (network/timeout).
  Validation errors (wrong args) không retry.
- Log mỗi tool call: `logger.info("Tool %s(%s) → %s", name, args_summary, status)`.

### `memory.py` - ConversationMemory

```python
class ConversationMemory:
    """Manages the messages list passed to the Anthropic API."""

    def __init__(self, max_turns: int = 50)
    def add_user(self, content: str | list) -> None
    def add_assistant(self, content: list) -> None   # raw content blocks
    def add_tool_results(self, results: list[ToolCallResult]) -> None
    def messages(self) -> list[dict]
    def clear(self) -> None
```

- `max_turns`: khi vượt quá, drop cặp oldest user+assistant (giữ lại system context).
  Week 2 chỉ cần conversation memory cơ bản.
  Episodic và semantic memory để Week 4 (KG integration).

### `harness.py` - AgentHarness

Core class của tuần này.

```python
class AgentHarness:
    def __init__(
        self,
        tools: list[Tool],
        model: str = "claude-haiku-4-5-20251001",
        system_prompt: str = _DEFAULT_SYSTEM,
        max_iterations: int = 10,
        llm_client: anthropic.Anthropic | None = None,
    )

    def chat(self, user_message: str, memory: ConversationMemory | None = None) -> str:
        """
        Run one user turn through the ReAct loop.
        Returns the final text response.

        If memory is None, creates a fresh ConversationMemory (stateless single turn).
        Pass in an existing ConversationMemory for multi-turn conversation.
        """
```

**ReAct loop** trong `chat()`:

```
memory.add_user(user_message)

for step in range(max_iterations):
    log: "Step N — calling Claude"
    response = client.messages.create(model, tools, memory.messages())

    if stop_reason == "end_turn":
        text = extract_text(response.content)
        log: "Done — final answer"
        break

    if stop_reason == "tool_use":
        memory.add_assistant(response.content)
        results = executor.execute_all(response.content)
        memory.add_tool_results(results)
        log: "Tool results added — continuing loop"
        continue

else:
    # max_iterations reached
    text = "(Max iterations reached. Partial answer follows.) ..."
    log warning

memory.add_assistant([{"type": "text", "text": text}])
return text
```

**Lý do không dùng streaming ở Week 2:**
Streaming làm phức tạp hóa việc hiểu cấu trúc response (content blocks).
Week 2 tập trung vào đúng flow của ReAct loop.
Streaming có thể thêm sau ở Week 5-6 khi build tutor UI.

**Model mặc định: `claude-haiku-4-5-20251001`**
Haiku đủ mạnh cho RAG Q&A, chi phí thấp khi thử nghiệm.
AgentHarness nhận `model` parameter nên dễ đổi.

---

## Decisions

### Tại sao không dùng LangChain/LangGraph?

Mục tiêu của Tuần 2 là *hiểu* Agent Harness - cách Anthropic tool use API hoạt động,
cách messages list được build lên qua nhiều vòng lặp, cách tool_use và tool_result connect
với nhau.
Dùng framework ẩn những chi tiết này đi và mất đi learning objective chính.
Tuần 3 có thể compare với LangGraph để thấy tradeoff.

### Tại sao ChromaDB + sentence-transformers?

- Không cần API key cho embeddings.
- Persistent, embedded DB - không cần server riêng.
- DefaultEmbeddingFunction của ChromaDB dùng `all-MiniLM-L6-v2` - lightweight, fast, đủ tốt.
- Nếu sau này cần embedding quality tốt hơn, chỉ cần swap `EmbeddingFunction`.

### Tại sao tách `retrieval/` thành module riêng?

`VectorStore` và `ChunkIndexer` sẽ được dùng bởi nhiều component sau:
- `BookSearchTool` (agents/) - Tuần 2
- TutorAgent (tutor/) - Tuần 5-6
- Knowledge Graph builder (knowledge_graph/) - Tuần 4

Đặt trong `agents/` sẽ tạo coupling không cần thiết.

---

## Sequence of Implementation

1. `bookmind_tutor/retrieval/models.py` + `vector_store.py` + tests
2. `bookmind_tutor/retrieval/indexer.py` + tests
3. `bookmind_tutor/agents/models.py`
4. `bookmind_tutor/agents/tools/base.py` + `book_search.py` + tests
5. `bookmind_tutor/agents/executor.py` + tests
6. `bookmind_tutor/agents/memory.py` + tests
7. `bookmind_tutor/agents/harness.py` + tests
8. Integration smoke test: index DDIA → ask a question → get answer

---

## Evaluation Criteria (từ roadmap)

- [x] Harness chạy ổn định qua nhiều vòng ReAct loop
- [x] Logging rõ ràng: mỗi bước log tool name + args summary + status
- [x] Tool failure không crash harness (is_error=True response tiếp tục loop)
- [x] Retry 3x on transient API errors
- [x] Max iterations fallback message
- [x] BookSearchTool trả về relevant passage từ indexed book

---

## Dependencies mới

```
chromadb>=0.5
sentence-transformers>=3.0   # pulled in by chromadb's DefaultEmbeddingFunction
```

Thêm vào `requirements.txt`.
