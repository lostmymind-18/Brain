# Plan: Session Persistence (Book Library + Chat History)

## Problem

All state lives in Streamlit `session_state` and an ephemeral ChromaDB client.
When the server restarts, the user loses everything: indexed books must be re-uploaded and re-embedded (~30s each), and all conversation history is gone.

Expected behavior: restart the app, see the library of previously indexed books, open one, and continue the conversation where it left off.

## What to persist and what not

| State | Persist? | Why |
|---|---|---|
| ChromaDB embeddings | Yes | Most expensive to recompute |
| Book registry (id, name, chunk count, indexed date) | Yes | Needed to rebuild the library list |
| Chat messages (role, content, msg_id) | Yes | Continue the conversation |
| `trace`, `sources`, `suggestions` per message | No | Runtime dataclasses, UI decoration only; cheap to live without after reload |
| `approved_msg_ids` | No | Only meaningful while suggestions exist; restored messages have no suggestions |
| User KG | Already persisted | Neo4j has its own volume |

Consequence of not persisting sources/trace: restored messages render as plain text without the trace/sources expanders.
This is acceptable; new messages get them as usual.

## Storage layout

```
data/
  pdf/                # existing, gitignored
  chroma/             # ChromaDB PersistentClient directory (new)
  library.json        # book registry
  history/
    {book_id}.json    # one file per book
```

`data/chroma/`, `data/history/`, `data/library.json` added to `.gitignore`.

File formats (both carry a schema version for future migration):

```json
// library.json
{
  "version": 1,
  "books": {
    "fundamental_of_software_architecture": {
      "name": "Fundamental of software architecture",
      "n_chunks": 543,
      "indexed_at": "2026-07-09T10:00:00"
    }
  }
}
```

```json
// history/{book_id}.json
{
  "version": 1,
  "messages": [
    {"role": "assistant", "content": "intro...", "msg_id": 0},
    {"role": "user", "content": "question...", "msg_id": null}
  ]
}
```

## Design decisions

**JSON files, not SQLite.**
Single local user, small data, human-readable for debugging.
SQLite adds schema management for no benefit at this scale.

**One class, `LibraryStore`, in `bookmind_tutor/tutor/persistence.py`.**
Single responsibility: serialize/deserialize library + history to a root directory.
No Streamlit imports - pure I/O, fully unit-testable with `tmp_path`.

```python
@dataclass
class BookRecord:
    name: str
    n_chunks: int
    indexed_at: str  # ISO timestamp

class LibraryStore:
    def __init__(self, root: str | Path) -> None: ...
    def load_library(self) -> dict[str, BookRecord]: ...
    def save_library(self, books: dict[str, BookRecord]) -> None: ...
    def load_history(self, book_id: str) -> list[dict]: ...
    def save_history(self, book_id: str, messages: list[dict]) -> None: ...
    def delete_history(self, book_id: str) -> None: ...   # future use
```

**Atomic writes.**
Write to a `.tmp` sibling then `os.replace()`.
A crash mid-write can never corrupt the existing file.

**Corrupt or missing files never crash the app.**
`load_*` catches `json.JSONDecodeError` / `OSError`, logs a warning, returns empty.
Worst case equals current behavior (start fresh).

**History serialization strips runtime fields.**
`save_history` keeps only `role`, `content`, `msg_id` from each message dict.
`load_history` re-adds defaults: `trace=None`, `sources=[]`, `suggestions=[]`.
This keeps persisted files stable even if UI-side message fields evolve.

**ChromaDB switches to persistent mode in the app only.**
`VectorStore(persist_dir="data/chroma", collection_name=f"book_{bid}")`.
Tests keep using the ephemeral default - no test changes needed.
`get_or_create_collection` means restart reconnects to existing embeddings for free.

**Agent memory is seeded from restored history.**
New method `QAAgent.seed_memory(messages: list[dict])`:
- Replays persisted user/assistant text turns into its `ConversationMemory`.
- Skips leading assistant messages (the book intro) - the Anthropic API requires the first message to be a `user` turn.
- Sets `_turn_count` to the number of user turns, so Socratic/Feynman triggers resume at the right cadence.
- `ConversationMemory.max_turns` trimming applies naturally, so very long histories are capped.

Called from `_setup_agent()` right after constructing the QAAgent.

## App integration points

**Startup (`_init_state`), runs once per session:**
1. `library = LibraryStore("data")`, stored in `session_state`.
2. `load_library()`; for each book, create `VectorStore(persist_dir="data/chroma", collection_name=f"book_{bid}")`.
3. Stale-record guard: if the collection `count() == 0` (chroma dir deleted but library.json survived), skip the book and show one `st.warning` listing skipped books; user can re-upload.
4. `load_history(bid)` for each restored book.
5. `next_msg_id = max(all msg_ids) + 1` (0 if none) - avoids Streamlit form-key collisions.

**Save points (write-through, files are small):**
- After successful indexing: `save_library()`.
- After intro generation: `save_history(bid)`.
- After each assistant turn in `_handle_question` (including the error-placeholder path): `save_history(bid)`.

**`_setup_agent()`:** after creating QAAgent, call `seed_memory(_current_messages())`.

## Known limitations (documented, not solved)

- No "delete book" UI yet; removing a book means deleting its chroma collection + registry entry + history file by hand. Goes to `backlog.md`.
- Two concurrent sessions write last-wins. Single-user local app; acceptable.
- Restored messages lose trace/sources expanders (see above).
- The embedding model itself (all-MiniLM-L6-v2) still loads once per process at startup; that cost is unavoidable and unrelated to this change.

## Tests

New `tests/tutor/test_persistence.py` (pure I/O, `tmp_path`, no mocks needed):
- Library round-trip: save then load returns equal records.
- History round-trip: runtime fields stripped on save, defaults restored on load.
- Missing files return empty dict/list.
- Corrupt JSON returns empty and does not raise.
- `os.replace` atomicity is trusted, not tested.

Extend `tests/agents/test_qa_agent.py`:
- `seed_memory` skips leading assistant messages.
- `seed_memory` restores `_turn_count` from user-turn count.
- Seeded memory appears in the messages passed to the router on the next `chat()`.

## Scope

- New: `bookmind_tutor/tutor/persistence.py`, `tests/tutor/test_persistence.py`.
- Modified: `app.py` (init/save hooks, persistent VectorStore), `qa_agent.py` (`seed_memory`), `.gitignore`.
- Not touched: ingestion, retrieval internals, strategies, KG modules.
