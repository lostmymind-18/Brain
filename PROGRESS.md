# PROGRESS

Each entry here must reference either a plan in `plans/` or a log in `backlog.md`.

---

## Week 1 - PDF Ingestion Pipeline

### Core implementation - complete

Built `PdfExtractor`, `StructureDetector`, `HierarchicalChunker`, data models,
custom exceptions, and `ingestion/README.md`.
40 unit tests pass. No external references (this was the baseline build; subsequent
changes are tracked in `backlog.md`).

### Post-implementation fixes - complete

All four issues discovered during real-book testing.
Each fix is documented in `backlog.md`:

- Header/footer stripping: two-phase extraction in `PdfExtractor`, strips
  repeated margin-band text before building `PageContent`. - `backlog.md`
- Cover page font filtering: `_build_heading_level_map` requires a font size to
  appear on >= N distinct pages before entering the level map. - `backlog.md`
- Ancestor page_range propagation: `_extend_page_range` updates all open
  ancestors in the stack, not just the deepest node. - `backlog.md`
- `print_tree` utility: added `utils.py` with ASCII tree visualization,
  exported from `__init__.py`. - `backlog.md`

### 3-tier structure detection - complete

All 3 tiers implemented per `plans/structure_detection_tiers.md`:

- Tier 1: `PdfExtractor.extract_toc()` reads embedded PDF bookmarks.
  `StructureDetector.detect(pages, toc=...)` uses it when >= 3 entries present.
  `_build_tree_from_toc` infers page ranges and assigns page text to nodes
  without parent/child overlap. - `plans/structure_detection_tiers.md`
- Tier 2: `_find_toc_page` scans first 20 pages for a printed Contents page.
  `_parse_toc_page_with_llm` sends the page text to `claude-haiku-4-5-20251001`
  and parses the structured JSON response. Requires `anthropic.Anthropic` client
  passed at construction time; gracefully skipped otherwise.
  - `plans/structure_detection_tiers.md`
- Tier 3: heuristic now two-phase - classify all lines first, then batch-verify
  ambiguous bold-only detections with LLM before building the tree.
  `_classify_line_with_confidence` exposes the `is_ambiguous` flag.
  - `plans/structure_detection_tiers.md`

65 tests pass (25 new tests added across test_structure_detector.py and
test_pdf_extractor.py). LLM calls mocked with `unittest.mock.MagicMock`.

### Inverted page_range bug fix - complete

Fixed `_build_tree_from_toc` in `StructureDetector`: when two consecutive TOC
entries at the same level share the same start page (e.g. "About the Author"
and "Colophon" both at the last page, or two sub-sections on one physical page),
the old formula `end = next_start - 1` produced an inverted range like `(29, 28)`.
Fix: clamp `end = max(start, next_start - 1)`.
1 regression test added.

### Tier 2 + 3 accuracy improvements - complete

Evaluated Tier 2 and Tier 3 accuracy against Tier 1 ground truth, then improved.

Tier 2 improvements:
- Multi-page TOC: `_find_toc_pages()` collects consecutive TOC pages instead of
  just the first. `_looks_like_toc_continuation()` detects continuation pages by
  counting lines ending with digits (right-aligned page numbers). Fix to not
  reject long lines that end with a digit (they are TOC entries, not body text).
  Result: recall 10.6% → 89.4%, page accuracy 90.5% → 98.3%.

Tier 3 improvements:
- Label-only filter: removes "CHAPTER 1", "PART II" decorative labels (0 content,
  0 children). Condition requires empty raw_text + no children to avoid removing
  legitimate "Chapter 1" headings that carry content.
- Split-title merge: `_is_title_fragment()` merges consecutive same-level siblings
  on the same page when the first title ends with a comma or conjunction (", and",
  " or", " but") — a clear indicator of a wrapped line. Conservative condition
  avoids merging legitimate consecutive short sections.
  Result: precision 60.4% → 64.4%, F1 72.5% → 75.3%. Recall held at 90.6%.

17 new tests added. 84 tests pass total.

### Tier 2 + 3 branch testing and fixes - complete

Exercised Tier 2 (LLM TOC parse) and Tier 3 (font-size heuristic) on real books.
Found and fixed 3 bugs:

- Markdown fence stripping: LLM wraps JSON in ```json...``` despite instruction.
  Fix: extract substring between first `[` and last `]` before `json.loads()`.
  Applied to both `_parse_toc_page_with_llm` and `_verify_candidates_with_llm`.
- None page filter: Part headers in printed TOC have no page number (LLM returns
  `page=null`). Fix: filter entries where `page` is not `int` before building tree.
- Page offset detection: printed TOC uses book page numbers, not PDF page numbers.
  Added `_detect_page_offset()` which finds chapter titles in pages > 20 (skipping
  the TOC pages themselves) and computes the mode offset. Applied in Tier 2 flow.

Known limitations documented in backlog.md (not blocking for Week 2).

### Sibling text duplication bug fix - complete

Fixed `_build_tree_from_toc` Step 4: when two sibling nodes share a start page,
both were receiving that page's full text because the assignment loop only excluded
child pages, not pages already claimed by earlier siblings. This caused duplicate
chunks in the RAG pipeline.
Fix: added a global `claimed: set[int]` in DFS order - the first sibling claims
a shared page, subsequent siblings skip it.
1 regression test added. 67 tests pass.

---

## Week 2 - Agent Harness Foundations

### Core implementation - complete

Built two new modules per `plans/week2_agent_harness.md`:

**`bookmind_tutor/retrieval/`** - connects ingestion pipeline to vector search:
- `VectorStore`: ChromaDB + sentence-transformers/all-MiniLM-L6-v2 embeddings.
  Ephemeral (in-memory) for tests, persistent on disk for production.
  Upsert-based indexing (idempotent), cosine similarity search.
- `ChunkIndexer`: orchestrates PdfExtractor → StructureDetector → HierarchicalChunker → VectorStore.
- `SearchResult` dataclass with chapter, section, page_range, score.

**`bookmind_tutor/agents/`** - AgentHarness with ReAct loop:
- `AgentHarness`: ReAct loop using Anthropic tool_use API directly (no LangChain).
  Supports stateless single-turn and stateful multi-turn via ConversationMemory.
  Logs each step (step number, tool name, token usage, final answer length).
  Returns a fallback message if max_iterations is reached without end_turn.
- `ConversationMemory`: manages messages list for Anthropic API.
  Trims oldest user+assistant pairs when max_turns exceeded.
- `ToolExecutor`: dispatches tool_use blocks to Tool instances.
  Retries up to 3x on transient errors (ConnectionError, TimeoutError, OSError).
  Non-retryable errors → ToolCallResult(is_error=True) so harness continues.
- `Tool` (abstract base) + `BookSearchTool`: searches VectorStore, returns
  numbered excerpts with chapter/section/page citations.
- `ToolSpec` + `ToolCallResult` dataclasses with to_anthropic_dict() helpers.

47 new tests added (retrieval: 16, agents: 31 across harness/executor/memory/book_search).
131 tests pass total. LLM calls mocked in all harness and indexer tests.

---

## Week 3 - Advanced Reasoning Strategies

### Core implementation - complete

Built `bookmind_tutor/agents/strategies/` per `plans/week3_reasoning_strategies.md`:

- `ReActStrategy`: thin wrapper around `AgentHarness` — uniform interface, no added logic.
- `PlanExecuteStrategy`: 3-phase pipeline (Plan → Execute sub-tasks → Synthesize).
  Planning uses a no-tool LLM call returning JSON sub-task list (max 5).
  Each sub-task runs through `AgentHarness` (stateless).
  Falls back to single-task if plan JSON is unparseable.
- `ReflexionStrategy`: calls `AgentHarness` for initial answer, then a reflection
  LLM call scores quality (1–5) on citations, depth, completeness.
  Retries once with feedback if score < 4. Graceful fallback if reflection fails.
- `StrategyRouter`: keyword heuristic routes to the right strategy.
  "compare/vs/differ/contrast" → PlanExecute; "in depth/detailed/thorough" → Reflexion;
  default → ReAct. Known limitations documented in `router.py`.
- `ReasoningTrace` dataclass added to `models.py` — records steps, num_llm_calls,
  elapsed_seconds per chat() call for comparison.

33 new tests added in `tests/agents/strategies/`. 164 tests pass total.
End-to-end demo: `scripts/demo_strategies.py` (side-by-side or interactive per strategy).

---

## Week 4 - Knowledge Graph & Long-term Memory

### Infrastructure - complete

Neo4j 5 chạy qua Docker (`bookmind-neo4j` container, port 7687).
Persistent volume `bookmind-neo4j-data`. Credentials trong `.env`.
Python driver `neo4j==6.2.0` installed.

### Core implementation - complete

Built `bookmind_tutor/knowledge_graph/` per `plans/week4_knowledge_graph.md`:

- `models.py`: `Entity`, `Relation`, `BuildStats` dataclasses. Fixed entity types
  (CONCEPT, SYSTEM, PRINCIPLE, DEFINITION, PERSON) and relation types
  (IS_A, RELATED_TO, CONTRADICTS, PART_OF, DEFINED_IN).
- `graph_store.py`: `GraphStore` — Neo4j CRUD via Python driver. All writes use
  MERGE (idempotent). Unique constraint on Entity.name. Entity name cache for
  efficient substring matching in GraphRAG.
- `extractor.py`: `EntityExtractor` — LLM-based extraction from Chunk objects.
  Returns ([], []) on any failure. Validates type against fixed sets. Drops
  relations referencing entities not in the extracted set.
- `builder.py`: `KnowledgeGraphBuilder` — orchestrates extraction → upsert.
  Per-chunk error isolation (one failure does not abort the build).
  Supports `max_chunks` for partial builds during calibration.
- `graph_rag.py`: `GraphRAGRetriever` — query expansion via graph traversal.
  Matches entity names in query → 1-hop graph traversal → expand query string
  → second vector search → merge and deduplicate results.

35 new tests in `tests/knowledge_graph/`. 199 tests pass total.
Demo: `scripts/build_kg.py` — builds KG from PDF then compares GraphRAG vs plain vector search.

### Post-implementation fixes - complete

- **Neo4j cartesian product warning**: fixed `_RELATION_QUERIES` in `graph_store.py` to use two
  separate `MATCH` clauses instead of comma-separated form. Query planner now does direct index
  lookups instead of cross-join. Warning eliminated completely.
- **JSON truncation in extractor**: increased `max_tokens` 1024 → 2048 in `EntityExtractor._call_llm`.
  Recovered 71 entities and 49 relations that were silently dropped (6/20 chunks affected).

### Integration checkpoint - complete

Wired `GraphRAGRetriever` into `AgentHarness` via `GraphRAGSearchTool`:
- `bookmind_tutor/agents/tools/graph_rag_search.py`: `GraphRAGSearchTool` wraps `GraphRAGRetriever`,
  exposes the same `book_search` tool name as `BookSearchTool` — harness is unaware of the switch.
- Exported from `bookmind_tutor/agents/tools/__init__.py`.
- 5 new tests in `tests/agents/test_graph_rag_search.py`. 204 tests pass total.
- End-to-end demo: `scripts/demo_e2e.py` — runs full pipeline (PDF → VectorStore + KG → GraphRAGSearchTool → StrategyRouter → AgentHarness) in interactive chat mode.

---

## Week 5–6 - Responsive Q&A Agent + Streamlit Frontend

### User Knowledge Graph (Personal Brain) - partially complete

Extended `bookmind_tutor/knowledge_graph/` to support a per-user knowledge graph that grows through conversation:

- `graph_store.py`: added `upsert_user_entity`, `get_user_entity_names`, `get_user_related_entities`,
  `clear_user_graph` — all operating on `UserEntity` nodes distinct from book `Entity` nodes.
- `models.py`: added `KGUpdateSuggestion`, `KGUpdateResult` dataclasses.
- `suggester.py`: `KGUpdateSuggester` — after each turn, extracts entities from retrieved chunks,
  filters out names already in the user KG, returns up to 5 new suggestions.
  LLM extraction results cached by `chunk_id` within the session to avoid re-calling.
- 7 new tests in `tests/knowledge_graph/test_suggester.py`. 211 tests pass total.

**GraphRAGRetriever personalization - complete**

Updated `graph_rag.py` to use the user's personal KG as a second expansion source:
- `_match_book_entities(query_lower)` - renamed from `_match_query_entities`, matches book graph entities in query.
- `_match_user_entities(query_lower)` - new method, matches user KG entity names in query.
  If the user knows "architecture pattern" and asks about "layered pattern", those names
  are added to the expansion query so retrieval surfaces chunks that bridge what the user
  knows to what they're learning.
- Expansion now merges both sources: `book_expansion | user_expansion`.
  Cold start (empty user KG) falls back to original behaviour.
- `app.py` switched from `BookSearchTool` to `GraphRAGSearchTool` + `GraphRAGRetriever`
  wired to session `graph_store`, so personalization is live in the UI.
- 3 new tests added in `test_graph_rag.py`. 214 tests pass total.

### Streamlit UI - partially complete

`bookmind_tutor/tutor/app.py` (262 lines) implements the main UI:

- Upload page: file uploader → `ChunkIndexer.index_pdf()` → session state ready.
- Chat page: `st.chat_message` for user/assistant turns.
  After each answer shows strategy name, LLM call count, and elapsed time via `st.caption`.
- KG suggestions: expander with one checkbox per suggested entity, form submit saves approved
  entities to Neo4j via `graph_store.upsert_user_entity`.
- Sidebar: displays user KG entities (sorted list) and a "Clear" button.

**Per-book vector store isolation - complete**

Redesigned session state so each book gets its own isolated ChromaDB collection,
per `plans/multi_book_isolation.md`:

- `book_stores: dict[book_id, VectorStore]` - a new `VectorStore(collection_name=f"book_{book_id}")`
  is created at index time; queries never cross books.
- `book_names: dict[book_id, str]` and `book_histories: dict[book_id, list]` - display name
  and per-book conversation history.
- `current_book_id` replaces the old `indexed: bool` flag as the routing signal.
- `graph_store` (user KG) remains shared across all books — it is the user's personal brain,
  not tied to any single book.
- `_book_id(filename)` derives a stable, readable collection key from the PDF filename.
- `_switch_book(book_id)` atomically sets the active book and re-initialises the agent.
- Upload page shows "Already indexed" section with "Chat →" button per book when returning
  to upload more books.
- Sidebar Library section shows all indexed books; active book displayed as plain text (●),
  inactive books as clickable buttons — switching updates title, agent, and restores that
  book's conversation history.
- Verified end-to-end with 2 books (Fundamentals of SA + DDIA): answers cite correct book,
  no cross-contamination; KG shared correctly across both.

**Not yet implemented:**
- Sidebar strategy selector (Auto / ReAct / Plan-and-Execute / Reflexion) — current app
  always uses `StrategyRouter` auto-routing with no manual override.
- Source display (chapter, page range) under each answer.

### QAAgent + pedagogical techniques - complete

Per `plans/qa_agent.md`:

- `bookmind_tutor/tutor/qa_agent.py`: `QAAgent` wraps `StrategyRouter` + `AgentHarness`.
  Manages its own `ConversationMemory` for multi-turn context, updates `harness.system_prompt`
  dynamically before each LLM call — no harness reconstruction needed.
- `AgentHarness` extended with `system_prompt` property (getter + setter) so `QAAgent` can
  update it in-place without recreating the strategies.
- `app.py`: `_setup_agent()` now creates `QAAgent(harness, router)` instead of exposing the
  router directly. `_handle_question()` calls `qa_agent.chat(question)` and reads
  `qa_agent.last_trace` for the strategy trace caption.

**Pedagogical techniques implemented:**
- Student modeling (keyword heuristics, no extra LLM calls):
  - `_infer_level(question)` scans last 6 user messages from `ConversationMemory` + current question.
  - Beginner signals: "I don't understand", "in simple terms", "can you explain", "how does X work".
  - Advanced signals: "tradeoffs?", "why does/would", "edge cases", "failure modes", "underlying", etc.
  - Result: "beginner" / "intermediate" / "advanced" → adjusts language guidance in system prompt.
- Socratic mixing (soft hint to LLM, never mandatory):
  - Trigger: every 3rd turn, only when question is not a short/factual lookup.
  - Adds "you may end with one thought-provoking question" to system prompt.
  - Skipped for "what is X?"-style questions and questions under 5 words.
- Feynman suggestion (soft hint to LLM, never mandatory):
  - Trigger: after 5+ turns in the session.
  - Adds "you may suggest the user explain a key concept in their own words" to system prompt.
  - "May" and "optionally" — LLM decides based on actual context.

18 new tests in `tests/agents/test_qa_agent.py`. 232 tests pass total (234 after user KG edges).

**Sidebar strategy selector - complete**

- `StrategyRouter.chat()` extended with `force_strategy: str | None = None` param.
  `_resolve_strategy()` bypasses keyword routing when force is set.
  Accepted values: "react", "plan-execute", "reflexion". None = auto-route.
- `QAAgent.chat()` accepts `force_strategy` and passes it to the router.
- Sidebar `st.selectbox` in `_render_sidebar()` with options Auto / ReAct / Plan-Execute / Reflexion.
  Stored in `session_state.strategy_mode`. Passed to `qa_agent.chat()` on each question.

**Source display - complete**

- `_render_sources(sources)` renders an `st.expander("Sources (N)")` with chapter and page range
  for each unique search result (deduplicates by (chapter, page_range) key).
- `search_results` stored in each assistant message dict alongside trace and KG suggestions.
- Rendered in both the live response and the conversation replay loop, so sources appear on
  page reload.

**Knowledge Graph visual page - complete**

Per `plans/kg_visual_page.md`:

- `bookmind_tutor/knowledge_graph/graph_store.py`: added `get_user_graph_data()` — bulk
  query returning all UserEntity nodes + relations for visualization.
- `bookmind_tutor/tutor/pages/1_Knowledge_Graph.py`: new Streamlit multi-page page.
  Streamlit auto-discovers the `pages/` directory and adds it to the sidebar.
- Visualization: `streamlit-agraph` (vis.js) — force-directed interactive graph.
  Supports drag, zoom, click-to-select.
- Node color coding: CONCEPT=blue, SYSTEM=green, PRINCIPLE=orange, DEFINITION=purple, PERSON=red.
- Full graph mode: shows all UserEntity nodes. Text search dims non-matching nodes to grey.
  Type filter multiselect hides/shows nodes by type.
- Neighborhood mode: selectbox for center concept + depth radio (1/2) → subgraph of that
  node and its reachable neighbors. Center node highlighted orange.
- Node detail panel (right column): click a node → shows name, type (color-coded),
  description, connected nodes list.
- Legend: only shows entity types that are present in the current graph.
- `requirements.txt`: added `streamlit-agraph>=0.0.45`.
**User KG edges via lazy book KG - complete**

Per `plans/user_kg_edges.md`:

- `bookmind_tutor/knowledge_graph/suggester.py`:
  - Cache đổi sang `tuple[list[Entity], list[Relation]]` để giữ cả relations.
  - `_extract_cached()` trả về `(entities, relations)`.
  - `suggest()`: sau mỗi chunk extraction, upsert cả entities + relations vào book KG
    trước khi filter suggestions cho user. Book KG được xây lazily theo dấu chân user.
- `bookmind_tutor/knowledge_graph/graph_store.py`:
  - `get_user_graph_data()`: edges được derive từ book KG — query Entity relations
    giữa các user-known names. Edges xuất hiện tự động khi book KG có đủ dữ liệu.
  - `get_user_related_entities()`: traversal qua Entity graph (typed relations),
    filter về UserEntity names.
- 2 tests mới: `test_upserts_entities_to_book_kg`, `test_upserts_relations_to_book_kg`.

**Book introduction message on upload - complete**

Per `plans/book_intro_message.md`:

- `_generate_book_intro(vs, book_name, llm_client)` in `app.py`: samples up to 8 chunks
  via 3 broad probes ("introduction overview", "key concepts principles", "conclusion summary"),
  calls `claude-haiku-4-5-20251001` to synthesize a welcome intro (themes, structure, audience,
  open-ended invitation). One LLM call per new book upload.
- Upload flow in `_upload_page`: for freshly indexed books (not re-index), generates intro
  and stores it as the first assistant message in `book_histories[bid]`.
- Chat page renders it naturally via existing `_current_messages()` loop — no UI changes needed.
- Bug fix: `_book_id()` now strips trailing underscores after slug truncation to satisfy
  ChromaDB's `[a-zA-Z0-9]` boundary constraint.

**Streaming responses - complete**

Added `stream_chat()` generator method to the full agent stack:
- `AgentHarness.stream_chat()`: uses `client.messages.stream()` on every ReAct loop iteration.
  Tool-use steps yield no tokens (model generates no text). The end_turn step yields tokens
  in real-time as they arrive from the API.
- `ReActStrategy.stream_chat()`: delegates to harness, sets `last_trace` after generator exhaustion.
- `PlanExecuteStrategy.stream_chat()`: runs plan + parallel execute blocking, streams synthesis step.
- `ReflexionStrategy.stream_chat()`: runs initial answer + reflection blocking, streams retry (if needed)
  or yields initial answer in one go (if no retry needed).
- `StrategyRouter.stream_chat()`: routes and delegates, sets `last_trace` when exhausted.
- `QAAgent.stream_chat()`: sets system prompt same as `chat()`, delegates to router.
- `app.py`: `_handle_question` uses `st.write_stream()` - renders tokens as they arrive, returns
  full concatenated string for history storage. KG suggestions run after streaming completes.

234 tests pass total.

**Bug fix: agent asking "which book?" - complete**

- `QAAgent.__init__` now accepts `book_name: str = ""`.
- `_build_system_prompt()` prepends `The book currently loaded is: "{book_name}".` to the base system
  prompt so Claude knows the exact book title on every turn — even when the user asks in Vietnamese
  using generic terms like "cuốn sách".
- `_setup_agent()` in `app.py` passes `book_name` from session state when constructing `QAAgent`.

**In-UI observability trace - complete**

Added per-step tool call logging to `AgentHarness`, surfaced in the existing `ReasoningTrace.steps`:

- `AgentHarness._last_steps: list[str]` — reset at the start of each `chat()` / `stream_chat()` call.
  Records each tool call with its query (`[N] book_search('...')`), each tool result size
  (`[N]  → 11402 chars`), and the final end_turn token counts (`[N] end_turn — X out / Y in tokens`).
- `last_steps: list[str]` property exposed on `AgentHarness`.
- `ReActStrategy.chat()` and `stream_chat()` now use `harness.last_steps` instead of a placeholder string.
- `_render_trace()` in `app.py` upgraded: summary line becomes an expander title; clicking it reveals
  all steps as captions. Answers with no steps fall back to a plain caption.

**Graceful API error handling - complete**

- `_api_error_message(exc)` helper in `app.py` maps `anthropic.APIStatusError` subclasses to
  actionable messages: credit balance low (with billing URL), invalid API key, rate limit, connection error.
- Upload flow: outer `except anthropic.APIStatusError` shows `st.error` and aborts; inner catch
  around `_generate_book_intro` shows `st.warning` and lets the user continue without an intro.
- Chat flow: `st.write_stream()` wrapped in `try/except`; on error shows `st.error` in the assistant
  bubble and saves an error placeholder to history so the turn is not lost.
- KG suggestion extraction: silent fallback to empty suggestions on API error — non-critical path.

**Session persistence: book library + chat history - complete**

Per `plans/session_persistence.md` (updated to include delete UI):

- `bookmind_tutor/tutor/persistence.py`: new `LibraryStore` class — atomic JSON writes
  (`os.replace` via .tmp sibling), fault-tolerant reads (missing/corrupt → empty, never raises),
  schema version field for future migrations. `BookRecord` dataclass holds name, n_chunks, indexed_at.
- `bookmind_tutor/retrieval/vector_store.py`: added `drop()` method — deletes ChromaDB collection
  entirely (vs `clear()` which recreates it empty). Used by delete-book flow.
- `bookmind_tutor/tutor/qa_agent.py`: `seed_memory(messages)` — replays persisted UI messages
  into `ConversationMemory` on agent construction. Skips leading assistant messages (Anthropic API
  requires first message to be user). Restores `_turn_count` so Socratic/Feynman triggers resume
  at the correct cadence after reload.
- `bookmind_tutor/tutor/app.py` changes:
  - `VectorStore` now created with `persist_dir="data/chroma"` → embeddings survive server restart.
  - `_restore_from_disk()`: on first session, loads `library.json`, reconnects each book to its
    ChromaDB collection, loads history. Stale books (chroma data missing) shown as one-shot warning.
  - `_save_library()` / `_delete_book()` helpers. `book_records: dict[str, BookRecord]` added to
    session state as persistence source of truth.
  - `_setup_agent()` calls `qa_agent.seed_memory(_current_messages())` after construction.
  - History saved after every assistant turn (including error turns) and after intro generation.
  - Delete UI in sidebar: trash icon per book → inline confirmation (warning + Delete/Cancel buttons).
    Deletes ChromaDB collection, history file, registry entry, and session state atomically.
- `.gitignore`: added `data/chroma/`, `data/history/`, `data/library.json`.
- `tests/tutor/test_persistence.py`: 16 tests covering library/history round-trips, corrupt/missing
  file handling, field stripping/restoration, atomic write, version mismatch.

249 tests pass total (one test removed: `test_skips_non_tool_use_blocks` — no longer applicable since executor takes typed `ToolCallRequest` list, not raw content blocks).

**LLM provider abstraction - complete**

Per `plans/llm_provider_abstraction.md`:

- `bookmind_tutor/llm/` new package:
  - `base.py`: `LLMClient` abstract class, `CompletionResponse`, `ToolCallRequest`, `TokenUsage`, `LLMAPIError` dataclasses.
    All messages flow in OpenAI format (canonical internal format).
  - `anthropic_client.py`: `AnthropicClient` — converts OpenAI format to/from Anthropic wire format.
    Groups consecutive `role="tool"` messages into Anthropic `tool_result` user blocks.
    Maps Anthropic error subclasses to `LLMAPIError` with typed codes.
  - `openai_client.py`: `OpenAIClient` — near-passthrough (OpenAI format is already canonical).
    Accumulates streamed tool-call argument fragments across chunks.
    Maps OpenAI error codes to `LLMAPIError`.
  - `__init__.py`: `create_client(provider, model, api_key)` factory. Lazy imports so only the active provider's SDK is loaded.

- Changed to use the abstraction:
  - `agents/models.py`: `ToolSpec.to_dict()` returns OpenAI function-tool format (replaces `to_anthropic_dict()`). `ToolCallResult.to_anthropic_dict()` removed.
  - `agents/executor.py`: `execute_all()` now takes `list[ToolCallRequest]` instead of raw content blocks.
  - `agents/memory.py`: stores messages in OpenAI format. `add_user()` stores plain string. `add_assistant(raw_message: dict)` takes OpenAI format dict. `add_tool_results()` stores `role="tool"` messages.
  - `agents/harness.py`: uses `LLMClient.complete()` / `.stream()`. Reads `CompletionResponse`. Passes `response.tool_calls` to executor.
  - `agents/strategies/plan_execute.py`, `reflexion.py`: use `client.complete()` / `client.stream()`.
  - `knowledge_graph/extractor.py`, `ingestion/structure_detector.py`: use `client.complete()`.
  - `tutor/qa_agent.py`: `seed_memory()` uses `add_assistant({"role": "assistant", "content": ...})`.
  - `tutor/app.py`: `create_client()` at startup, reads `LLM_PROVIDER` + `LLM_MODEL` from `.env`. Catches `LLMAPIError` instead of `anthropic.APIStatusError`.

- Config: `LLM_PROVIDER=openai|anthropic`, `LLM_MODEL=gpt-4o-mini|claude-haiku-4-5-20251001` in `.env`. `.env.example` updated.
- `requirements.txt`: added `openai>=1.0.0`.
- Tests: updated 5 test files to mock `LLMClient.complete()` instead of `anthropic.Anthropic`. 249 tests pass.

---

## Week 7 - Evaluation Framework

Per `plans/week7_eval_framework.md`:

**`ReasoningTrace.retrieved_chunks` - complete**

- `agents/models.py`: added `retrieved_chunks: list[str] = field(default_factory=list)` to `ReasoningTrace`.
  Accumulates raw text returned by `book_search` tool calls across all ReAct steps.
  Used by eval framework (scoring factual grounding) and will feed DPO data in Week 10-11.
- `agents/harness.py`: added `_last_retrieved_chunks: list[str]`, reset at the start of each `chat()` / `stream_chat()` call.
  In the `tool_use` branch, zips `response.tool_calls` with `results`; appends `r.content` for any `tc.name == "book_search"` call.
  `last_retrieved_chunks: list[str]` property exposed.
- `agents/strategies/react.py`: `ReActStrategy.chat()` and `stream_chat()` populate `retrieved_chunks=self._harness.last_retrieved_chunks` in the trace.

**`bookmind_tutor/evaluation/` package - complete**

- `logger.py`: `InteractionRecord` dataclass (15 fields: record_id, timestamp, book context, question, retrieved_chunks, answer, strategy/model/provider, cost metrics, optional eval/human scores).
  `InteractionLogger` appends one JSONL line per turn; `load_all()` returns latest record per `record_id` (append-only update pattern), skips malformed lines.
- `evaluator.py`: `AnswerEvaluator` with LLM-as-judge on 3 dimensions (factual_accuracy, citation_quality, depth_and_relevance, each 0.0-1.0).
  Retries once with stricter prompt on JSON parse failure.
  Returns sentinel `EvalResult` (all scores = -1.0) on double failure.
  `EvalResult.overall` = simple mean of the three dimensions.
- `runner.py`: `EvalRunner` benchmarks a list of `RunConfig` (provider, model, strategy, label) against benchmark questions.
  For each (question, config): builds a fresh `QAAgent`, calls `.chat()`, logs via `InteractionLogger`, scores via `AnswerEvaluator`, collects into `EvalReport`.
  `EvalReport.summary_by_config()` returns mean scores per label. `to_markdown_table()` and `to_csv()` output methods.
  Report saved to `data/eval_reports/<timestamp>.json`.
- `__main__.py`: CLI entry point — `python -m bookmind_tutor.evaluation --book-id <id> --questions data/eval_questions.json`.

**Integration in `app.py` - complete**

- `InteractionLogger` initialized once in `_init_state()` at `data/interactions.jsonl`.
- After each successful streaming Q&A turn, logs an `InteractionRecord` (non-blocking; bare `except` so UI never crashes on log failure).
- Bug fix: `except anthropic.APIStatusError` in KG suggestion path replaced with `except LLMAPIError` (was broken after provider abstraction).

**Tests - complete**

- `tests/evaluation/test_logger.py`: 10 tests (roundtrip, dedup by record_id, malformed lines, empty file, missing file, parent dir creation, eval field persistence).
- `tests/evaluation/test_evaluator.py`: 9 tests (valid JSON, markdown fences, overall = mean, retry logic, double failure sentinel, API exception, no chunks, model_used field).
- `tests/evaluation/test_runner.py`: 10 tests (rows per question, multi-config cross product, logger called, eval scores in log, report JSON saved, chunks passed to evaluator).

**Other**

- `.gitignore`: added `data/interactions.jsonl`, `data/eval_reports/`.
- `data/eval_questions.example.json`: starter template for writing benchmark questions per book.

278 tests pass total.
