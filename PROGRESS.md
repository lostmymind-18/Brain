# PROGRESS

Each entry here must reference either a plan in `plans/` or a log in `backlog.md`.

---

## Search layer subsection support + contextual embedding - complete

Follow-up to the structure reconciler: the search layer did not exploit the new
subsection metadata (ref: backlog.md "Search layer does not exploit new subsection metadata").

- **Contextual embedding** (`retrieval/vector_store.py`): chunks are now EMBEDDED
  (and BM25-tokenized) with their heading breadcrumb prepended
  ("Chapter > Section > Subsection\n<text>"), while the STORED document text stays
  clean for display. `index_chunks()` computes embeddings explicitly from the context
  text; `_get_bm25()` tokenizes context text so heading words are searchable even when
  absent from the chunk body. Lightweight version of Anthropic's contextual retrieval -
  the context comes free from structure detection instead of an LLM call.
- **Chunk size** (`ingestion/hierarchical_chunker.py`): default `max_tokens` 180 -> 160
  to leave headroom for the breadcrumb under the ~195-word embedding truncation point.
- **`book_outline` 3 levels** (`agents/tools/book_outline.py`): outline now includes
  subsections, so the LLM can discover exact heading names for `get_book_section`
  (previously 113 FoSA subsection names were matchable but undiscoverable).
- **Location labels** (`book_search.py`, `graph_rag_search.py`): search excerpts now
  labeled "Chapter / Section / Subsection, p.X" so answers can cite at subsection precision.
- **`SourceRef.subsection`** (`agents/tools/base.py`, `tutor/app.py`): Sources panel
  shows "Chapter › Subsection" - a page-precise heading the user can look up in the book.
- **12 new tests** (contextual embedding, breadcrumb format, clean stored text,
  book_outline nesting, subsection labels). 386 tests passing total.
- **Re-indexed** both books (FoSA 1022 chunks, DDIA 2112).
- **Measured** (FoSA spot-check, k=3): heading-name queries all hit labeled chunks in
  top-3 (Broker Topology 0.811, Risk Storming 0.850, ADR 0.857); "fitness functions"
  query now retrieves the real content (p.103) instead of the appendix quiz page (p.395).
  Avg on the 5 mixed queries: 0.760 vs 0.750 original - raw cosine is not the story;
  result relevance is.
- **E2E verified** (Playwright, gpt-4o-mini): "Mediator topology hoạt động như thế nào?"
  -> grounded answer citing Chapter 14 p.205, fact-check 5/5 verified, Sources shows
  "Part II. Architecture Styles › Mediator Topology · pp. 205-215". Zero console errors.

---

## Hybrid Structure Reconciler + Chunk Size Fix + Subsection Metadata - complete

Three interlinked ingestion improvements (ref: `plans/structure_reconciler.md`,
backlog entries "Chunk size exceeds embedding model input limit",
"HierarchicalChunker discards detected L2 subsection titles",
"Structure reconciler: TOC anchors + heading candidates").

- **Chunk size fix** (`ingestion/hierarchical_chunker.py`): default `max_tokens` lowered
  512 → 180 words, `overlap_tokens` 50 → 30. all-MiniLM-L6-v2 truncates at 256 BPE tokens;
  empirically 52.8% of FoSA chunks had un-embedded tails at the old default. New default
  keeps all chunks under the truncation limit. Existing books must be re-indexed.

- **`Chunk.subsection` field** (`ingestion/models.py`): new optional field (default None)
  carries the L2 heading title. `HierarchicalChunker._chunk_node()` now propagates the
  `subsection` parameter at level 2; level 3+ inherits. `VectorStore.index_chunks()` indexes
  it as ChromaDB metadata. `GetBookSectionTool` searches it alongside `chapter` and `section`,
  and displays it in chunk location labels. `SearchResult` also carries the field.

- **`normalize_heading()`** (`ingestion/utils.py`): shared normalization (lowercase, strip
  punctuation, collapse whitespace). `GetBookSectionTool._normalize()` now imports this
  instead of defining its own copy.

- **`StructureReconciler`** (new `ingestion/structure_reconciler.py`):
  - Input: embedded TOC + `HeadingCandidate` list from Tier 3 font-size heuristic + anchor tree.
  - Filters false positives: no-letter candidates, rare fonts (<min_heading_pages distinct pages),
    candidates before the first TOC anchor page.
  - Excludes TOC-level headings (normalized match against TOC title set).
  - Injects survivors as child `DocumentNode`s under their anchor, redistributing `raw_text`
    by splitting at the normalized heading line.
  - Local font-size ranking per anchor to determine sub-level (no global clustering).
  - Monotonic: returns anchor_tree unchanged when no candidates survive.

- **`StructureDetector.detect()` (Tier 1 path extended)** (`ingestion/structure_detector.py`):
  when an embedded TOC is present, now ALSO runs Tier 3 heuristic classification
  via new `_extract_heading_candidates()` method, then passes both to `StructureReconciler`.
  The Tier 2/Tier 3 fallback paths are unchanged.

- **15 new tests** (`tests/ingestion/test_structure_reconciler.py`). 374 tests passing total.
  Covers: false-positive filter (symbol, single-char, rare font), TOC match exclusion
  (exact + punctuation variant), TOC anchor preservation, sub-heading injection,
  level assignment, text redistribution, chunk subsection propagation.

Next: re-index existing books to apply new chunk sizes and subsection metadata.

---

## GetBookSectionTool v2: single-name interface + honest Sources panel - complete

Root-cause fix after E2E testing exposed systematic failures with gpt-4o-mini
(ref: backlog.md "get_book_section: field/contains interface leaks metadata schema to the LLM"
and "Sources panel disconnected from agent retrieval").

- **Tool interface redesigned** (`agents/tools/get_book_section.py`): single `section_name`
  parameter replaces `field` + `contains`. The tool matches against distinct heading values
  from BOTH metadata fields (a chapter outside any part lands at part level, as Chapter 1 does
  in "Fundamentals of Software Architecture"). Matching is case- and punctuation-insensitive,
  so 'Chapter 1. Introduction', 'Chapter 1: Introduction', and 'Chapter\xa01: Introduction'
  group as one heading. Ambiguous names return a disambiguation list (name, chunk count,
  page range) instead of dumping wrong content; unknown names return the available headings.
- **Honest Sources panel** (`agents/tools/base.py`, `graph_rag_search.py`, `tutor/app.py`):
  new `SourceRef` dataclass; retrieval tools record what they retrieved in `Tool.last_sources`.
  The chat handler clears them per turn and renders them after. Removed the independent
  semantic search that previously populated Sources with chunks the agent never read.
- **23 tests** in `tests/agents/test_get_book_section.py`. 353 tests passing total.
- **E2E verified** ("Chuong 1 cua cuon sach noi ve chu de gi?", gpt-4o-mini):
  1 tool call, 44k chars of real Chapter 1 (pages 21-40), 2 LLM calls / 23k tokens
  (down from 7 calls / 63k tokens), Sources shows "Chapter 1. Introduction pp. 21-393",
  fact-check 4/5 claims verified with cross-language evidence quotes.

---

## GetBookSectionTool - complete

Structured retrieval by metadata filter to support chapter/part summarization
(ad-hoc from conversation, no prior plan).

- **New tool** (`agents/tools/get_book_section.py`): filter ChromaDB chunks by `chapter`
  or `section` metadata field using case-insensitive substring match. Optional `page_from`/`page_to`
  for page-range queries. Returns all matching chunks sorted by `page_start`, capped at 30 to
  stay within context limits (with truncation note).
- **Registered** in `agents/tools/__init__.py`; wired into `tutor/app.py` and `evaluation/runner.py`
  alongside `book_outline` and `book_search`.
- **14 new tests** (`tests/agents/test_get_book_section.py`). 345 tests passing total.
- **E2E verified**: "tóm tắt nội dung chương 5" → LLM called `get_book_section` → 27873 chars
  (full chapter) vs 5 chunks from `book_search`. Answer had 8 structured points covering all
  major themes of Chapter 5.

Design insight: "Part I" substring matches "Part II" — LLM should pass the full name copied
from `book_outline` (e.g. "Part I.") for unambiguous filtering.

---

## Hallucination Defense - complete

Systematic 4-layer defense against hallucination (ref: `plans/greedy-enchanting-yeti.md`).
All 331 tests pass. E2E verified with Playwright.

- **Layer 1 - Hybrid search** (`retrieval/vector_store.py`): BM25 + dense vector fusion
  with `rank-bm25`. Score = 0.5 * cosine_sim + 0.5 * BM25_normalized. BM25 index built
  lazily on first search from ChromaDB, cached until re-index. Exact-term and proper-noun
  queries (e.g. "Chapter 14 Event-Driven Architecture") now rank the right chunk first.
- **Layer 2 - Grounded generation** (`agents/harness.py`): Strengthened `_DEFAULT_SYSTEM`
  with explicit no-parametric-memory rules: "never assert facts not in retrieved excerpts",
  "say 'I could not find this' rather than guessing".
- **Layer 3 - Citation verifier** (`agents/verification.py`, `tutor/app.py`): Post-answer
  LLM call extracts up to 5 claims from the answer and checks each against retrieved chunks.
  Renders "Fact-check: N verified, M unverified" expander - auto-expanded when unverified
  claims exist. Stored serialized in message dict so it persists across `st.rerun()`.
- **Layer 4 - Hallucination metric** (`evaluation/evaluator.py`, `evaluation/runner.py`):
  Added `hallucination_score` (0.0=grounded, 1.0=hallucinated) to `_EVAL_PROMPT`,
  `EvalResult`, `EvalRow`, `EvalReport.summary_by_config()`, markdown table, and CSV.
- **Bonus bug fix** (`agents/harness.py`): `_last_retrieved_chunks` now captures both
  `book_search` and `graph_rag_search` tool results (was only capturing `book_search`).
- **KG extraction temporarily disabled** (`tutor/app.py`): `suggester.suggest()` skipped
  per user request to reduce latency. Spinner renamed "Verifying citations...".

---

## Post Week-9 fixes + BookOutlineTool - complete

Fixed hallucination on "list all chapters" queries. Root cause: TOC chunk embeddings
represent chapter topics, not the concept "table of contents", so the TOC ranked #37
of 317 in retrieval. Two fixes implemented (ref: `backlog.md`):

- **`BookOutlineTool`** (`agents/tools/book_outline.py`): New deterministic tool that
  derives the part/chapter hierarchy from ChromaDB metadata (chapter + section fields
  set at ingest time). No embedding lookup - 100% accurate, matches `doc.get_toc()`.
  Registered alongside `GraphRAGSearchTool` in `_setup_agent()`.
- **Relevance threshold** in `BookSearchTool` and `GraphRAGSearchTool`: When best
  cosine similarity < 0.25, prepend a warning to results so the model degrades
  honestly ("do not invent details") instead of hallucinating.
- **English query enforcement**: Tool descriptions now instruct the model to always
  search in English (Vietnamese queries against English embeddings gave strictly worse
  retrieval).

---

## Post Week-9 bug fixes - complete

Discovered and fixed 7 bugs via code review + E2E testing.
All 321 tests pass. Changes tracked in `backlog.md`.

**Local LLM support bug fixes (`llm/`, `tutor/`, `evaluation/`):**

- `tutor/app.py`: Guard `base_url` in kwargs with `_LLM_PROVIDER == "openai"` check.
  Prevented `TypeError` crash when `LLM_BASE_URL` is set while using the Anthropic provider.
- `llm/__init__.py`: Strip unknown kwargs from `AnthropicClient` call (it only accepts `model`
  and `api_key`). Auto-set `stream_usage=False` in factory when `base_url` is present for OpenAI.
- `llm/openai_client.py`: Replace `base_url is None` heuristic with explicit `stream_usage: bool = True`
  constructor param. Remove errant `continue` after usage update in stream loop (prevented silent
  content drops when provider sends usage + content in same chunk).
- `evaluation/runner.py` + `evaluation/__main__.py`: Add `RunConfig.base_url` and `--base-url`
  CLI arg (defaults to `LLM_BASE_URL` env var) so eval runner respects the same provider config as
  the Streamlit app.

**structlog fix (`observability/setup.py`):**

- Removed `structlog.stdlib.add_logger_name` from shared processors.
  This processor requires `logging.Logger.name` which is unavailable on `structlog.PrintLogger`.
  Bug surfaced during E2E test - first real OpenAI-backed chat request crashed the harness.

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

---

## Week 8-9 - Safety/Guardrails + Observability

Per `plans/safety_guardrails.md` and `plans/observability.md`.

**Safety/Guardrails - complete**

`bookmind_tutor/safety/` package:

- `models.py`: `Severity` (BLOCK/WARN/LOG), `ViolationType` (INPUT_TOO_LONG, PROMPT_INJECTION, OFF_TOPIC, OUTPUT_TOO_LONG), `GuardContext` (book_name, book_id, session_id), `GuardResult` (passed, violation_type, severity, user_message, internal_detail).
- `input_guards.py`:
  - `LengthGuard(max_chars=2000)`: BLOCK if input exceeds character limit. Returns user-facing count in the message.
  - `PromptInjectionGuard`: regex patterns across 4 attack categories - ROLE_OVERRIDE ("ignore all instructions", "act as", "pretend to be"), SYSTEM_LEAK ("repeat your system prompt"), JAILBREAK ("DAN mode", "developer mode"), DELIMITER_INJ (`</system>`, `<|im_start|>`, `[INST]`). BLOCK severity. Logs matched category for monitoring.
  - `TopicalityGuard(llm_client, enabled=True)`: single cheap LLM call (`max_tokens=5`) asking yes/no whether the question relates to the loaded book. WARN severity (never BLOCK). Fails OPEN on any LLM error - never punishes user for guard infra failure. Input truncated to 500 chars for cost control.
- `output_guards.py`: `OutputLengthGuard(max_chars=8000)`: WARN if response is unusually long (data-collection only - never blocks the answer).
- `layer.py`: `GuardrailsLayer(input_guards, output_guards)` - runs guards in registration order. First BLOCK short-circuits (remaining guards skipped). WARN results are logged by the guard itself; layer continues. Returns `GuardResult(passed=True)` if all pass.
- `__init__.py`: public re-exports for all types.

Integration in `app.py`:
- `GuardrailsLayer` initialized once in `_init_state()` with LengthGuard + PromptInjectionGuard + TopicalityGuard (input), OutputLengthGuard (output).
- `_handle_question()`: input check runs before appending to history or calling LLM. BLOCK: renders user bubble + error in assistant bubble, then returns without saving to history. WARN: proceeds normally (logged only). Output check runs after streaming completes (WARN-only, side-effect logging).

Tests:
- `tests/safety/test_input_guards.py`: 21 tests (LengthGuard boundary cases, all 4 injection categories, TopicalityGuard mocked LLM - related/off-topic/disabled/no-book-name/fail-open/truncation).
- `tests/safety/test_output_guards.py`: 5 tests (boundary conditions, WARN severity, internal_detail content).
- `tests/safety/test_layer.py`: 8 tests (BLOCK short-circuits, WARN continues, context forwarded, no guards, output WARN passes through).

**Observability (OTel + structlog) - complete**

`bookmind_tutor/observability/` package:

- `setup.py`:
  - `configure_structlog(level, json_output)`: idempotent. Shared processors: contextvars merge, log level, logger name, ISO timestamp, stack info. Pretty output (ConsoleRenderer with colors) in dev; JSON lines in production (`LOG_FORMAT=json`). `LOG_LEVEL` env var respected.
  - `configure_observability(service_name, export_to, otlp_endpoint)`: idempotent. Builds a `TracerProvider` with `service.name` resource attribute. `OTEL_EXPORT=console` (default) uses ConsoleSpanExporter. `OTEL_EXPORT=otlp` uses gRPC OTLP exporter (`opentelemetry-exporter-otlp-proto-grpc`) toward `OTEL_EXPORTER_OTLP_ENDPOINT` (default: localhost:4317). Falls back to console if OTLP package missing.
  - `reset_for_testing()`: resets both idempotency flags AND OTel global state (`_TRACER_PROVIDER`, `_TRACER_PROVIDER_SET_ONCE._done`) so tests can install their own `InMemorySpanExporter`.
- `tracing.py`:
  - `get_tracer()`: returns global OTel tracer. No-op ProxyTracer if `configure_observability()` was never called - all span calls are safe no-ops in tests.
  - `GenAIAttrs`: GenAI Semantic Convention constants (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.stop_reason`, `gen_ai.operation.name`).
  - `AgentAttrs`: `agent.strategy`, `agent.step_num`, `agent.steps_taken`, `agent.total_llm_calls`.
  - `ToolAttrs`: `tool.name`, `tool.input_summary`, `tool.result_chars`, `tool.is_error`.
  - `QAAttrs`: `qa.book_id`, `qa.turn_count`, `qa.strategy_selected`.
  - `set_error(span, exc)`: records exception + sets ERROR status code.

LLM client instrumentation (Template Method pattern):
- `llm/base.py`: added `provider` abstract property. Renamed `complete` → `_complete`, `stream` → `_stream`. New concrete `complete()` wraps `_complete()` with `gen_ai.complete` span (sets system, model, operation, input/output tokens, stop reason; calls `set_error` on exception). New concrete `stream()` wraps `_stream()` with `gen_ai.stream` span via `yield from` inside the `with` block - context stays open across yields; post-stream attributes set after `yield from` completes using `self.last_response`.
- `llm/anthropic_client.py`: added `provider = "anthropic"`, renamed methods to `_complete`/`_stream`.
- `llm/openai_client.py`: added `provider = "openai"`, renamed methods to `_complete`/`_stream`. Replaced `import logging` with `structlog`.

Agent instrumentation:
- `agents/harness.py`: replaced `import logging` with `structlog`. All log calls use keyword-argument style (`logger.info("event", key=val)`). `chat()` wrapped in `agent.react_turn` span (`agent.strategy=react`). Each iteration wrapped in `agent.react_step` span (`agent.step_num=N`). `agent.total_llm_calls` set on turn span at end.
- `agents/executor.py`: replaced `import logging` with `structlog`. Each `_execute_one()` call wrapped in `tool.execute` span with `tool.name`, `tool.input_summary`, `tool.result_chars`, `tool.is_error`.
- `tutor/qa_agent.py`: both `chat()` and `stream_chat()` wrapped in `qa.turn` span with `qa.turn_count` and `qa.strategy_selected` (set after router returns).

Startup in `app.py`:
- `configure_structlog()` and `configure_observability()` called at module level (before `import streamlit as st`) so all modules pick up the configured providers.

Tests:
- `tests/observability/test_tracing.py`: 9 tests using `InMemorySpanExporter` via `isolated_tracer` fixture (resets OTel state per test). Covers tracer creation, span capture, `set_error` exception recording, all 4 attribute constant classes, attribute value setting, nested span parent-child linking.

Documentation:
- `docs/local_observability.md`: Jaeger Docker setup, env var configuration, span hierarchy diagram, useful attributes table, Grafana LGTM stack instructions for production.

`requirements.txt` additions: `structlog>=24.0.0`, `opentelemetry-api>=1.20.0`, `opentelemetry-sdk>=1.20.0`, `opentelemetry-exporter-otlp-proto-grpc>=1.20.0`.

321 tests pass total.
