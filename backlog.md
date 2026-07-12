# Backlog

## Week 1 - Ingestion Pipeline

### ~~Cover page dominates heading level hierarchy~~ (fixed)
**File:** `bookmind_tutor/ingestion/structure_detector.py` - `_build_heading_level_map()`

Fixed by requiring each heading font size to appear on `min_heading_pages` (default 3) distinct pages before it enters the level map. Adaptive threshold: lowered for short documents so synthetic test data still works. Cover page fonts (appear on 1 page) are now excluded. DDIA now yields 4 distinct chapter values ("Foundations of Data Systems", "Distributed Data", "Derived Data", None).

---

### ~~Parent node page_range not propagated through ancestors~~ (fixed)
**File:** `bookmind_tutor/ingestion/structure_detector.py` - `_build_tree()`

Fixed by introducing `_extend_page_range(stack, page_number)` which updates ALL open ancestors in the stack, not just `stack[-1]`. Verified: DDIA preamble node correctly shows `page_range=(1, 613)`.

---

### ~~Running headers/footers pollute chunk text~~ (fixed)
**File:** `bookmind_tutor/ingestion/pdf_extractor.py`

Fixed by two-phase extraction: collect raw page dicts → identify repeating text patterns in top/bottom margin bands → strip those lines when building PageContent. Page-number normalization (`_normalize_hf`) ensures "300 | Chapter 7" and "301 | Chapter 7" are recognized as the same repeating pattern.

---

### ~~Unit tests too clean to catch real-book failures~~ (addressed)
**File:** `tests/ingestion/`

Added `scripts/sanity_check.py` (see below) for manual integration check against real PDFs. Unit tests retained for fast CI feedback.

---

### char_offset_start/end relative to node raw_text, not PDF page text
**File:** `bookmind_tutor/ingestion/hierarchical_chunker.py`, `models.py`

`Chunk.char_offset_start` and `char_offset_end` currently point into `DocumentNode.raw_text`, not the original PDF page text.
For Week 4 Knowledge Graph work, it would be more useful to have offsets relative to the page text so entities can be located precisely in the source document.

Approach: track cumulative character offsets per page during `StructureDetector._build_tree()` and carry them through to `Chunk`.

---

### Front matter mixed into first chapter node
**File:** `bookmind_tutor/ingestion/structure_detector.py`

Pages before the first real chapter (TOC, copyright, preface) get absorbed into the "Architecture" cover node.
This means chunks from the TOC and copyright page carry the same chapter label as body content.

Approach: detect front matter heuristically (pages before the first "CHAPTER N" or "Part N" heading) and tag them separately, or simply exclude them from chunking.

---

### ~~Signal 2 matching figure captions and sidebar labels~~ (fixed)
**File:** `bookmind_tutor/ingestion/structure_detector.py` - `_classify_line()`

Spans slightly above body size (e.g. figure captions at 1.1x body) were being pattern-matched by Signal 2.
Example: `"GROUP BY"` at 11.6pt in a 10.5pt body doc matched `_RE_ALL_CAPS` and became a spurious root chapter spanning 186 pages.

Fixed by requiring Signal 2 to use the same `min_heading_size_ratio` threshold as Signal 1, instead of just `> body_size`.

---

### Tier 2: front matter page offset computed incorrectly for dual numbering

**File:** `bookmind_tutor/ingestion/structure_detector.py` - `_detect_page_offset()`

Books with roman numerals for front matter (Preface = xiii, etc.) and arabic from Chapter 1 have two discontinuous offsets. The current `_detect_page_offset` returns the mode offset (22 for DDIA main content), but front matter entries like Preface end up at wrong PDF pages (p35 instead of p15).

Approach: detect the discontinuity - entries whose `toc_page` is small and whose title matches a page in the 1-20 range should use their own offset. Or: treat roman-numeral pages (small values <= 30) separately from arabic chapter pages.

Not urgent: Tier 2 is only triggered when there is no embedded TOC. Most published books have one.

---

### Tier 2: only first TOC page sent to LLM

**File:** `bookmind_tutor/ingestion/structure_detector.py` - `_find_toc_page()`, `_parse_toc_page_with_llm()`

`_find_toc_page` returns a single page. For DDIA, the TOC spans pages 9-14. Only page 9 is sent to the LLM, so Chapter 2+ have incomplete subsection structure from Tier 2.

Approach: scan and concatenate consecutive TOC pages (up to some limit) before calling the LLM. Heuristic to detect end of TOC: page that no longer starts with right-aligned numbers or dot leaders.

Not urgent: same caveat as above.

---

### Wrapper TOC node ("Contents") swallows the whole book hierarchy

**File:** `bookmind_tutor/ingestion/structure_detector.py` - `_build_tree_from_toc()`

Found 2026-07-11 while investigating a verbatim-quote failure with "The Beginning of Infinity".
This book's embedded TOC nests every chapter under a level-1 "Contents" entry
(top level: Cover Page, Title Page, Contents, Copyright Page).
Result: `chapter = "Contents"` for 1749/1758 chunks, real chapter titles land in `section`,
and the whole hierarchy is shifted one level down.

Consequences:
- The contextual-embedding breadcrumb wastes budget on the meaningless word "Contents" in every chunk.
- `book_outline` renders one giant chapter, hiding the real structure.
- Source labels read `Contents / 4: Creation / ...`, which confuses both the LLM and the user.

Approach: after building the tree from TOC, detect wrapper nodes whose title normalizes to
a known front-matter name ("contents", "table of contents", "toc") and that contain most of
the book's children - then promote the children one level up.
Sibling front-matter leaves (Cover Page, Copyright Page) stay as-is.

---

### Reconciler TOC filter misses numbering variants, injects duplicate subsections

**File:** `bookmind_tutor/ingestion/structure_reconciler.py` - TOC exclusion filter

Same book as above: the TOC lists `"4: Creation"` but the body heading is `"Creation"`.
`normalize_heading("4: Creation")` != `normalize_heading("Creation")`,
so the reconciler treated the body chapter heading as a NEW sub-heading candidate
and injected a subsection that duplicates its own chapter
(`section = "4: Creation"`, `subsection = "Creation"` - observed on 17 chapters).

Approach: when checking a candidate against the TOC set, also compare
numbering-stripped forms (drop leading `\d+[.:]?` and trailing page numbers before comparing).
A candidate matching any TOC entry in either form is structure already known, not a new sub-heading.
Note: keep the comparison conservative - stripping too aggressively could merge
genuinely different headings like "1: Introduction" and "2: Introduction" in appendix quizzes.

---

## Week 3 - Reasoning Strategies

### StrategyRouter: heuristic routing có thể sai với câu hỏi mơ hồ

**File:** `bookmind_tutor/agents/strategies/router.py`

Keyword heuristic ("compare", "vs"...) route dựa trên từ khóa, không hiểu intent thực sự.
Ví dụ: "compare apples and oranges" → Plan-and-Execute dù là câu đơn giản.
Documented trong `router.py` docstring.

Approach nếu muốn cải thiện: thêm 1 LLM call nhỏ để classify intent thay vì keyword matching.
Không urgent — heuristic đủ tốt cho phần lớn câu hỏi thực tế.

---

### Reflexion ít có giá trị với Book QA + strong model

**File:** `bookmind_tutor/agents/strategies/reflexion.py`

Trong demo thực tế, Reflexion không trigger retry lần nào (score 4-5/5 liên tục).
Lý do: Haiku self-evaluation generous + book QA initial answer thường đủ tốt.
Reflexion có giá trị hơn khi model yếu hơn hoặc task có tỷ lệ fail cao (coding, reasoning).

Approach nếu muốn test Reflexion kích hoạt: hạ SCORE_THRESHOLD hoặc dùng câu hỏi
yêu cầu tổng hợp phức tạp không có sẵn trong sách.

---

## Week 5-6 - Streamlit Frontend

### Streamlit cần PYTHONPATH để chạy đúng

**File:** `bookmind_tutor/tutor/app.py`

Chạy `streamlit run bookmind_tutor/tutor/app.py` từ project root mà không set PYTHONPATH sẽ gặp `ModuleNotFoundError: No module named 'bookmind_tutor'`.

Cách chạy đúng:
```
PYTHONPATH=/Users/victor/bookmind-tutor streamlit run bookmind_tutor/tutor/app.py
```

Approach: thêm `pyproject.toml` hoặc `setup.py` để package được install editable (`pip install -e .`), giúp import hoạt động trong mọi môi trường. Hoặc thêm script wrapper `run_app.sh`. Không urgent — dev workflow ổn với PYTHONPATH.

---

### User KG personalization chỉ match entities xuất hiện trong query text

**File:** `bookmind_tutor/knowledge_graph/graph_rag.py` - `_match_user_entities()`

Hiện tại user KG entities chỉ được thêm vào expansion nếu tên entity xuất hiện literal trong query (substring match). Nếu user biết "architecture pattern" nhưng hỏi "what is hexagonal architecture?", "architecture pattern" không match vì không có trong query string.

Approach nâng cao: dùng embedding similarity để tìm user-known entities gần với query topic (thay vì string match). Tốn thêm 1 embed call nhưng coverage rộng hơn nhiều. Để TODO — string match đủ tốt cho proof-of-concept.

---

### ~~KG visual page: neighborhood mode kém hữu ích khi chưa có edges~~ (fixed)

Đã implement lazy book KG build. `KGUpdateSuggester` giờ persist entities + relations
vào book KG sau mỗi extraction. Edges trong KG visual page được derive từ book KG
relations giữa UserEntity names. Neighborhood mode sẽ có edges sau vài turns chat.

---

### `streamlit-agraph` gửi `groups: null` lên vis.js

**File:** dependency `streamlit-agraph`

Library bug: `streamlit-agraph` luôn truyền `groups=null` vào vis.js options.
vis.js log error "Invalid type received for groups. Expected: object." trong console.
Không ảnh hưởng đến rendering - graph vẫn hoạt động đúng.

Không cần fix - đây là library bug không trong tầm kiểm soát của dự án.

---

### ~~Global aggregation questions ("list all chapters") hallucinate - RAG cannot ground them~~ (fixed)

**Files:** `bookmind_tutor/agents/tools/book_search.py`, `bookmind_tutor/retrieval/vector_store.py`, ingestion pipeline

Reported 2026-07-10: asking "co nhung chuong nao trong moi phan?" produces a confidently wrong chapter list (invented titles, off-by-one numbering).

Root cause chain (verified against the live index):
1. The TOC IS ingested (5 chunks, pages 7-14, chapter label "Table of Contents"), and the PDF even has a built-in outline (`doc.get_toc()`, 287 entries) with the exact ground truth.
2. But vector search cannot surface it: query "table of contents" ranks the first TOC chunk at #37 of 317.
   TOC chunk embeddings represent the *topics* of the chapter titles (architecture, microservices...), not the concept "table of contents". Dot leaders and page numbers further dilute the embedding.
3. `BookSearchTool` has no relevance threshold. All top-5 distances were > 0.78 (nothing close), yet the tool returned 5 topically unrelated excerpts instead of "no good match".
4. The LLM, given zero grounding but an answerable-looking question, reconstructs the chapter list from parametric memory and translates it to Vietnamese. Result: plausible but wrong.

This is a structural weakness of chunk-level RAG for global aggregation queries ("list all X in the book"), not a bug in any single component. Per the roadmap, fast lookup/synthesis from the book is the must-hit value, so this deserves a real fix.

Proposed fix (two independent parts):
- **`book_outline` tool** (primary): persist `doc.get_toc()` at ingest time (e.g. into `library.json` or a sidecar file) and expose a tool that returns the part/chapter structure verbatim. Deterministic, no embeddings involved. The chunk metadata already carries the full hierarchy (chapter = Part, section = Chapter), so a fallback can also be derived from ChromaDB metadata for books without an embedded outline.
- **Relevance threshold in `BookSearchTool`** (secondary): when the best distance is above a cutoff, prepend a warning like "results may be irrelevant" or return "no relevant passages found", so the model degrades honestly instead of hallucinating.

Status: root cause confirmed, fix not yet implemented. Related earlier fix: tool descriptions now instruct the model to always search in English (Vietnamese queries against English embeddings made retrieval strictly worse).

---

### ~~Verbatim-quote requests are not grounded; weak models skip re-search entirely~~ (system prompt fixed; model-choice recommendation documented)

**Files:** `bookmind_tutor/agents/harness.py` (`_DEFAULT_SYSTEM`), `.env` (model choice)

Found 2026-07-11: user asked for the author's exact wording behind one point of a
book summary ("The Beginning of Infinity", gpt-4o-mini) and the agent failed three turns in a row.
Verified from `data/interactions.jsonl`:

1. The summary turn itself ran 1 LLM call with 0 retrievals - the "insight" the user
   wanted quoted was the model's own synthesis, not a paraphrase of any passage.
   The model never disclosed that.
2. The verbatim turn retrieved exactly 1 chunk (an unrelated William Paley quote) and gave up.
3. The two follow-up nudges ("sao không lấy được nguyên văn?", "trong cùng một chương đó mà")
   ran 0 retrievals each - the model apologized or re-summarized from context instead of
   calling `get_book_section("Creation")`, which would have returned the full chapter text.

Direct retrieval tests confirm the index could serve the request
(e.g. "good explanations hard to vary" hits Chapter 13 at 0.756), so the failure is
agent behavior, not pipeline capability.

Two-part approach:
- **System prompt:** add an explicit rule for quote requests - when the user asks for
  exact/verbatim text, ALWAYS call search or get_book_section first and quote only text
  that appears in the returned excerpts; if the statement being asked about was the
  model's own synthesis, say so explicitly before offering the closest real passages.
- **Model choice:** gpt-4o-mini repeatedly stops using tools once a first attempt fails;
  claude-haiku-4-5 with the same harness re-searches. Recommend switching `.env` back to
  the Anthropic provider for real usage, and treat gpt-4o-mini as a budget/stress-test config.

---

### ~~get_book_section: field/contains interface leaks metadata schema to the LLM~~ (fixed)
**File:** `bookmind_tutor/agents/tools/get_book_section.py`

Root cause found while E2E-testing "Chuong 1 noi ve chu de gi?" with gpt-4o-mini.
Three stacked problems:

1. The tool required the LLM to pick a metadata `field` ("chapter" = part level, "section" = chapter level).
   This naming is inverted relative to common usage, and the correct choice depends on book structure the LLM cannot see.
2. In this book, Chapter 1 sits BEFORE Part I, so the structure detector placed it at part level: its 17 chunks (pages 21-40) live under `chapter='Chapter 1. Introduction'`, while their `section` values are subsections.
   Calling `field='section', contains='Chapter 1'` therefore matched Chapters 10-19 (73k chars of wrong content) and never the real Chapter 1.
3. The only section value containing "Introduction" was `'Chapter\xa01: Introduction'` from the Self-Assessment Questions appendix (page 393).
   The model ended up summarizing Chapter 1 from the quiz page.

Fix: redesign the tool interface around a single `section_name` parameter.
The tool matches against DISTINCT heading values collected from BOTH metadata fields.
Exactly one heading matches: return all its chunks.
Multiple headings match: return a short disambiguation list (name, chunk count, page range) instead of dumping content.
Zero match: return the list of available headings.
This keeps the schema knowledge inside the tool where it belongs.

Follow-up found during E2E: matching must also be punctuation-insensitive.
gpt-4o-mini guessed 'Chapter 1: Introduction' (colon); the body heading uses a period and the appendix TOC uses a colon plus a non-breaking space, so the exact match hit only the appendix quiz page.
Normalization now strips punctuation and collapses whitespace, grouping all spelling variants of a heading together.
E2E result: one tool call, 44k chars of the real Chapter 1, 2 LLM calls total (down from 7), fact-check 4/5 verified.

---

### ~~Sources panel disconnected from agent retrieval~~ (fixed)
**Files:** `bookmind_tutor/tutor/app.py`, `bookmind_tutor/agents/tools/base.py`,
`get_book_section.py`, `graph_rag_search.py`

Initial diagnosis blamed stale `_last_retrieved_chunks` ordering.
The true root cause was simpler and worse: the Sources panel never showed agent retrieval at all.
`app.py` ran an INDEPENDENT `_current_vs().search(question, k=5)` on the raw (Vietnamese) question and rendered those top-5 semantic hits as "Sources".
Whatever tools the agent actually called was irrelevant to the panel.

Fix: retrieval tools now record structured `SourceRef` entries (chapter, section, page range) in `Tool.last_sources` on each successful execute.
The chat handler clears them before each turn and renders them after.
The independent semantic search was removed.
Sources now shows exactly what the agent read, and shows nothing when the agent retrieved nothing (honest for ungrounded answers).

---

### ~~Fact-check cross-language mismatch~~ (resolved - was a symptom, not a bug)
**Context:** All fact-check claims showed "not found in retrieved excerpts" for Vietnamese answers.

This turned out to be a downstream symptom of the get_book_section retrieval bug above:
the verifier was checking claims against the WRONG chunks (appendix quiz page instead of the real chapter).
With correct retrieval, the LLM verifier handles Vietnamese claims against English chunks fine.
E2E confirmed: 4/5 claims verified with English evidence quotes, e.g.
"Vai tro cua kien truc su phan mem da mo rong..." matched against
"the role of software architect embodies a massive amount and scope of responsibility".

No code change needed.

---

### ~~Chunk size exceeds embedding model input limit~~ (fixed)
**File:** `bookmind_tutor/ingestion/hierarchical_chunker.py`

`all-MiniLM-L6-v2` truncates at 256 BPE tokens. Default lowered to `max_tokens=180`,
`overlap_tokens=30`. Existing books need to be re-indexed for the fix to take effect.
See PROGRESS.md (2026-07-11 entry) and `plans/structure_reconciler.md` step 1.

---

### ~~HierarchicalChunker discards detected L2 subsection titles~~ (fixed)
**Files:** `bookmind_tutor/ingestion/hierarchical_chunker.py`, `models.py`

Added `Chunk.subsection` field, propagated from level-2 nodes in `_chunk_node()`.
Indexed as ChromaDB metadata. `get_book_section` now searches it via `_HEADING_FIELDS`.
Existing books need to be re-indexed for subsection metadata to be populated.
See PROGRESS.md (2026-07-11 entry) and `plans/structure_reconciler.md` step 2.

---

### ~~Structure reconciler: TOC anchors + heading candidates~~ (implemented)
**Plan:** `plans/structure_reconciler.md`

`StructureReconciler` (new `ingestion/structure_reconciler.py`) implemented.
`StructureDetector` Tier 1 path now runs both TOC and Tier 3 heuristic, then passes
both to the reconciler. See PROGRESS.md (2026-07-11 entry) for full details.
Existing books need to be re-indexed.

---

### ~~Search layer does not exploit new subsection metadata~~ (fixed)
**Files:** `agents/tools/book_search.py`, `graph_rag_search.py`, `book_outline.py`,
`base.py` (SourceRef), `tutor/app.py`, `retrieval/vector_store.py`

All four gaps plus the contextual-embedding upgrade implemented 2026-07-11:
subsection in location labels, 3-level book_outline, SourceRef.subsection + UI,
and contextual embedding (heading breadcrumb prepended at embedding/BM25 time,
stored text stays clean; chunker max_tokens 180 -> 160 for breadcrumb headroom).
Both books re-indexed. E2E verified with Playwright.
See PROGRESS.md "Search layer subsection support + contextual embedding".

---

### Retrieval upgrades toward full contextual retrieval (planned)
**Reference:** https://www.anthropic.com/engineering/contextual-retrieval
**Context:** Compared our implementation against Anthropic's technique on 2026-07-11.
We have 2 of their 3 pillars in miniature (contextual embeddings + contextual BM25,
using free structural breadcrumbs instead of LLM-generated context). Their measured
numbers: contextual embeddings alone -35% retrieval failures, + contextual BM25 -49%,
+ reranking -67%.

Six gaps, ordered by recommended implementation sequence:

1. ~~**Retrieval eval set (do FIRST - everything else needs it to be measurable).**~~
   ~~20-30 queries per book with expected section/subsection answers.~~
   ~~Measure recall@k / labeled-hit rate like Anthropic's failure-rate metric.~~
   **Done 2026-07-12.** `data/eval/` contains 3 JSON sets (Deutsch 22q, FoSA 22q, DDIA 25q).
   Script: `scripts/eval_retrieval.py`. Baselines: Deutsch@5=82%, FoSA@5=91%, DDIA@5=100%.

2. ~~**RRF (Reciprocal Rank Fusion) instead of max-norm alpha blend.**~~
   ~~`vector_store.py _search_hybrid()` normalizes BM25 by max score in candidate set,~~
   ~~so the top BM25 doc always gets 1.0 -> hybrid 0.500 even when dense disagrees.~~
   **Done 2026-07-12.** `1/(60+rank_dense) + 1/(60+rank_bm25)`. See PROGRESS.md.

3. ~~**Cross-encoder reranker (biggest measured win in Anthropic's data: 49% -> 67%).**~~
   **Done 2026-07-12.** `bookmind_tutor/retrieval/reranker.py` - `CrossEncoderReranker`
   wrapping `cross-encoder/ms-marco-MiniLM-L-6-v2`. VectorStore now accepts optional
   `reranker=` param; when set, fetches k*8 candidates then reranks to top-k.
   Wired into `app.py` via `_RERANKER` singleton (disable with `ENABLE_RERANKER=false`).
   A/B measured with `scripts/eval_retrieval.py --reranker`:
   @5 Deutsch +4% (82→86%), FoSA +5% (91→96%), DDIA flat (100%). @1 drops because
   cross-encoder and RRF optimize different relevance notions - not a regression for RAG
   (LLM reads all k, so hit@3 = hit@1 in practice). 395 tests passing.

4. ~~**Raise top-k 5 -> ~20 (only together with reranker).**~~
   **Done 2026-07-12.** `VectorStore.search()` default k=5→20, `GraphRAGRetriever`
   k=5→20 in app.py and runner.py. With reranker: k*8=160 candidates → rerank → top-20.
   Deutsch @5: 82%→86%, @10: 86% (same), @20: 91% (same ceiling). DDIA/FoSA unchanged.

5. **LLM-generated situating context (the "full" contextual retrieval).**
   Breadcrumb says WHERE the chunk is; LLM context says WHAT it discusses -
   resolves referential ambiguity ("this approach", "the previous technique") that
   headings cannot. Haiku at ingest time, one-time ~$0.5-1/book. Run as an A/B
   experiment against breadcrumb-only using the eval set from item 1.

6. **Embedding model upgrade (structural ceiling, decide separately).**
   all-MiniLM-L6-v2's 256-BPE limit forced the entire chunk-size dance and caps
   quality. Voyage/Gemini handle 8k+ tokens. Bigger decision: cost, API dependency.

Honest caveat from the article, worth knowing for interviews: for knowledge bases
under ~200k tokens (~500 pages), long-context + prompt caching beats RAG entirely.
DDIA is safely above the threshold; FoSA is borderline. For this learning project,
RAG remains the right choice by objective.

---

### ~~EvalRunner: conversation memory bleeds across benchmark questions~~ (fixed)

**File:** `bookmind_tutor/evaluation/runner.py` - `run()` (lines 154-157)

Found 2026-07-11 while auditing eval harness capabilities.
`run()` builds ONE QAAgent per config and reuses it for every question,
but `QAAgent` creates its `ConversationMemory` at init and passes it into every `chat()` call.
Result: question 10 in a benchmark sees the full conversation of questions 1-9.

Consequences for single-turn benchmarking:
- Scores are contaminated by unrelated prior context (student-level inference,
  strategy routing, and answer content all read the accumulated history).
- Question ORDER affects results, so runs are not comparable across question-set edits.

Fix for the current single-turn design: rebuild the agent (or call a memory reset)
per question inside the inner loop.
Design note for the planned multi-turn agent-behavior eval set: the memory mechanism
is exactly what multi-turn scenarios need - the fix should make memory lifetime
EXPLICIT (fresh per question in single-turn mode, fresh per scenario and shared
within it in multi-turn mode), not simply always-reset.

Related harness gaps recorded in the "Agent behavior eval set" entry below:
no multi-turn scenario support, no programmatic assertion layer,
no baseline/pass-fail gating, no retrieval-only mode.

---

### Agent behavior eval set distilled from real interaction history (planned)

**Files:** `data/interactions.jsonl` (source material), `bookmind_tutor/evaluation/` (existing infra)
**Relation to the retrieval eval set (item 1 above):** complementary, not overlapping.
The retrieval set measures the PIPELINE (single query -> did the right section surface).
This set measures the AGENT (multi-turn conversation -> did it behave correctly end-to-end).
Four of the five observed failure patterns cannot be expressed as single-query retrieval tests.

**Status 2026-07-11:** does not exist yet.
`data/eval_questions.example.json` is a 3-question placeholder template;
no real question set has ever been written for any book.
The Week 7 EvalRunner infra (LLM-as-judge, multi-config, hallucination_score) works
but has never been fed realistic questions.

**Harness gaps (audited 2026-07-11):** the existing EvalRunner supports single-turn
benchmarking only. Missing for the two planned eval sets:
1. Multi-turn scenario support (scripted user turns referencing the agent's prior output).
2. A programmatic assertion layer (raw signals exist in the trace - `num_llm_calls`,
   `retrieved_chunks` - but there is nowhere to declare per-scenario checks).
3. Baseline comparison / pass-fail gating (reports produce means, never "regressed vs last run").
4. A retrieval-only mode for the recall@k set (runner always builds full QAAgent + judge -
   expensive, and judge-score variance drowns the retrieval signal).
Also see the memory-bleed bug entry above - its fix should introduce explicit
memory-lifetime control, which multi-turn scenarios will reuse.

**Source material:** 68 real interactions logged in `data/interactions.jsonl`,
including every investigated failure case (hallucinated chapter list, appendix quiz page p.395,
the verbatim-quote 3-turn failure with "The Beginning of Infinity").
Real usage questions expose weaknesses that invented benchmark questions do not.

**Failure patterns to cover (all observed in real sessions):**
1. Verbatim-quote requests referencing the assistant's own earlier summary -
   forces the agent to distinguish "what the book says" from "what I synthesized".
2. Challenge follow-ups ("sao bạn không lấy được?", "trong cùng một chương đó mà") -
   measures whether the agent re-searches instead of apologizing from context.
3. Global aggregation ("tổng kết insights", "có những chương nào trong mỗi phần?") -
   chunk RAG is structurally blind to these; book_outline must be used.
4. Vague conversational continuations ("sự phát triển tri thức đi", "sang chương 4 đi") -
   depends on conversation context being carried into tool queries.
5. Vietnamese questions against English embeddings.

**Approach:**
- Distill each pattern into 3-5 multi-turn scenarios per book
  (scripted user turns; some turns must reference the agent's previous output).
- Grade with LLM-as-judge using per-scenario rubrics
  (e.g. for pattern 2: "did the agent call a retrieval tool after the challenge?" -
  checkable from num_llm_calls / retrieved_chunks in the interaction log, no judge needed).
- Prefer programmatic checks (tool-call counts, retrieval counts, quoted-text-appears-in-chunk)
  over judge scores wherever possible - they are cheaper and not gameable.
- Run as a regression suite before/after every agent-affecting change
  (model swap, system prompt edit, RRF, reranker) so old failures cannot silently return.

Depends on: nothing (can be built now). The retrieval eval set (item 1) remains
first for the retrieval upgrades; this set gates agent/prompt/model changes instead.

---

## Beyond the roadmap - closing the SoTA gap

Gap analysis done 2026-07-11 (session: "đã có thể coi là SoTA chưa?").
Assessment: modern architecture patterns with deliberately modest components.
The 6 retrieval upgrades above are the near-term path; the items below are the
larger jumps. All are outside the 13-week roadmap scope - schedule explicitly.

### Long-context baseline: whole book in prompt vs RAG
**Reference:** Anthropic contextual-retrieval article ("under ~200k tokens, skip RAG")

For book QA, the honest SoTA competitor to our entire pipeline is: put the whole
book in the prompt with caching. FoSA is borderline (~200k tokens), DDIA is above.
Experiment: run the retrieval eval set (backlog item 1 above) against a
long-context baseline; compare quality, cost per question, latency.
Whatever the result, it is strong interview material: either RAG wins measurably,
or we know exactly when it does not. Depends on: retrieval eval set.

---

### GraphRAG full variant: community detection + hierarchical summaries
**Files:** `knowledge_graph/graph_rag.py`, `graph_store.py`
**Reference:** Microsoft GraphRAG (community detection via Leiden + community summaries)

Current implementation is a lightweight variant: entity-based query expansion only.
It cannot answer global/thematic questions ("what are the main themes of this book?",
"how do the ideas in Part I connect to Part III?") - chunk RAG is structurally
blind to them and book_outline only covers structure, not synthesis.
Full variant: cluster the entity graph into communities, LLM-summarize each
community at ingest time, retrieve community summaries for global questions.
Cost: one-time LLM pass per book. Router decides chunk-RAG vs community-RAG per query.

---

### Table and figure extraction (upgraded from stretch goal)
**File:** new module `bookmind_tutor/ingestion/table_extractor.py`

Tables are currently skipped entirely; figures too. Both books are technical -
DDIA's figures carry real explanatory weight.
- Tables: PyMuPDF `page.find_tables()` -> markdown text in chunks.
- Figures: extract images at ingest, one VLM call each for a text description,
  index the description as a chunk with a figure reference.
This is the largest content-coverage gap: a chunk of the books' information
simply does not exist in our index today.

---

### Late-interaction / visual retrieval (exploratory)

If the embedding model upgrade (retrieval item 6) happens, also evaluate
late-interaction retrievers (ColBERT-style) and visual retrieval (ColPali) which
skip text extraction entirely by embedding page images.
Exploratory - only worth it after the eval set exists to judge the trade-off.
