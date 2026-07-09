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

### Table extraction (stretch goal)
**File:** new module `bookmind_tutor/ingestion/table_extractor.py`

Tables are currently skipped (PyMuPDF block type 1 = image, type 0 = text - tables in PDF are often rendered as text blocks with irregular spacing or as images).
PyMuPDF has a `page.find_tables()` API that can extract structured table data.

Deferred per roadmap: not needed until evaluation or KG work requires it.
