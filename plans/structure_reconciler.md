# Plan: Hybrid Structure Reconciler (TOC Anchors + Heading Candidates)

## Context

This plan extends `structure_detection_tiers.md`.
It is based on an empirical comparison run on 2026-07-10 between the current pipeline, Docling, and the `docling-hierarchical-pdf` post-processor, using "Fundamentals of Software Architecture" (which has an embedded PDF outline) as the test book.

### What the comparison showed

**Current pipeline (PyMuPDF + StructureDetector + HierarchicalChunker):**

- Detects 3 real levels (L0 Part, L1 Chapter, L2 Section) - structurally correct.
- But the chunker only stores L0/L1 in `Chunk.chapter`/`Chunk.section`; L2 titles are detected and then discarded.
  Result: only 28 distinct section labels reach the vector store, although the tree contains ~300 L2 headings.
- Known false positives at L2: math symbols (`'∑'`), format labels (`'Constant width bold'`), title-page fragments (`'Fundamentals of'`, `'Software'`, `'Architecture'` as three separate headings).
- Known failure: headings that wrap across two lines become two sibling nodes (`'Measuring and Governing'` + `'Architecture Characteristics'`).

**Docling (raw):**

- Layout model detects 395 section headers reliably, plus tables/code/formulas.
- But the PDF backend hardcodes `level=1` for every heading (verified in source: `doc.add_heading(text, prov)` is called without a level argument).
  The hierarchy is flat.
  Docling's own `HierarchicalChunker` produced 3165 chunks, 100% with breadcrumb depth 1.
- Retrieval spot-check on 5 queries: identical average top-1 cosine (0.750) versus the current pipeline.
  No retrieval gain from switching parser.

**docling-hierarchical-pdf (community post-processor):**

- Infers depth from heading numbering patterns, falling back to font-size DBSCAN clustering.
  Produced 4 levels (H1 x5, H2 x33, H3 x124, H4 x95) - real depth.
- But the hierarchy is anchored wrongly: the whole book ended up nested under `'Preface: Invalidating Axioms'`.
- Root cause is visible in its warnings: it reads the embedded TOC but matches TOC titles to body headings by **exact string**.
  `'Chapter 1. Introduction'` (TOC) never matches `'CHAPTER 1 Introduction'` (body), so every Part/Chapter anchor is lost and clustering runs unanchored.

### The key insight

The exact-match failure is the same bug class we fixed in `get_book_section` this week: LLM/TOC/body spellings of the same heading differ in case, punctuation, and whitespace, so matching must be normalization-insensitive.
The post-processor had the right architecture (TOC as ground truth + clustering for depth) but failed on a solved problem.

We can build the same architecture on our own pipeline, reusing the `_normalize()` matching layer we already wrote, and keep every component we trust.

---

## Design

```
[ANCHOR LAYER]      Embedded TOC entries [level, title, page]     <- ground truth skeleton
      |             (fallback: Tier 2 LLM parse of printed TOC)
      v
[CANDIDATE LAYER]   Heading candidates from Tier 3 font heuristic <- generality and depth
      |             (text, page, font_size, bold, y-position)
      v
[RECONCILE]         Match candidates <-> TOC entries:
      |               normalized text equality (case/punct/whitespace-insensitive)
      |               + page proximity window (TOC page +- offset tolerance)
      |               + join consecutive same-page heading lines before failing
      v
[ATTACH]            Unmatched candidates = sub-heading candidates:
      |               attach under nearest preceding anchor (page, then y-position)
      |               relative depth within a chapter via local font-size ranking
      |               false-positive filter before attachment
      v
[OUTPUT]            DocumentTree with levels 0..3 (Part/Chapter/Section/Subsection)
```

### Why each layer covers the others' weaknesses

| Weakness | Covered by |
|---|---|
| TOC only has 2-3 top levels, no sub-sections | Candidates supply L2/L3 depth |
| Font heuristic false positives (`'∑'`, format labels, title page) | Unmatched candidates pass a filter before attachment; TOC anchors are never displaced by them |
| Two-line headings split into siblings | TOC has the full title; reconciler tries joining consecutive heading lines on the same page |
| Global clustering mis-levels the whole book (the Preface failure) | Part/Chapter levels are fixed by anchors; font ranking only decides depth *within* a chapter |
| Book has no embedded TOC | Tier 2 (LLM TOC parse) supplies anchors; with no TOC at all, the reconciler degrades to the current Tier 3 tree unchanged |

The design is monotonic: it never produces a worse tree than the current pipeline, because anchors only add structure and the fallback path is the existing behavior.

---

## Components

### 1. Shared normalization utility

Move `_normalize()` from `bookmind_tutor/agents/tools/get_book_section.py` to a shared module (e.g. `bookmind_tutor/ingestion/utils.py` or a new `bookmind_tutor/text_utils.py`), import it in both places.
Same semantics: lowercase, replace `\xa0`, strip punctuation, collapse whitespace.

### 2. `StructureReconciler` (new class, `bookmind_tutor/ingestion/structure_reconciler.py`)

Input:
- `toc_entries: list[TocEntry]` - `(level, title, page)` from `PdfExtractor.extract_toc()` or Tier 2.
- `candidates: list[HeadingCandidate]` - from Tier 3 classification, each with `text`, `page`, `font_size`, `bold`, `y_pos`.

Steps:

**a. Anchor matching.**
For each TOC entry, search candidates within a page window (entry page +- 2, to absorb page-offset noise).
Match on normalized equality first; then normalized prefix/containment; then try joining 2-3 consecutive heading-classified lines on the same page and re-matching (handles wrapped titles).
A TOC entry with no candidate match still becomes an anchor node at its TOC page (the TOC is trusted; the body heading was simply missed or mis-classified).

**b. Anchor tree construction.**
Build the upper tree directly from TOC levels (existing `_build_tree_from_toc` logic).
Page ranges from consecutive anchors, as today.

**c. Sub-heading attachment.**
Candidates not consumed by step (a) are sub-heading candidates.
Filter first (see d).
Attach each survivor under the deepest anchor whose page range contains it.
Depth below the anchor: rank the distinct font sizes of survivors within that anchor's page range; largest size = anchor level + 1, next = anchor level + 2, capped at one extra level below the deepest TOC level (avoid manufacturing depth the book does not have).
This ranking is local per anchor, so a Part with large section fonts cannot distort another Part's levels.

**d. False-positive filter for unmatched candidates.**
Reject a candidate when any of:
- normalized text is empty, a single character, or contains no letters (kills `'∑'`);
- its font size appears on fewer than `min_heading_pages` distinct pages across the book (existing rule, now applied to attachment instead of the level map);
- it sits on a page before the first TOC anchor (title page/praise-page fragments).

### 3. `StructureDetector` integration

Current Tier 1 short-circuits: if the embedded TOC exists, the font heuristic never runs.
Change the flow: when TOC exists, run **both** - TOC for anchors, heuristic classification for candidates - then hand both to `StructureReconciler`.
Tier 2 and Tier 3 fallbacks keep their current roles.

### 4. `Chunk.subsection` field (prerequisite, independently valuable)

Add `subsection: str | None` to `Chunk`, propagate L2 titles in `HierarchicalChunker._chunk_node()` (and L3 when the reconciler lands).
Index it as ChromaDB metadata.
`get_book_section` then matches against three fields instead of two - its interface does not change, since it already searches all fields by a single name.

### 5. Chunk size fix (prerequisite, independently valuable)

`all-MiniLM-L6-v2` truncates at 256 BPE tokens (~195 English prose words); measured empirically on 2026-07-10 (embeddings of 260+ word texts are byte-identical when only the tail differs).
Current `max_tokens=512` (whitespace words) means 52.8% of chunks have an un-embedded tail.
Change `HierarchicalChunker` default to `max_tokens=180`, `overlap_tokens=30`, and re-index existing books.
This is a retrieval-quality bug fix independent of the reconciler.

---

## What we deliberately do NOT do

- **Do not adopt Docling as a dependency.**
  Evidence: retrieval quality tied (0.750 vs 0.750), its PDF hierarchy is flat, its text has spacing artifacts, and it costs 60-120s per book versus 5-10s.
  The candidate layer stays on our Tier 3.
  Docling could later be swapped in as an alternative candidate source behind the same `HeadingCandidate` interface if we hit a book class our heuristic cannot read (scanned PDFs), but that is out of scope.
- **Do not run DBSCAN/global clustering.**
  Local font-size ranking within an anchor's page range is simpler, deterministic, and cannot mis-level across chapters.

---

## Verification

1. Unit tests: synthetic TOC + candidates covering: exact match, punctuation variant match, wrapped two-line title, unmatched TOC entry, false-positive rejection, book with no TOC (fallback path unchanged).
2. Integration sanity check (`scripts/sanity_check.py`) on both real books:
   - % of TOC entries matched to a body candidate (target: >90% for FoSA, which failed at 0% in docling-hierarchical-pdf);
   - count of L2/L3 nodes attached; count of candidates rejected by the filter;
   - spot-check that `'∑'` and `'Constant width bold'` no longer appear as sections.
3. Re-run the retrieval spot-check queries and confirm no regression.
4. E2E: "Laws of Software Architecture là gì?" should retrieve chunks labeled with that subsection.

---

## Implementation Order

| Step | What | Effort | Depends on |
|------|------|--------|-----------|
| 1 | `max_tokens` 512 -> 180 + re-index | Small | - |
| 2 | `Chunk.subsection` + propagate existing L2 | Small | - |
| 3 | Shared `_normalize()` utility | Trivial | - |
| 4 | `StructureReconciler` + tests | Medium | 3 |
| 5 | Wire into `StructureDetector`, re-index, sanity check | Small | 4 |

Steps 1-2 ship value on their own and should land before the reconciler.
