# Plan: 3-Tier Structure Detection

## Context

The current `StructureDetector` uses font-size heuristics as the sole signal
for inferring document hierarchy (chapter/section titles).
After testing on two real O'Reilly books, the heuristic works reasonably well
but is fundamentally guessing at structure that the PDF may already encode
as metadata.

This plan proposes a 3-tier fallback chain that uses the most reliable source
first and only falls back to less reliable methods when needed.

---

## Architecture

```
Tier 1: doc.get_toc()               - embedded PDF outline (free, 100% accurate)
         |
         v (if empty / fewer than 3 entries)
Tier 2: LLM parse of TOC page       - 1-2 LLM calls on the TOC text only
         |
         v (if no TOC page found, or LLM fails)
Tier 3: Font-size heuristic         - current implementation (fallback of last resort)
```

---

## Tier 1 - Embedded PDF Outline via get_toc()

### Why first

`pymupdf.Document.get_toc()` returns the outline/bookmark tree embedded in the
PDF file itself.
Many published books (especially ebooks and well-produced O'Reilly titles) include
this metadata.
It is ground truth: 100% accurate, zero cost, no LLM, no heuristic needed.

### Output format

`doc.get_toc()` returns a list of `[level, title, page_number]` triples where
`level` is 1-based (1 = chapter, 2 = section, 3 = subsection).

### Implementation notes

- `PdfExtractor` already opens the `pymupdf.Document` object.
  Add `extract_toc(pdf_path) -> list[list]` method (or return TOC alongside pages)
  so `StructureDetector` can receive it without re-opening the file.
- Alternatively: add an optional `toc: list | None = None` parameter to
  `StructureDetector.detect()` and pass the raw TOC list from the caller.
- Build `DocumentNode` tree from TOC entries using the same node-stack approach
  as `_build_tree()`, but driven by the TOC level instead of classified line spans.
- Page ranges: infer from consecutive entries (section N ends where section N+1
  at the same or higher level starts, last section ends at the last page).
- Body text (raw_text) for each node still requires iterating page spans and
  assigning text to the correct node by page number - the TOC gives us the
  boundaries, not the content.
- Threshold: require at least 3 TOC entries before trusting it.
  A 1-entry TOC (e.g. just "Cover") is not useful.

### Priority

**Implement now (Week 1)** - this is 30-40 lines of code, no new dependencies,
no LLM cost, and could make the heuristic entirely unnecessary for well-formed books.

---

## Tier 2 - LLM Parse of Table of Contents Page

### Why second

When no embedded outline exists, the printed "Table of Contents" page is the
next best source.
It contains chapter titles and page numbers in structured form, but with messy
formatting: dot leaders (`......`), right-aligned page numbers, indentation
that may or may not survive PDF text extraction.

Regex alone is fragile here because every book formats its TOC differently.
An LLM given 1-2 pages of TOC text can reliably return a clean JSON list of
`{title, level, page}` objects - this is exactly the kind of semi-structured
parsing where LLMs outperform regex.

### Cost profile

- Detect TOC page: scan first 20 pages for "Contents" / "Table of Contents"
  keyword (free).
- Parse: 1 LLM call on ~500-1500 tokens of text.
  At Sonnet pricing this is well under $0.01 per book.

### Implementation notes

- Add `_find_toc_page(pages) -> PageContent | None` in `StructureDetector`:
  scan pages[0:20] for a page whose text contains "Contents" or "Table of Contents"
  near the top.
- Build a prompt: `"Parse this Table of Contents into JSON: [{title, level, page}].
  level 1 = chapter, 2 = section. Return only valid JSON."` + page text.
- Parse the JSON response and build the node tree the same way as Tier 1.
- Fallback: if LLM returns invalid JSON or fewer than 3 entries, move to Tier 3.

### Priority

**Defer to backlog** - do not implement in Week 1.
Reasons:
1. Week 1 scope is "pure ingestion pipeline, no LLM."
   Introducing an LLM client here blurs the learning objective for this phase.
2. LLM client setup, cost monitoring, and prompt engineering belong in
   Weeks 2-3 when agents are in scope.
3. Tier 1 likely handles the common case (published books with embedded TOC)
   already.
4. The heuristic (Tier 3) is an adequate fallback for books without TOC pages.

When to revisit: after Week 3 (agent harness is wired up and LLM client is in place),
add Tier 2 as an `EnhancedStructureDetector` or an optional flag on the existing class.

---

## Tier 3 - Font-Size Heuristic (Current Implementation)

### Role

Last-resort fallback for:
- Scanned PDFs with no embedded outline and no readable TOC page.
- Uniformly-formatted documents (slides, reports) where Tiers 1 and 2 yield nothing.

The current implementation after Week 1 fixes is already solid for most cases.
Keep as-is.

---

## LLM-as-Verifier for Ambiguous Headings

### Pattern description

Run the full heuristic (Tier 3) to produce a list of heading candidates.
For lines where the heuristic assigns low confidence (e.g. bold same-size as body,
borderline font size, or a line that looks like it could be either a heading or
an emphasized sentence), send only those candidates to an LLM for binary
classification: "Is this a section heading or body text?"

### Why this is better than LLM-for-full-book

- Heuristic screens the entire book cheaply and quickly.
- LLM sees only the ambiguous subset - 10-50 candidates at most per book.
- Avoids hallucination risk from long-context reasoning about book structure.
- Controls cost: most lines are unambiguous, so LLM is rarely invoked.

### Priority

**Defer to Week 7 (Evaluation)** - this pattern is more naturally framed as
"evaluating and improving heading detection quality" than as ingestion logic.
The evaluation week already plans to measure pipeline quality; adding a verifier
loop fits that scope better than Week 1.

---

## Implementation Order

| Step | What | When | Effort |
|------|------|------|--------|
| 1 | Add `extract_toc()` to `PdfExtractor` | Now (Week 1) | ~20 lines |
| 2 | Add TOC-driven tree building to `StructureDetector` | Now (Week 1) | ~40 lines |
| 3 | Wire Tier 1 -> Tier 3 fallback chain | Now (Week 1) | ~10 lines |
| 4 | LLM TOC parsing (Tier 2) | After Week 3 | Medium |
| 5 | LLM-as-verifier for ambiguous headings | Week 7 | Medium |
