# Ingestion Pipeline — Week 1

Three-stage pipeline: **PDF extraction** -> **structure detection** -> **hierarchical chunking**.

---

## Heading Detection Heuristic

The `StructureDetector` uses two independent signals to identify heading lines:

### Signal 1: Font size (primary)

1. **Body size estimation.** We scan all text spans across all pages, bucket
   their font sizes to the nearest 0.5 pt, and weight each bucket by total
   character count in that bucket.
   The bucket with the most characters is the body font size.
   Sizes below 6 pt are excluded (subscripts, footnotes, page numbers).

2. **Heading threshold.** A line whose dominant span size exceeds
   `body_size * min_heading_size_ratio` (default: 1.15, i.e. 15% larger)
   is a heading candidate.

3. **Level assignment.** Distinct heading sizes are sorted descending.
   The largest size maps to level 0 (chapter), the next to level 1 (section),
   and so on - capped at level 2 (subsection) to avoid spurious deep nesting.
   Sub-point rendering variation is handled by rounding to 0.5 pt buckets.

### Signal 2: Text patterns (fallback)

Used when a line is at body size but matches a conventional heading pattern:

| Pattern | Level | Example |
|---|---|---|
| `Chapter N`, `Part N`, `Appendix A` | 0 | "Chapter 3: Concurrency" |
| `N. Title` | 0 | "1. Introduction" |
| `N.M Title` | 1 | "2.4 Background" |
| `N.M.K Title` | 2 | "3.1.2 Details" |
| ALL CAPS, <= 6 words | 0 | "CONCLUSIONS" |

### Signal 3: Bold same-size (optional)

If `bold_as_section=True` (default), bold lines at body font size with <= 8
words are treated as level-2 section headings.
This catches books that use bold sub-headings without a font size increase.

---

## Known Limitations and Failure Modes

**1. Multi-column layouts.**
PyMuPDF's `sort=True` sorts blocks by (y0, x0), which approximates reading
order for single-column text but can interleave columns incorrectly for
two-column academic papers.
Impact: heading detection still works (font size analysis), but body text
under a heading may include fragments from the adjacent column.

**2. Body size estimation fails on heavily formatted documents.**
If a document has an unusual amount of decorative large text (e.g. coffee-table
books with large captions), the estimated body size may be inflated, causing
some real headings to fall below the threshold.
Mitigation: lower `min_heading_size_ratio` to 1.05-1.10 for such documents.

**3. Pull quotes and figure captions.**
Large-font pull quotes (decorative repeated text) are excluded by
`max_heading_words` (default 15) and `max_heading_chars` (default 150).
Figure captions in a larger-than-body font can still slip through if they are
short. They will appear as spurious chapter breaks in the tree.

**4. Documents with no font variation.**
Some PDFs are produced by scan-to-PDF workflows that flatten all text to a
single font size.
In this case no heading sizes are detected, and `detect()` falls back to a
flat single-node document.
The chunker will still work correctly on the flat tree.

**5. `char_offset_start/end` in `Chunk` are relative to `DocumentNode.raw_text`,
not to the original PDF page text.**
To trace a chunk back to a PDF page: use `chunk.page_range` to identify the
page(s), then search for `chunk.text` within those pages' text.
This will be revisited when building the Knowledge Graph in Week 4.

**6. Overlapping headings at the same level.**
If a level-1 heading appears inside what the heuristic thinks is another
level-1 heading (both are the same font size), it will be treated as a sibling
rather than a child.
True sub-headings at the same visual size as their parent cannot be detected
by font-size alone - pattern matching (Signal 2) is the only recourse.

---

## Configuration Quick Reference

```python
StructureDetector(
    min_heading_size_ratio=1.15,  # increase for noisy docs, decrease if headings are missed
    bold_as_section=True,          # set False to ignore bold-only headings
    max_heading_words=15,          # increase if chapter titles are long
    max_heading_chars=150,
    line_tolerance_pts=3.0,        # increase for PDFs with inconsistent line spacing
)

HierarchicalChunker(
    max_tokens=512,       # approx word count (BPE tokens ~= 1.3x word count)
    overlap_tokens=50,    # words of overlap between consecutive chunks
)
```
