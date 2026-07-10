# Lessons: Embedding Truncation, Structure Reconciler, and PDF Text Splitting

Session date: 2026-07-10 to 2026-07-11
Related plan: `plans/structure_reconciler.md`

---

## 1. Silent embedding truncation is easy to miss and expensive to debug

`all-MiniLM-L6-v2` (ChromaDB's default embedding function) truncates input at 256 BPE
tokens without any warning, error, or metadata field indicating truncation.
When the input exceeds that limit, the tail is silently ignored - the embedding only
represents the first ~195 English prose words.

With `max_tokens=512` (the old default), 52.8% of FoSA chunks (239/453) had
un-embedded tails.
Queries about content that happened to land in the second half of a chunk would
never retrieve that chunk, silently.

**How to detect it empirically:**
Generate two texts that are identical for the first N words and differ only afterward.
If their cosine similarity is 1.000000 (byte-identical vectors), the model truncated
at word N.
Confirmed at ~255 whitespace-split words (~195 for real English prose, ~128 for
single-syllable filler words).

**Lesson:** When choosing `max_tokens` for any chunker paired with a sentence-transformer,
verify the model's actual truncation point empirically, not from documentation alone.
The safe default is 50-70% of the model's stated limit to account for tokenization
ratio (BPE tokens > whitespace words).

---

## 2. Heading matching needs normalization at every layer

The same heading appears with different spelling in three places:
- The embedded PDF TOC: `"Chapter 1. Introduction"` (period)
- The body text: `"CHAPTER 1 Introduction"` (all-caps, no punctuation)
- An LLM guessing the name: `"Chapter 1: Introduction"` (colon)

All three normalize to `"chapter 1 introduction"` after lowercasing, stripping
punctuation, and collapsing whitespace.

This pattern showed up in three independent contexts in this project:
1. `get_book_section` tool - LLM vs. ChromaDB metadata mismatch
2. `docling-hierarchical-pdf` post-processor failure - exact string match lost every
   TOC anchor (0% match rate)
3. `StructureReconciler` - needed for both TOC exclusion filter and heading injection

**Lesson:** Any system that matches user-provided or LLM-generated text against
stored headings needs normalization.
Exact string match is wrong for heading names.
Centralizing the normalization function (one implementation, imported everywhere)
prevents the bug class from re-appearing in new modules.

---

## 3. The monotonic enhancement pattern

When adding a new processing layer on top of existing correct behavior, design it
to be monotonic: the output is never worse than the input.

For `StructureReconciler`:
- If no candidates survive the false-positive filter → return the anchor_tree unchanged.
- If a candidate can't be located in the raw_text (text matching failure) → skip it,
  don't abort.
- If no embedded TOC exists → the reconciler isn't called at all; Tier 3 fallback
  is unchanged.

This property makes the feature safe to ship incrementally:
deploying the reconciler cannot regress books that were working correctly before.

**Lesson:** For ingestion enhancements that risk corrupting existing retrieval,
build in a no-op path and test it explicitly.
"Returns input unchanged when nothing to do" is a test case, not a given.

---

## 4. Local font-size ranking vs. global clustering

`docling-hierarchical-pdf` used DBSCAN clustering globally across all headings in the
document to assign hierarchy levels.
The failure mode: if Preface has larger subheading fonts than Chapter 1, DBSCAN
assigns Preface subheadings to level 1 and Chapter 1 subheadings to level 2,
mis-leveling the entire book.

The reconciler uses **local ranking per anchor**:
within each anchor's page range, rank the distinct font sizes found there (largest = L2,
next = L3, etc.).
A Part with large section fonts cannot distort another Part's levels.

This is simpler (no sklearn dependency, no hyperparameter tuning), deterministic,
and correct for the actual problem: depth is relative to the surrounding structure,
not absolute across the whole document.

**Lesson:** For hierarchical document analysis, local context usually beats global
statistics.
Global clustering assumes the same font size means the same level everywhere in the
book, which breaks for books with variable chapter designs.

---

## 5. NLTK sentence tokenizer fails on non-prose text

The `HierarchicalChunker` uses `nltk.sent_tokenize()` to split text at sentence
boundaries.
This works well for prose (paragraphs, explanations, arguments).

It fails silently for:
- **Index pages**: `"A\naborts (transactions), 222\naggregation pipeline, 48\n..."` -
  no sentence-ending punctuation, so the entire index becomes one "sentence".
  Result: one 11,000-word chunk that can never be split.
- **Code blocks**: no periods at line ends, so a 200-line code example becomes one chunk.
- **Numbered lists**: list items often don't end with periods.

The `max_tokens` limit is never enforced for a single sentence: the chunker always
emits at least one sentence per chunk, even if that sentence exceeds the limit.

**Lesson:** NLTK sentence tokenization is a good default for textbook prose, but
ingestion pipelines need a separate strategy for structured content (code, indexes,
tables).
For retrieval purposes, oversized non-prose chunks are usually acceptable (they
won't rank highly for most queries) - but knowing this happens is important for
debugging unexpected large chunks.

---

## 6. PDF text is usually split into blocks, which enables reliable heading detection

In PyMuPDF, text is extracted as blocks (rectangular regions).
In well-typeset books (O'Reilly, Addison-Wesley), headings are almost always in
their own block - they are visually separate from body text.

This means a heading's text should appear as a standalone "line" in the page text
(separated by `\n\n` from the surrounding body), making normalized line-by-line
matching reliable for finding headings in `raw_text`.

In contrast, body sentences wrap across multiple lines within one block, so
sentence-level search in `raw_text` is less reliable.

**Lesson:** When splitting `raw_text` at a heading position, compare LINES (not
substrings, not sentences) against the normalized heading text.
This is specific to PDF-extracted text; scanned PDFs (OCR output) would need
a different approach.

---

## 7. Python venv symlink confusion

When a venv is created with `python3.12`, the venv's `python` symlink may still
point to the system default Python (3.14 in this case, which was broken due to a
Homebrew libexpat mismatch).

Symptom: `.venv/bin/python -c "import nltk"` fails with ModuleNotFoundError,
but `.venv/bin/python3.12 -c "import nltk"` succeeds.
Both point into the same venv but the first uses a broken Python binary.

**Lesson:** When a venv behaves unexpectedly (package installed but not importable),
check which interpreter `python` resolves to.
Always use `python3.X` explicitly in scripts and Makefiles that target a specific
Python version.
The `python` symlink in a venv is not always the same binary that created the venv.
