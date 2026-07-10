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

## 7. Metadata is only as valuable as the layers that consume it

After the reconciler landed, 113 FoSA subsection names existed in ChromaDB metadata -
and were almost entirely wasted.
An audit of the search layer found four consumers that ignored the new field:

1. Search result labels showed only chapter/section, so the LLM could not cite at
   subsection precision.
2. `book_outline` did not list subsections, so the agent could never DISCOVER the
   names that `get_book_section` was able to match.
   The metadata was matchable but unreachable - a tool chain is only as good as its
   weakest discovery link.
3. `SourceRef` had no subsection field, so the UI Sources panel stayed chapter-level.
4. The embeddings themselves knew nothing about the headings (see lesson 8).

**Lesson:** When adding a metadata field to a pipeline, walk every downstream consumer
and ask "does this layer need the new field?"
The gap between "data exists" and "data is used" is invisible - nothing errors,
nothing warns, the system just silently underperforms.
The audit question that found all four gaps: "với metadata được cải thiện, có cần
chỉnh sửa gì ở search không?"

---

## 8. Contextual embedding: separate the retrieval representation from the display text

A chunk inside the "Broker Topology" subsection that never contains the word "broker"
is invisible to the query "broker topology" - the embedding only represents the words
in the chunk body.

Fix: prepend the heading breadcrumb ("Chapter 14 > Topologies > Broker Topology")
to the text **at embedding and BM25-tokenization time**, while storing the clean text
as the document.
ChromaDB supports this directly: pass explicit `embeddings=` computed from the context
text alongside clean `documents=` - provided embeddings are used as-is, documents are
not re-embedded.

This is a lightweight version of Anthropic's "contextual retrieval" technique.
The full version uses an LLM call per chunk to generate situating context;
here the context comes free from structure detection.

Two design points that mattered:
- **The stored text stays clean.** The breadcrumb is a retrieval representation,
  not content. Location labels are added by tools at format time; baking them into
  the text would duplicate them and pollute every downstream consumer (fact-check,
  KG extraction).
- **Budget interaction:** breadcrumb (~10-20 words) + chunk text must stay under the
  embedding model's ~195-word truncation point, so `max_tokens` went 180 -> 160.
  Every addition to the embedded text spends the same fixed budget.

Measured result: heading-name queries ("Risk Storming", "Broker Topology",
"Architecture Decision Records") all hit correctly-labeled chunks in top-3.
The "fitness function" query stopped retrieving the appendix quiz page (p.395)
and started retrieving the real content (p.103).

---

## 9. Raw similarity scores do not measure retrieval quality across index versions

Comparing average top-1 cosine scores between the old and new index (0.795 vs 0.760)
suggested a regression.
Looking at WHAT was retrieved showed the opposite:
the old index's "high-scoring" hit for "fitness functions" was the appendix
self-assessment quiz page; the new index's "lower-scoring" hit was the actual
Governance and Fitness Functions content.

Two reasons scores are incomparable across index versions:
- Different embeddings (breadcrumb changes every document vector) shift the score
  distribution wholesale.
- Hybrid fusion normalizes BM25 by the max score in the candidate set, so scores
  depend on `k` and on what else is in the index.
  (A score of exactly 0.500 with alpha=0.5 is the fingerprint of a BM25-only hit
  that dense search never surfaced - useful for debugging fusion behavior.)

**Lesson:** For retrieval changes, evaluate "did the RIGHT chunk surface" (labeled-hit
checks, content inspection), not score deltas.
Scores are only comparable within a single index version and query configuration.

---

## 10. Always verify which server instance you are testing against

Playwright E2E against localhost:8501 initially showed the OLD Sources panel
(no subsection).
Root cause: a Streamlit instance from the previous day was still running on 8501;
the newly launched instance logged "Port 8501 is not available" and died.
The test was hitting day-old code while the log quietly recorded the failure.

**Lesson:** After starting a server for E2E verification, confirm the startup log
shows a successful bind (not just that the port responds - anything could be
listening there).
An HTTP 200 from the right port is not proof that YOUR code is serving it.

---

## 11. Python venv symlink confusion

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
