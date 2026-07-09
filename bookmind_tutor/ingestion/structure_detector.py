"""
StructureDetector: infers document hierarchy using a 3-tier fallback chain.

Tier 1 - Embedded PDF outline (get_toc):
    Many published books embed a bookmark tree in the PDF metadata. When present,
    this is ground truth - 100% accurate, free, no heuristic needed. We use it
    whenever the outline has at least 3 entries.

Tier 2 - LLM parse of Table of Contents page:
    When no embedded outline exists, we look for a printed "Contents" page in the
    first 20 pages and ask an LLM to parse it into structured JSON. This handles
    the variety of TOC formats (dot leaders, right-aligned page numbers, nested
    indentation) that are difficult to handle with regex alone. Requires an
    anthropic.Anthropic client to be passed at construction time; skipped otherwise.

Tier 3 - Font-size heuristic (always available as final fallback):
    Heading detection from font size deltas, bold flags, and text patterns.
    When an LLM client is provided, ambiguous bold-only detections are sent to
    the LLM in a single batch call for verification before the tree is built.

See README.md for a detailed description of the heuristic and its known limitations.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import TYPE_CHECKING

from .models import DocumentNode, DocumentTree, PageContent, TextSpan
from bookmind_tutor.llm.base import LLMClient


# Pre-compiled patterns for text-based heading detection (secondary signal).
_RE_CHAPTER = re.compile(
    r"^(Chapter|CHAPTER|Part|PART|Appendix|APPENDIX)\s+(\d+|[IVXLCDM]+|[A-Z])\b",
)
_RE_NUMBERED_H1 = re.compile(r"^\d+\.\s+\S")        # "1. Introduction"
_RE_NUMBERED_H2 = re.compile(r"^\d+\.\d+\.?\s+\S")  # "1.1 Background"
_RE_NUMBERED_H3 = re.compile(r"^\d+\.\d+\.\d+\.?\s+\S")  # "1.1.1 Detail"
_RE_ALL_CAPS = re.compile(r"^[A-Z][A-Z\s\d\-:&]+$")

_TOC_KEYWORDS = ("contents", "table of contents")
_LLM_MODEL = "claude-haiku-4-5-20251001"

# Matches decorative chapter/part labels that carry no title text of their own,
# e.g. "CHAPTER 1", "PART II", "APPENDIX A". These are filtered out in Tier 3
# post-processing because they are visual labels, not semantic section headings.
_RE_LABEL_ONLY = re.compile(
    r"^(CHAPTER|PART|APPENDIX|APPENDICES|SECTION|UNIT)\s+(\d+|[IVXLCDM]+|[A-Z])\s*$",
    re.IGNORECASE,
)


class StructureDetector:
    """
    Converts a list of PageContent objects into a DocumentTree via a 3-tier chain.

    All thresholds are configurable via constructor args so they can be tuned
    for different document styles without touching the logic.

    Args:
        min_heading_size_ratio: A span-line must have size >= body_size * this
            to be considered a font-size heading. Default 1.15 (15% larger).
        bold_as_section: Treat bold same-size lines as (ambiguous) section headings.
        max_heading_words: Upper bound on heading word count.
        max_heading_chars: Upper bound on heading character count.
        line_tolerance_pts: Two spans are on the same "line" if their vertical
            centers differ by at most this many PDF points.
        min_heading_pages: A heading font size must appear on at least this many
            distinct pages to enter the level map (filters cover-page-only fonts).
        llm_client: Optional LLMClient instance. When provided, enables
            Tier 2 (LLM TOC parsing) and LLM verification of ambiguous headings.
    """

    def __init__(
        self,
        min_heading_size_ratio: float = 1.15,
        bold_as_section: bool = True,
        max_heading_words: int = 15,
        max_heading_chars: int = 150,
        line_tolerance_pts: float = 3.0,
        min_heading_pages: int = 3,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.min_heading_size_ratio = min_heading_size_ratio
        self.bold_as_section = bold_as_section
        self.max_heading_words = max_heading_words
        self.max_heading_chars = max_heading_chars
        self.line_tolerance_pts = line_tolerance_pts
        self.min_heading_pages = min_heading_pages
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, pages: list[PageContent], toc: list[list] | None = None
    ) -> DocumentTree:
        """
        Build a DocumentTree using the 3-tier fallback chain.

        Args:
            pages: Extracted page content from PdfExtractor.
            toc: Optional embedded PDF outline from PdfExtractor.extract_toc().
                 Format: [[level, title, page_number], ...] (level is 1-based).
                 When provided and has >= 3 entries, used directly (Tier 1).

        Returns:
            A DocumentTree. Never raises - falls back to a flat document when
            no structure can be detected.
        """
        if not pages:
            return DocumentTree(root_nodes=[])

        # Tier 1: embedded PDF outline (free, 100% accurate when present)
        if toc and len(toc) >= 3:
            tree = self._build_tree_from_toc(toc, pages)
            if tree is not None:
                return tree

        # Tier 2: LLM parse of printed TOC page.
        # Attempted before font-size analysis because it doesn't require body_size
        # and is more reliable than the heuristic when a TOC page exists.
        if self._llm is not None:
            toc_pages = self._find_toc_pages(pages)
            if toc_pages:
                parsed_toc = self._parse_toc_page_with_llm(toc_pages)
                if parsed_toc and len(parsed_toc) >= 3:
                    offset = self._detect_page_offset(parsed_toc, pages)
                    if offset != 0:
                        parsed_toc = [
                            [lvl, title, page + offset]
                            for lvl, title, page in parsed_toc
                        ]
                    tree = self._build_tree_from_toc(parsed_toc, pages)
                    if tree is not None:
                        return tree

        # Tier 3: font-size heuristic (with optional LLM verification of ambiguous lines)
        body_size = self._estimate_body_size(pages)
        if body_size is None:
            return self._flat_document(pages)

        heading_level_map = self._build_heading_level_map(pages, body_size)
        return self._build_tree(pages, body_size, heading_level_map)

    # ------------------------------------------------------------------
    # Tier 1: TOC-driven tree building
    # ------------------------------------------------------------------

    def _build_tree_from_toc(
        self, toc: list[list], pages: list[PageContent]
    ) -> DocumentTree | None:
        """
        Build a DocumentTree from a pymupdf get_toc() style outline.

        toc format: [[level, title, start_page], ...] where level is 1-based.

        Page ranges: each node's range ends just before the next sibling or
        ancestor at the same/higher level starts. The last node extends to the
        last page. Raw text is assigned to each node from the pages it owns
        (pages within its range but not covered by any child's range), so that
        parent and child nodes never duplicate text in the chunker.
        """
        if not toc:
            return None

        last_page = pages[-1].page_number if pages else 1

        # Step 1: create nodes (convert 1-based level to 0-based)
        nodes: list[DocumentNode] = []
        for entry in toc:
            level, title, start_page = entry[0], entry[1], entry[2]
            nodes.append(DocumentNode(
                title=str(title).strip(),
                level=level - 1,
                page_range=(start_page, last_page),
                children=[],
                raw_text="",
            ))

        # Step 2: refine end pages - a node ends just before the next
        # entry at the same or higher level (lower level number).
        # Clamp end >= start: two TOC entries on the same page cause
        # next_start - 1 < start, which would invert the range.
        for i, node in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                if nodes[j].level <= node.level:
                    end = max(node.page_range[0], nodes[j].page_range[0] - 1)
                    node.page_range = (node.page_range[0], end)
                    break

        # Step 3: build tree structure via node stack
        root_nodes: list[DocumentNode] = []
        stack: list[DocumentNode] = []
        for node in nodes:
            while stack and stack[-1].level >= node.level:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                root_nodes.append(node)
            stack.append(node)

        if not root_nodes:
            return None

        # Step 4: assign raw_text from page content.
        # Each page is assigned to exactly one node: the first node in DFS order
        # whose range includes the page and whose own children do not cover it.
        # The global `claimed` set prevents sibling nodes that share a start page
        # from both receiving the same page's text, which would produce duplicate
        # chunks in the RAG pipeline.
        page_map = {p.page_number: p.text for p in pages}
        claimed: set[int] = set()
        for node in self._flatten_tree(root_nodes):
            child_pages: set[int] = set()
            for child in node.children:
                child_pages.update(range(child.page_range[0], child.page_range[1] + 1))
            own_pages = [
                pn for pn in range(node.page_range[0], node.page_range[1] + 1)
                if pn not in child_pages and pn not in claimed and pn in page_map
            ]
            node.raw_text = "\n\n".join(
                page_map[pn] for pn in own_pages if page_map[pn].strip()
            )
            claimed.update(own_pages)

        return DocumentTree(root_nodes=root_nodes)

    @staticmethod
    def _flatten_tree(root_nodes: list[DocumentNode]) -> list[DocumentNode]:
        """DFS traversal returning all nodes in the tree."""
        result: list[DocumentNode] = []

        def _dfs(node: DocumentNode) -> None:
            result.append(node)
            for child in node.children:
                _dfs(child)

        for node in root_nodes:
            _dfs(node)
        return result

    # ------------------------------------------------------------------
    # Tier 2: LLM parse of printed TOC page
    # ------------------------------------------------------------------

    def _find_toc_page(self, pages: list[PageContent]) -> PageContent | None:
        """Return the first TOC page, or None. Delegates to _find_toc_pages."""
        result = self._find_toc_pages(pages)
        return result[0] if result else None

    def _find_toc_pages(self, pages: list[PageContent]) -> list[PageContent]:
        """
        Find the printed TOC page and any immediately following continuation pages.

        Returns a list of consecutive pages that together form the full TOC, up to
        a maximum of 10 pages. Stops when a page no longer looks like TOC content
        (i.e. it appears to be the start of body text).
        """
        result: list[PageContent] = []
        for page in pages[:20]:
            start = page.text.strip()[:50].lower()
            if not result:
                if any(start.startswith(kw) for kw in _TOC_KEYWORDS):
                    result.append(page)
            else:
                if len(result) >= 10:
                    break
                if self._looks_like_toc_continuation(page):
                    result.append(page)
                else:
                    break
        return result

    @staticmethod
    def _looks_like_toc_continuation(page: PageContent) -> bool:
        """
        Return True if this page looks like a continuation of a printed TOC.

        Two signals:
        - Body text starts with long sentences (first non-empty line > 80 chars).
          If detected, the TOC is over.
        - TOC pages have many lines ending with a digit (right-aligned page numbers).
          We require at least 2 such lines as the positive signal.
        """
        lines = [ln.strip() for ln in page.text.splitlines() if ln.strip()]
        if not lines:
            return False
        # Body text: first line is long prose NOT ending with a digit.
        # A long TOC entry (e.g. "Relational Versus Document Databases Today  38")
        # ends with a page number, so we keep it.
        first = lines[0]
        if len(first) > 80 and not first[-1].isdigit():
            return False
        # TOC pages have many lines ending with a digit (right-aligned page numbers)
        ends_with_digit = sum(1 for ln in lines if ln[-1].isdigit())
        return ends_with_digit >= 2

    def _parse_toc_page_with_llm(
        self, pages: list[PageContent]
    ) -> list[list] | None:
        """
        Use an LLM to parse one or more printed TOC pages into [[level, title, page], ...].

        Accepts a list of consecutive TOC pages so that multi-page tables of contents
        are fully captured. Text is concatenated and truncated to 8000 chars before
        being sent to the LLM.

        TOC formatting varies widely between publishers (dot leaders, right-aligned
        page numbers, indentation). An LLM handles this variety better than regex.
        Returns None if the LLM fails or yields fewer than 3 entries.
        """
        assert self._llm is not None
        combined_text = "\n".join(p.text for p in pages)
        prompt = (
            "Parse this Table of Contents into a JSON array.\n"
            'Each entry must be: {"level": 1, "title": "...", "page": 5}\n'
            "level: 1 = chapter or part, 2 = section, 3 = subsection.\n"
            "Return ONLY a valid JSON array, nothing else. If the text is not a TOC, return [].\n\n"
            f"TOC text:\n{combined_text[:8000]}"
        )
        try:
            response = self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
            )
            raw = response.text.strip()
            # Strip markdown code fences (```json ... ```) that some models add
            # despite the "Return ONLY a valid JSON array" instruction.
            start, end = raw.find("["), raw.rfind("]")
            if start != -1 and end != -1:
                raw = raw[start : end + 1]
            entries = json.loads(raw)
            result = [
                [e["level"], e["title"], e["page"]]
                for e in entries
                if isinstance(e, dict)
                and all(k in e for k in ("level", "title", "page"))
                and isinstance(e["page"], int)  # skip Part headers with page=None
            ]
            return result if len(result) >= 3 else None
        except Exception:
            # Any failure (API error, JSON parse error, network) falls through to Tier 3
            return None

    @staticmethod
    def _detect_page_offset(
        toc: list[list], pages: list[PageContent]
    ) -> int:
        """
        Estimate the difference between PDF page numbers and printed book page numbers.

        Printed TOC entries use the page numbers visible at the bottom of each page
        (which often start from 1 at Chapter 1, with roman numerals for front matter).
        PDF page numbers start from 1 at the very first page of the file, so there is
        typically an offset equal to the number of front-matter pages.

        Strategy: for each TOC entry, search all PDF pages for a page whose text
        contains the entry title. Collect (pdf_page - toc_page) for each match and
        return the most common offset. Falls back to 0 if nothing matches.
        """
        offsets: list[int] = []
        # Skip the first 20 pages - they contain the TOC itself, so title text
        # found there would produce a wrong (small) offset instead of the real one.
        page_texts = {
            p.page_number: p.text for p in pages if p.page_number > 20
        }

        for _, title, toc_page in toc:
            if not isinstance(toc_page, int) or not title:
                continue
            title_lower = title.lower().strip()
            if len(title_lower) < 5:
                continue
            for pdf_page, text in sorted(page_texts.items()):
                if title_lower in text.lower():
                    offsets.append(pdf_page - toc_page)
                    break

        if not offsets:
            return 0
        # Return the most common offset (mode)
        return max(set(offsets), key=offsets.count)

    # ------------------------------------------------------------------
    # Font size analysis (shared by Tier 3 and body size estimation)
    # ------------------------------------------------------------------

    def _estimate_body_size(self, pages: list[PageContent]) -> float | None:
        """
        Find the dominant (body text) font size by total character weight.

        We round sizes to the nearest 0.5pt to group visually identical sizes
        that may differ by sub-point rounding in the PDF renderer.
        Sizes below 6pt are excluded (likely subscripts/footnotes).
        """
        size_char_count: dict[float, int] = defaultdict(int)
        for page in pages:
            for span in page.spans:
                if span.size < 6:
                    continue
                bucket = round(span.size * 2) / 2
                size_char_count[bucket] += len(span.text)

        if not size_char_count:
            return None

        return max(size_char_count, key=lambda s: size_char_count[s])

    def _build_heading_level_map(
        self, pages: list[PageContent], body_size: float
    ) -> dict[float, int]:
        """
        Collect distinct font sizes that are large enough to be headings and
        map them to heading levels.

        Sizes are rounded to 0.5pt buckets (same as body size estimation).
        Larger size -> lower level number (0 = most prominent, like chapter).

        Sizes that appear on fewer than the effective threshold of distinct pages
        are excluded. This filters out cover-page-only fonts (e.g. a 72pt title
        that appears on exactly 1 page) which would otherwise dominate the
        hierarchy and push all real chapter headings to level 2.

        The threshold is adaptive: for short documents (tests, excerpts) it is
        lowered so that a single-page heading is still detected.
        """
        size_pages: dict[float, set[int]] = defaultdict(set)
        threshold = body_size * self.min_heading_size_ratio

        for page in pages:
            for span in page.spans:
                bucket = round(span.size * 2) / 2
                if bucket > threshold:
                    size_pages[bucket].add(page.page_number)

        # Adaptive minimum: require more pages for longer documents.
        # For 1-59 pages: effective_min = 1  (short docs / tests)
        # For 60-119 pages: effective_min = 2
        # For 120+ pages: effective_min = min_heading_pages (default 3)
        effective_min = min(self.min_heading_pages, max(1, len(pages) // 40))

        heading_sizes = {
            size for size, pgs in size_pages.items() if len(pgs) >= effective_min
        }
        sorted_sizes = sorted(heading_sizes, reverse=True)
        return {size: min(i, 2) for i, size in enumerate(sorted_sizes)}

    # ------------------------------------------------------------------
    # Line grouping and classification (Tier 3)
    # ------------------------------------------------------------------

    def _group_spans_into_lines(
        self, spans: list[TextSpan]
    ) -> list[list[TextSpan]]:
        """
        Group consecutive spans into visual lines using vertical proximity.

        Spans are expected to arrive in reading order (as produced by
        PdfExtractor with sort=True). Two spans belong to the same line
        if their vertical centers are within line_tolerance_pts of each other.
        """
        if not spans:
            return []

        lines: list[list[TextSpan]] = []
        current: list[TextSpan] = [spans[0]]

        for span in spans[1:]:
            prev_center = (current[-1].bbox[1] + current[-1].bbox[3]) / 2
            curr_center = (span.bbox[1] + span.bbox[3]) / 2
            if abs(curr_center - prev_center) <= self.line_tolerance_pts:
                current.append(span)
            else:
                lines.append(current)
                current = [span]

        lines.append(current)
        return lines

    def _classify_line(
        self,
        line_spans: list[TextSpan],
        body_size: float,
        heading_level_map: dict[float, int],
    ) -> int | None:
        """
        Returns heading level (0, 1, 2) or None if the line is body text.

        Public interface; delegates to _classify_line_with_confidence.
        """
        level, _ = self._classify_line_with_confidence(
            line_spans, body_size, heading_level_map
        )
        return level

    def _classify_line_with_confidence(
        self,
        line_spans: list[TextSpan],
        body_size: float,
        heading_level_map: dict[float, int],
    ) -> tuple[int | None, bool]:
        """
        Returns (heading_level, is_ambiguous).

        is_ambiguous is True when the detection relies solely on Signal 3
        (bold at body font size). These are candidates for LLM verification
        because bold text is often just emphasis, not a heading.

        Signal 1 (font size map) and Signal 2 (text patterns) are considered
        high-confidence and return is_ambiguous=False.
        """
        if not line_spans:
            return None, False

        line_text = "".join(s.text for s in line_spans).strip()
        if not line_text:
            return None, False

        if (len(line_text.split()) > self.max_heading_words
                or len(line_text) > self.max_heading_chars):
            return None, False

        dominant_size = max(s.size for s in line_spans)
        size_bucket = round(dominant_size * 2) / 2

        # Signal 1: font size matches a pre-computed heading level (high confidence)
        if size_bucket in heading_level_map:
            return heading_level_map[size_bucket], False

        # Signal 2: text matches a common heading pattern.
        # When the document has font-size headings (heading_level_map is non-empty),
        # require the same size threshold as Signal 1 so that figure captions and
        # sidebar labels slightly above body size are not mistaken for headings.
        # When the document has no font variation (uniform-font PDF), apply pattern
        # matching at any size as a last-resort.
        word_count = len(line_text.split())
        size_ok = (
            dominant_size >= body_size * self.min_heading_size_ratio
            if heading_level_map else True
        )
        if size_ok and word_count <= 8:
            level = self._pattern_level(line_text)
            if level is not None:
                return level, False

        # Signal 3 (optional): bold same-size as body.
        # Flagged as ambiguous because bold text is commonly used for emphasis
        # in body paragraphs, not just headings.
        if self.bold_as_section and dominant_size >= body_size * 0.9:
            is_bold = any(
                "Bold" in s.font or "bold" in s.font or "Heavy" in s.font
                for s in line_spans
            )
            if is_bold and len(line_text.split()) <= 8:
                return 2, True  # is_ambiguous=True

        return None, False

    def _pattern_level(self, text: str) -> int | None:
        """Map text patterns to heading levels (independent of font size)."""
        stripped = text.strip()

        if _RE_CHAPTER.match(stripped) and not stripped.endswith(".") and ")" not in stripped:
            return 0

        if _RE_NUMBERED_H3.match(stripped):
            return 2
        if _RE_NUMBERED_H2.match(stripped):
            return 1
        if _RE_NUMBERED_H1.match(stripped) and not stripped.endswith("."):
            return 0

        if _RE_ALL_CAPS.match(stripped) and len(stripped) <= 60 and len(stripped.split()) <= 6:
            return 0
        return None

    # ------------------------------------------------------------------
    # Tier 3: heuristic tree building (two-phase when LLM available)
    # ------------------------------------------------------------------

    def _build_tree(
        self,
        pages: list[PageContent],
        body_size: float,
        heading_level_map: dict[float, int],
    ) -> DocumentTree:
        # Phase 1: classify every line in the document
        # Tuple: (page, line_spans, line_text, level, is_ambiguous)
        classified: list[tuple] = []
        for page in pages:
            for line_spans in self._group_spans_into_lines(page.spans):
                line_text = "".join(s.text for s in line_spans).strip()
                if not line_text:
                    continue
                level, is_ambiguous = self._classify_line_with_confidence(
                    line_spans, body_size, heading_level_map
                )
                classified.append((page, line_spans, line_text, level, is_ambiguous))

        # Phase 2: batch-verify ambiguous (bold-only) headings with LLM
        if self._llm is not None:
            classified = self._verify_ambiguous_lines(classified)

        # Phase 3: build tree from resolved classifications
        root_nodes: list[DocumentNode] = []
        stack: list[DocumentNode] = []

        for page, _, line_text, level, _ in classified:
            if level is not None:
                while stack and stack[-1].level >= level:
                    stack.pop()

                new_node = DocumentNode(
                    title=line_text,
                    level=level,
                    page_range=(page.page_number, page.page_number),
                    children=[],
                    raw_text="",
                )

                if stack:
                    stack[-1].children.append(new_node)
                else:
                    root_nodes.append(new_node)

                stack.append(new_node)
                self._extend_page_range(stack, page.page_number)
            else:
                if stack:
                    stack[-1].raw_text += line_text + "\n"
                    self._extend_page_range(stack, page.page_number)
                else:
                    if not root_nodes or root_nodes[-1].level != -1:
                        preamble = DocumentNode(
                            title=None,
                            level=-1,
                            page_range=(page.page_number, page.page_number),
                            children=[],
                            raw_text="",
                        )
                        root_nodes.append(preamble)
                        stack.append(preamble)
                    stack[-1].raw_text += line_text + "\n"
                    self._extend_page_range(stack, page.page_number)

        if not root_nodes:
            return self._flat_document(pages)

        root_nodes = self._post_process_tree3(root_nodes)
        return DocumentTree(root_nodes=root_nodes)

    def _post_process_tree3(
        self, nodes: list[DocumentNode]
    ) -> list[DocumentNode]:
        """
        Post-process a Tier 3 tree bottom-up to reduce two categories of noise:

        1. Label-only nodes: decorative chapter/part labels like "CHAPTER 1" that
           carry no title text of their own. The font-size heuristic detects them as
           headings (they use a distinct font), but they add nothing useful to the
           hierarchy and confuse downstream chunking.

        2. Split titles: a single long heading that wraps across two printed lines
           produces two consecutive same-level sibling nodes on the same page.
           For example "Reliable, Scalable, and" followed by "Maintainable
           Applications". Both are merged into one node with the combined title.
           Merge condition: same level, same start page, first title has <= 6 words
           and does not end with sentence-closing punctuation.
        """
        for node in nodes:
            node.children = self._post_process_tree3(node.children)
        return self._merge_and_filter_siblings(nodes)

    @staticmethod
    def _is_title_fragment(title: str) -> bool:
        """
        Return True if a heading title looks like an incomplete line that wrapped.

        A wrapped line ends mid-phrase: with a comma, or with a coordinating
        conjunction ("and", "or", "but"). A standalone heading like
        "Comparison to document databases" ends with a noun — not a fragment.
        We cap at 8 words so we don't accidentally merge two long headings that
        happen to share a page.
        """
        t = title.rstrip().lower()
        return (
            len(title.split()) <= 8
            and (
                t.endswith(",")
                or t.endswith(" and")
                or t.endswith(" or")
                or t.endswith(" but")
            )
        )

    def _merge_and_filter_siblings(
        self, siblings: list[DocumentNode]
    ) -> list[DocumentNode]:
        """Apply label filtering then split-title merging to one sibling list."""
        # Step 1: remove decorative label-only nodes (e.g. "CHAPTER 1", "PART II").
        # Only filter when the node has no content and no children: a node titled
        # "Chapter 1" that actually carries body text is a legitimate heading, not
        # a decorative label.
        filtered = [
            n for n in siblings
            if not (
                n.title
                and _RE_LABEL_ONLY.match(n.title.strip())
                and not n.raw_text.strip()
                and not n.children
            )
        ]

        # Step 2: merge consecutive same-level siblings on the same page whose
        # first title looks like an incomplete fragment (no sentence-ending
        # punctuation and short enough to be a wrapped line, not a standalone heading).
        result: list[DocumentNode] = []
        i = 0
        while i < len(filtered):
            node = filtered[i]
            next_node = filtered[i + 1] if i + 1 < len(filtered) else None
            if (
                next_node is not None
                and next_node.level == node.level
                and next_node.page_range[0] == node.page_range[0]
                and node.title
                and self._is_title_fragment(node.title)
            ):
                sep = "\n\n" if node.raw_text and next_node.raw_text else ""
                merged = DocumentNode(
                    title=node.title.rstrip() + " " + next_node.title.lstrip(),
                    level=node.level,
                    page_range=(node.page_range[0], next_node.page_range[1]),
                    children=node.children + next_node.children,
                    raw_text=node.raw_text + sep + next_node.raw_text,
                )
                result.append(merged)
                i += 2
            else:
                result.append(node)
                i += 1
        return result

    # ------------------------------------------------------------------
    # LLM verification of ambiguous heading candidates (Tier 3 enhancement)
    # ------------------------------------------------------------------

    def _verify_ambiguous_lines(self, classified: list[tuple]) -> list[tuple]:
        """
        Replace ambiguous (bold-only) heading detections with LLM-verified results.

        Collects all is_ambiguous=True lines, sends them to the LLM in one batch
        call, then substitutes the verified level back into the classified list.
        Lines where the LLM says "not a heading" get level=None (body text).
        """
        ambiguous_indices = [
            i for i, (_, _, _, level, is_amb) in enumerate(classified)
            if is_amb and level is not None
        ]
        if not ambiguous_indices:
            return classified

        candidates = [
            (classified[i][2], classified[i][3])  # (line_text, heuristic_level)
            for i in ambiguous_indices
        ]
        verified_levels = self._verify_candidates_with_llm(candidates)

        result = list(classified)
        for idx, new_level in zip(ambiguous_indices, verified_levels):
            page, spans, text, _, _ = result[idx]
            result[idx] = (page, spans, text, new_level, False)
        return result

    def _verify_candidates_with_llm(
        self, candidates: list[tuple[str, int | None]]
    ) -> list[int | None]:
        """
        Ask the LLM whether each bold-only line candidate is a real section heading.

        Sends all candidates in a single API call to minimize latency and cost.
        Returns a list of resolved levels (or None for body text), same length as input.
        Falls back to the heuristic level if the LLM response is malformed.

        candidates: [(line_text, heuristic_level), ...]
        """
        assert self._llm is not None

        items = "\n".join(
            f'{i + 1}. {text!r}  (heuristic: level {lvl})'
            for i, (text, lvl) in enumerate(candidates)
        )
        prompt = (
            "You are reviewing lines of text from a technical book.\n"
            "The heuristic flagged these lines as POSSIBLY section headings "
            "(bold text at body font size - ambiguous).\n\n"
            "For each line, decide: is it a true section heading or just emphasized body text?\n"
            "Respond ONLY with a JSON array:\n"
            '[{"index": 1, "is_heading": true, "level": 2}, ...]\n'
            "level: 0=chapter, 1=section, 2=subsection. null if not a heading.\n\n"
            f"Lines:\n{items}"
        )
        try:
            response = self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            raw = response.text.strip()
            start, end = raw.find("["), raw.rfind("]")
            if start != -1 and end != -1:
                raw = raw[start : end + 1]
            data = json.loads(raw)
            level_map = {
                r["index"] - 1: (r["level"] if r.get("is_heading") else None)
                for r in data
                if isinstance(r, dict) and "index" in r
            }
            return [level_map.get(i, candidates[i][1]) for i in range(len(candidates))]
        except Exception:
            # Any failure (API error, JSON parse error) falls back to heuristic
            return [lvl for _, lvl in candidates]

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extend_page_range(stack: list[DocumentNode], page_number: int) -> None:
        """Extend the page_range end of every open ancestor to page_number."""
        for node in stack:
            if node.page_range[1] < page_number:
                node.page_range = (node.page_range[0], page_number)

    def _flat_document(self, pages: list[PageContent]) -> DocumentTree:
        """Fallback: return a single root node containing all page text."""
        all_text = "\n\n".join(p.text for p in pages)
        page_range = (
            (pages[0].page_number, pages[-1].page_number) if pages else (1, 1)
        )
        root = DocumentNode(
            title=None,
            level=0,
            page_range=page_range,
            children=[],
            raw_text=all_text,
        )
        return DocumentTree(root_nodes=[root])
