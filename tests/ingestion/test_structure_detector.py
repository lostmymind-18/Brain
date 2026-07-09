"""
Unit tests for StructureDetector using synthetic PageContent objects.

We construct PageContent directly (no real PDF needed) so tests are fast
and deterministic. The key thing being tested is the heuristic logic:
- body size estimation
- font-size-based heading classification
- pattern-based heading classification
- tree building with correct parent-child relationships
- fallback to flat document when no headings detected
"""
from __future__ import annotations

import pytest

from bookmind_tutor.ingestion import (
    DocumentNode,
    DocumentTree,
    PageContent,
    StructureDetector,
    TextSpan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def span(text: str, size: float, font: str = "Times-Roman", x: float = 50, y: float = 50) -> TextSpan:
    """Shorthand to create a TextSpan with a simple bbox."""
    return TextSpan(text=text, font=font, size=size, bbox=(x, y, x + len(text) * size * 0.6, y + size))


def page(page_number: int, spans_list: list[TextSpan]) -> PageContent:
    """Build a PageContent from a list of spans (text is derived from spans)."""
    text = " ".join(s.text for s in spans_list)
    return PageContent(page_number=page_number, text=text, spans=spans_list)


def make_page_with_structure(page_number: int = 1) -> PageContent:
    """
    A page with: a 20pt heading, 12pt body, then a 14pt section, then 12pt body.
    Body size should be detected as 12pt.
    """
    y = 50
    spans_list = [
        span("Introduction", size=20.0, y=y),
        span("This is the first paragraph of body text.", size=12.0, y=y + 30),
        span("It continues here with more sentences.", size=12.0, y=y + 50),
        span("1.1 Background", size=14.0, y=y + 80),
        span("The background section contains details.", size=12.0, y=y + 110),
    ]
    return page(page_number, spans_list)


# ---------------------------------------------------------------------------
# Body size estimation
# ---------------------------------------------------------------------------

class TestBodySizeEstimation:
    def test_dominant_size_is_body(self):
        detector = StructureDetector()
        # Lots of 12pt text, a few 18pt spans
        spans_list = (
            [span(f"word{i}", size=12.0, y=50 + i * 15) for i in range(20)]
            + [span("Big heading", size=18.0, y=400)]
        )
        pg = page(1, spans_list)
        body = detector._estimate_body_size([pg])
        assert body == pytest.approx(12.0, abs=0.5)

    def test_empty_pages_returns_none(self):
        detector = StructureDetector()
        assert detector._estimate_body_size([]) is None

    def test_no_spans_returns_none(self):
        detector = StructureDetector()
        pg = PageContent(page_number=1, text="", spans=[])
        assert detector._estimate_body_size([pg]) is None


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

class TestHeadingClassification:
    def setup_method(self):
        self.detector = StructureDetector(min_heading_size_ratio=1.15)

    def _level_map(self, body: float, sizes: list[float]) -> dict[float, int]:
        return {s: i for i, s in enumerate(sorted(sizes, reverse=True))}

    def test_large_font_classified_as_heading(self):
        line = [span("Chapter One", size=20.0)]
        body = 12.0
        level_map = self._level_map(body, [20.0])
        result = self.detector._classify_line(line, body, level_map)
        assert result == 0

    def test_body_size_not_classified_as_heading(self):
        line = [span("Regular body sentence here.", size=12.0)]
        body = 12.0
        level_map = {}
        result = self.detector._classify_line(line, body, level_map)
        assert result is None

    def test_too_many_words_excluded(self):
        long_heading = " ".join(["word"] * 20)
        line = [span(long_heading, size=20.0)]
        body = 12.0
        level_map = self._level_map(body, [20.0])
        result = self.detector._classify_line(line, body, level_map)
        assert result is None

    def test_pattern_chapter_detected(self):
        line = [span("Chapter 3: Advanced Topics", size=12.0)]
        result = self.detector._classify_line(line, 12.0, {})
        assert result == 0

    def test_pattern_numbered_section_detected(self):
        line = [span("2.1 Background", size=12.0)]
        result = self.detector._classify_line(line, 12.0, {})
        assert result == 1

    def test_two_heading_levels_get_different_levels(self):
        body = 12.0
        level_map = self._level_map(body, [20.0, 16.0])
        h1_line = [span("Big Heading", size=20.0)]
        h2_line = [span("Smaller Heading", size=16.0)]
        assert self.detector._classify_line(h1_line, body, level_map) == 0
        assert self.detector._classify_line(h2_line, body, level_map) == 1


# ---------------------------------------------------------------------------
# Tree building
# ---------------------------------------------------------------------------

class TestTreeBuilding:
    def test_headings_become_root_nodes(self):
        pg = make_page_with_structure()
        detector = StructureDetector(min_heading_size_ratio=1.15)
        tree = detector.detect([pg])

        assert isinstance(tree, DocumentTree)
        assert len(tree.root_nodes) >= 1
        # The 20pt "Introduction" should be the top-level node
        assert tree.root_nodes[0].title == "Introduction"

    def test_section_is_child_of_chapter(self):
        pg = make_page_with_structure()
        detector = StructureDetector(min_heading_size_ratio=1.15)
        tree = detector.detect([pg])

        chapter = tree.root_nodes[0]
        # "1.1 Background" (14pt) should be a child of "Introduction" (20pt)
        assert len(chapter.children) >= 1
        child_titles = [c.title for c in chapter.children]
        assert any("1.1" in t for t in child_titles if t)

    def test_body_text_in_correct_node(self):
        pg = make_page_with_structure()
        detector = StructureDetector(min_heading_size_ratio=1.15)
        tree = detector.detect([pg])

        chapter = tree.root_nodes[0]
        # Body text directly under "Introduction" (before 1.1)
        assert "paragraph" in chapter.raw_text.lower() or any(
            "paragraph" in c.raw_text.lower() for c in chapter.children
        )

    def test_page_range_set(self):
        pg = make_page_with_structure(page_number=3)
        detector = StructureDetector(min_heading_size_ratio=1.15)
        tree = detector.detect([pg])

        for node in tree.root_nodes:
            assert node.page_range[0] == 3
            assert node.page_range[1] >= 3

    def test_no_headings_produces_flat_document(self):
        """No headings -> single root node with all text (graceful fallback)."""
        all_body = [span(f"Sentence number {i}.", size=12.0, y=50 + i * 15) for i in range(10)]
        pg = page(1, all_body)
        detector = StructureDetector(min_heading_size_ratio=1.15)
        tree = detector.detect([pg])

        assert len(tree.root_nodes) == 1
        assert tree.root_nodes[0].title is None or len(tree.root_nodes) == 1

    def test_empty_pages_returns_empty_tree(self):
        tree = StructureDetector().detect([])
        assert tree.root_nodes == []

    def test_multi_page_chapter_spans_correct_range(self):
        spans_p1 = [span("Chapter 1", size=20.0, y=50), span("Text on page 1.", size=12.0, y=80)]
        spans_p2 = [span("More text on page 2.", size=12.0, y=50)]
        pages = [
            PageContent(page_number=1, text="Chapter 1 Text on page 1.", spans=spans_p1),
            PageContent(page_number=2, text="More text on page 2.", spans=spans_p2),
        ]
        detector = StructureDetector(min_heading_size_ratio=1.15)
        tree = detector.detect(pages)

        chapter = next(n for n in tree.root_nodes if n.title == "Chapter 1")
        assert chapter.page_range[1] == 2


# ---------------------------------------------------------------------------
# Tier 1: TOC-driven detection
# ---------------------------------------------------------------------------

class TestTocDrivenDetection:
    def _pages(self, n: int) -> list[PageContent]:
        return [
            PageContent(page_number=i, text=f"Body text on page {i}.", spans=[])
            for i in range(1, n + 1)
        ]

    def test_uses_toc_titles_when_provided(self):
        toc = [[1, "Foundation", 1], [2, "Basics", 2], [1, "Advanced", 3]]
        tree = StructureDetector().detect(self._pages(3), toc=toc)
        titles = [n.title for n in tree.root_nodes]
        assert "Foundation" in titles
        assert "Advanced" in titles

    def test_ignores_toc_with_fewer_than_3_entries(self):
        """TOC with < 3 entries is distrusted - fall back to heuristic."""
        toc = [[1, "Only One", 1]]
        pg = make_page_with_structure()
        tree = StructureDetector().detect([pg], toc=toc)
        # Heuristic should detect "Introduction" from font size
        assert any(n.title == "Introduction" for n in tree.root_nodes)

    def test_ignores_empty_toc(self):
        pg = make_page_with_structure()
        tree = StructureDetector().detect([pg], toc=[])
        assert any(n.title == "Introduction" for n in tree.root_nodes)

    def test_toc_page_ranges_end_before_next_sibling(self):
        """Chapter 1 ends on page 4 (one before Chapter 2 starts on page 5)."""
        toc = [[1, "Chapter 1", 1], [1, "Chapter 2", 5], [1, "Chapter 3", 10]]
        tree = StructureDetector().detect(self._pages(12), toc=toc)
        ch1 = next(n for n in tree.root_nodes if n.title == "Chapter 1")
        assert ch1.page_range == (1, 4)

    def test_toc_same_start_page_siblings_no_inverted_range(self):
        """Two siblings starting on the same page must not produce inverted range.

        Real books sometimes place two short TOC entries on the same page
        (e.g. 'About the Author' and 'Colophon' both at the last page, or
        two sub-sections on the same physical page). Before the fix, the first
        entry would get end = start - 1, producing an inverted tuple like (29, 28).
        """
        toc = [
            [1, "Chapter 1", 1],
            [2, "Section A", 5],
            [2, "Section B", 5],  # same start page as Section A
            [1, "Chapter 2", 8],
        ]
        tree = StructureDetector().detect(self._pages(10), toc=toc)
        ch1 = next(n for n in tree.root_nodes if n.title == "Chapter 1")
        sec_a = next(n for n in ch1.children if n.title == "Section A")
        sec_b = next(n for n in ch1.children if n.title == "Section B")
        # Neither section should have an inverted range
        assert sec_a.page_range[0] <= sec_a.page_range[1], (
            f"Section A has inverted range: {sec_a.page_range}"
        )
        assert sec_b.page_range[0] <= sec_b.page_range[1], (
            f"Section B has inverted range: {sec_b.page_range}"
        )
        # Section A is clamped to a single page (start == end == 5)
        assert sec_a.page_range == (5, 5)

    def test_last_toc_node_extends_to_last_page(self):
        toc = [[1, "Chapter 1", 1], [1, "Chapter 2", 5], [1, "Chapter 3", 8]]
        tree = StructureDetector().detect(self._pages(10), toc=toc)
        ch3 = next(n for n in tree.root_nodes if n.title == "Chapter 3")
        assert ch3.page_range[1] == 10

    def test_toc_level2_becomes_child_of_level1(self):
        toc = [
            [1, "Chapter 1", 1],
            [2, "Section 1.1", 2],
            [2, "Section 1.2", 3],
            [1, "Chapter 2", 5],
        ]
        tree = StructureDetector().detect(self._pages(6), toc=toc)
        ch1 = next(n for n in tree.root_nodes if n.title == "Chapter 1")
        assert len(ch1.children) == 2
        assert ch1.children[0].title == "Section 1.1"
        assert ch1.children[1].title == "Section 1.2"

    def test_toc_raw_text_no_overlap_between_parent_and_child(self):
        """Text from pages owned by a child must not appear in parent.raw_text."""
        toc = [[1, "Chapter 1", 1], [2, "Section 1.1", 2], [1, "Chapter 2", 3]]
        pages = [
            PageContent(page_number=1, text="Chapter intro text.", spans=[]),
            PageContent(page_number=2, text="Section body text.", spans=[]),
            PageContent(page_number=3, text="Chapter two text.", spans=[]),
        ]
        tree = StructureDetector().detect(pages, toc=toc)
        ch1 = next(n for n in tree.root_nodes if n.title == "Chapter 1")
        # Page 2 belongs to the child section; should NOT appear in ch1.raw_text
        assert "Section body text" not in ch1.raw_text
        assert "Chapter intro text" in ch1.raw_text

    def test_toc_same_page_siblings_no_duplicate_text(self):
        """Two siblings starting on the same page must not both receive that page's text.

        When two TOC entries point to the same physical page, the first sibling
        (in document order) claims the page. The second gets nothing from that page.
        Allowing both to claim it would produce duplicate chunks in the RAG pipeline.
        """
        toc = [
            [1, "Chapter 1", 1],
            [2, "Section A", 2],
            [2, "Section B", 2],  # same start page as Section A
            [1, "Chapter 2", 4],
        ]
        pages = [
            PageContent(page_number=1, text="Chapter 1 intro.", spans=[]),
            PageContent(page_number=2, text="Shared page content.", spans=[]),
            PageContent(page_number=3, text="More content.", spans=[]),
            PageContent(page_number=4, text="Chapter 2 content.", spans=[]),
        ]
        tree = StructureDetector().detect(pages, toc=toc)
        ch1 = next(n for n in tree.root_nodes if n.title == "Chapter 1")
        sec_a = next(n for n in ch1.children if n.title == "Section A")
        sec_b = next(n for n in ch1.children if n.title == "Section B")

        shared = "Shared page content."
        # Text must not appear in both siblings simultaneously
        assert not (shared in sec_a.raw_text and shared in sec_b.raw_text), (
            "Page 2 text duplicated across siblings - will produce duplicate RAG chunks"
        )
        # First sibling (Section A) gets the shared page; second gets nothing from it
        assert shared in sec_a.raw_text
        assert shared not in sec_b.raw_text


# ---------------------------------------------------------------------------
# Tier 2: LLM TOC page detection and parsing
# ---------------------------------------------------------------------------

class TestTocPageDetection:
    def test_finds_page_starting_with_contents(self):
        pages = [
            PageContent(page_number=1, text="Preface text here.", spans=[]),
            PageContent(page_number=2, text="Contents\nChapter 1...1\nChapter 2...5", spans=[]),
        ]
        result = StructureDetector()._find_toc_page(pages)
        assert result is not None
        assert result.page_number == 2

    def test_finds_table_of_contents_variant(self):
        pages = [
            PageContent(page_number=1, text="Table of Contents\nChapter 1...1", spans=[]),
        ]
        result = StructureDetector()._find_toc_page(pages)
        assert result is not None

    def test_returns_none_when_no_toc_page(self):
        pages = [
            PageContent(page_number=i, text=f"Body text on page {i}.", spans=[])
            for i in range(1, 5)
        ]
        result = StructureDetector()._find_toc_page(pages)
        assert result is None

    def test_only_scans_first_20_pages(self):
        pages = [
            PageContent(page_number=i, text=f"Body text page {i}.", spans=[])
            for i in range(1, 25)
        ]
        # Put TOC on page 22 (beyond scan window)
        pages[21] = PageContent(page_number=22, text="Contents\nChapter 1...1", spans=[])
        result = StructureDetector()._find_toc_page(pages)
        assert result is None


class TestLlmTocParsing:
    def _mock_llm(self, response_text: str):
        from unittest.mock import MagicMock
        from bookmind_tutor.llm.base import CompletionResponse, TokenUsage
        llm = MagicMock()
        llm.complete.return_value = CompletionResponse(
            text=response_text,
            stop_reason="end_turn",
            tool_calls=[],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            raw_message={"role": "assistant", "content": response_text},
        )
        return llm

    def test_parses_valid_llm_response(self):
        payload = '[{"level": 1, "title": "Chapter 1", "page": 5}, {"level": 2, "title": "Sec 1.1", "page": 7}, {"level": 1, "title": "Chapter 2", "page": 12}]'
        detector = StructureDetector(llm_client=self._mock_llm(payload))
        toc_page = PageContent(page_number=2, text="Contents\nChapter 1.....5", spans=[])
        result = detector._parse_toc_page_with_llm([toc_page])
        assert result is not None
        assert len(result) == 3
        assert result[0] == [1, "Chapter 1", 5]

    def test_returns_none_on_invalid_json(self):
        detector = StructureDetector(llm_client=self._mock_llm("not valid json"))
        toc_page = PageContent(page_number=2, text="Contents\n...", spans=[])
        result = detector._parse_toc_page_with_llm([toc_page])
        assert result is None

    def test_returns_none_when_fewer_than_3_entries(self):
        payload = '[{"level": 1, "title": "Only Chapter", "page": 1}]'
        detector = StructureDetector(llm_client=self._mock_llm(payload))
        toc_page = PageContent(page_number=2, text="Contents\n...", spans=[])
        result = detector._parse_toc_page_with_llm([toc_page])
        assert result is None

    def test_strips_markdown_fence_from_llm_response(self):
        """LLM sometimes wraps JSON in ```json...``` despite instructions."""
        payload = '```json\n[{"level": 1, "title": "Ch1", "page": 1}, {"level": 1, "title": "Ch2", "page": 5}, {"level": 1, "title": "Ch3", "page": 9}]\n```'
        detector = StructureDetector(llm_client=self._mock_llm(payload))
        toc_page = PageContent(page_number=1, text="Contents\n...", spans=[])
        result = detector._parse_toc_page_with_llm([toc_page])
        assert result is not None
        assert len(result) == 3

    def test_filters_entries_with_null_page(self):
        """Part headers without page numbers (page=null) must be skipped."""
        payload = '[{"level": 1, "title": "Part I", "page": null}, {"level": 1, "title": "Chapter 1", "page": 3}, {"level": 2, "title": "Sec 1.1", "page": 4}, {"level": 1, "title": "Chapter 2", "page": 8}]'
        detector = StructureDetector(llm_client=self._mock_llm(payload))
        toc_page = PageContent(page_number=1, text="Contents\n...", spans=[])
        result = detector._parse_toc_page_with_llm([toc_page])
        assert result is not None
        # "Part I" (page=null) must be absent; remaining 3 entries returned
        assert all(isinstance(e[2], int) for e in result)
        assert len(result) == 3

    def test_concatenates_multiple_toc_pages(self):
        """Text from all TOC pages is combined before sending to LLM."""
        payload = '[{"level": 1, "title": "Ch1", "page": 1}, {"level": 1, "title": "Ch2", "page": 5}, {"level": 1, "title": "Ch3", "page": 9}]'
        detector = StructureDetector(llm_client=self._mock_llm(payload))
        pages = [
            PageContent(page_number=1, text="Contents\nCh1...1\n2\n3\n4\n5", spans=[]),
            PageContent(page_number=2, text="Ch2...5\n6\n7\n8\n9", spans=[]),
        ]
        detector._parse_toc_page_with_llm(pages)
        # The LLM was called with concatenated text from both pages
        call_args = detector._llm.complete.call_args
        prompt_text = call_args.kwargs["messages"][0]["content"]
        assert "Ch1...1" in prompt_text
        assert "Ch2...5" in prompt_text

    def test_llm_toc_drives_tree_when_no_embedded_toc(self):
        """With llm_client set and a TOC page present, tree titles come from LLM."""
        payload = '[{"level": 1, "title": "Foundations", "page": 1}, {"level": 2, "title": "Basics", "page": 2}, {"level": 1, "title": "Applications", "page": 3}]'
        llm = self._mock_llm(payload)
        # Page 1 text starts with "Contents" to trigger TOC page detection
        pages = [
            PageContent(page_number=1, text="Contents\nFoundations...1\nApplications...3", spans=[]),
            PageContent(page_number=2, text="Basics text.", spans=[]),
            PageContent(page_number=3, text="Applications text.", spans=[]),
        ]
        tree = StructureDetector(llm_client=llm).detect(pages, toc=None)
        titles = [n.title for n in tree.root_nodes]
        assert "Foundations" in titles or "Applications" in titles


# ---------------------------------------------------------------------------
# Tier 2: multi-page TOC detection
# ---------------------------------------------------------------------------

class TestFindTocPages:
    def _make_toc_page(self, number: int, text: str) -> PageContent:
        return PageContent(page_number=number, text=text, spans=[])

    def test_returns_single_page_when_no_continuation(self):
        pages = [
            self._make_toc_page(1, "Contents\nChapter 1...1\nChapter 2...5"),
            self._make_toc_page(2, "This is the start of the preface. " * 5),
        ]
        result = StructureDetector()._find_toc_pages(pages)
        assert len(result) == 1
        assert result[0].page_number == 1

    def test_includes_continuation_pages_ending_with_numbers(self):
        pages = [
            self._make_toc_page(1, "Contents\nChapter 1...1\n2\n3"),
            self._make_toc_page(2, "Chapter 2...5\n6\n7"),
            self._make_toc_page(3, "This is body text. " * 10),
        ]
        result = StructureDetector()._find_toc_pages(pages)
        assert len(result) == 2
        assert result[1].page_number == 2

    def test_stops_at_body_text_page(self):
        long_line = "A" * 90
        pages = [
            self._make_toc_page(1, "Contents\nCh 1...1\n2\n3"),
            self._make_toc_page(2, f"{long_line}\nmore body text here"),
        ]
        result = StructureDetector()._find_toc_pages(pages)
        assert len(result) == 1

    def test_returns_empty_when_no_toc_page(self):
        pages = [
            self._make_toc_page(i, f"Body text page {i}.")
            for i in range(1, 5)
        ]
        assert StructureDetector()._find_toc_pages(pages) == []


# ---------------------------------------------------------------------------
# Tier 3: post-processing (label filter + split-title merge)
# ---------------------------------------------------------------------------

class TestTier3PostProcessing:
    def _node(self, title, level, page_start, page_end, children=None, raw_text=""):
        return DocumentNode(
            title=title, level=level,
            page_range=(page_start, page_end),
            children=children or [], raw_text=raw_text,
        )

    def test_label_only_node_is_removed(self):
        """'CHAPTER 1' without a real title should be filtered out."""
        nodes = [
            self._node("CHAPTER 1", 0, 10, 10),
            self._node("Introduction", 0, 10, 20),
        ]
        result = StructureDetector()._merge_and_filter_siblings(nodes)
        titles = [n.title for n in result]
        assert "CHAPTER 1" not in titles
        assert "Introduction" in titles

    def test_part_label_only_is_removed(self):
        nodes = [self._node("PART II", 0, 5, 5), self._node("Foundations", 0, 6, 20)]
        result = StructureDetector()._merge_and_filter_siblings(nodes)
        assert all(n.title != "PART II" for n in result)

    def test_real_heading_with_chapter_prefix_is_kept(self):
        """'Chapter 1. Introduction' has content beyond the label — keep it."""
        nodes = [self._node("Chapter 1. Introduction", 0, 1, 10)]
        result = StructureDetector()._merge_and_filter_siblings(nodes)
        assert len(result) == 1

    def test_split_title_on_same_page_is_merged(self):
        """Two same-level siblings on the same page whose first title is a fragment."""
        nodes = [
            self._node("Reliable, Scalable, and", 0, 25, 25),
            self._node("Maintainable Applications", 0, 25, 49),
        ]
        result = StructureDetector()._merge_and_filter_siblings(nodes)
        assert len(result) == 1
        assert result[0].title == "Reliable, Scalable, and Maintainable Applications"
        assert result[0].page_range == (25, 49)

    def test_no_merge_for_different_pages(self):
        """Nodes on different pages should not be merged even if same level."""
        nodes = [
            self._node("Reliability", 0, 10, 15),
            self._node("Scalability", 0, 16, 20),
        ]
        result = StructureDetector()._merge_and_filter_siblings(nodes)
        assert len(result) == 2

    def test_no_merge_when_first_title_ends_with_punctuation(self):
        """A title ending with sentence punctuation is complete — do not merge."""
        nodes = [
            self._node("Summary.", 0, 40, 42),
            self._node("References", 0, 42, 45),
        ]
        result = StructureDetector()._merge_and_filter_siblings(nodes)
        assert len(result) == 2

    def test_no_merge_when_first_title_ends_with_noun(self):
        """A title ending with a noun is a complete heading — do not merge."""
        nodes = [
            self._node("Comparison to document databases", 0, 60, 60),
            self._node("Relational Versus Document Databases Today", 0, 60, 61),
        ]
        result = StructureDetector()._merge_and_filter_siblings(nodes)
        assert len(result) == 2

    def test_merge_triggers_on_trailing_conjunction(self):
        """A title ending with 'and' is a clear fragment — merge."""
        nodes = [
            self._node("Reliable, Scalable, and", 0, 25, 25),
            self._node("Maintainable Applications", 0, 25, 49),
        ]
        result = StructureDetector()._merge_and_filter_siblings(nodes)
        assert len(result) == 1
        assert "and Maintainable" in result[0].title

    def test_merge_triggers_on_trailing_comma(self):
        """A title ending with a comma is a clear fragment — merge."""
        nodes = [
            self._node("Reliable,", 0, 25, 25),
            self._node("Scalable Applications", 0, 25, 49),
        ]
        result = StructureDetector()._merge_and_filter_siblings(nodes)
        assert len(result) == 1

    def test_post_process_applies_recursively_to_children(self):
        """Label-only nodes nested inside children must also be removed."""
        child_label = self._node("CHAPTER 2", 1, 50, 50)
        child_real = self._node("Data Models", 1, 50, 89)
        parent = self._node("Part I", 0, 25, 89, children=[child_label, child_real])
        result = StructureDetector()._post_process_tree3([parent])
        assert len(result) == 1
        child_titles = [c.title for c in result[0].children]
        assert "CHAPTER 2" not in child_titles
        assert "Data Models" in child_titles


# ---------------------------------------------------------------------------
# LLM heading verifier (Tier 3 enhancement)
# ---------------------------------------------------------------------------

class TestLlmHeadingVerifier:
    def _mock_llm(self, response_text: str):
        from unittest.mock import MagicMock
        from bookmind_tutor.llm.base import CompletionResponse, TokenUsage
        llm = MagicMock()
        llm.complete.return_value = CompletionResponse(
            text=response_text,
            stop_reason="end_turn",
            tool_calls=[],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            raw_message={"role": "assistant", "content": response_text},
        )
        return llm

    def test_bold_only_signal_is_marked_ambiguous(self):
        bold_span = span("Key Concept", size=12.0, font="Times-Bold")
        detector = StructureDetector()
        _, is_ambiguous = detector._classify_line_with_confidence([bold_span], 12.0, {})
        assert is_ambiguous is True

    def test_font_size_signal_is_not_ambiguous(self):
        big_span = span("Chapter Title", size=20.0)
        detector = StructureDetector()
        level_map = {20.0: 0}
        _, is_ambiguous = detector._classify_line_with_confidence([big_span], 12.0, level_map)
        assert is_ambiguous is False

    def test_body_text_is_not_ambiguous(self):
        body_span = span("Regular body sentence here.", size=12.0)
        detector = StructureDetector()
        _, is_ambiguous = detector._classify_line_with_confidence([body_span], 12.0, {})
        assert is_ambiguous is False

    def test_llm_confirms_bold_line_as_heading(self):
        payload = '[{"index": 1, "is_heading": true, "level": 2}]'
        detector = StructureDetector(llm_client=self._mock_llm(payload))
        result = detector._verify_candidates_with_llm([("Important Concept", 2)])
        assert result == [2]

    def test_llm_rejects_bold_line_as_body_text(self):
        payload = '[{"index": 1, "is_heading": false, "level": null}]'
        detector = StructureDetector(llm_client=self._mock_llm(payload))
        result = detector._verify_candidates_with_llm([("Note that this matters.", 2)])
        assert result == [None]

    def test_llm_verifier_falls_back_on_json_error(self):
        detector = StructureDetector(llm_client=self._mock_llm("oops not json"))
        result = detector._verify_candidates_with_llm([("Bold text", 2)])
        # Falls back to heuristic level
        assert result == [2]

    def test_no_llm_calls_when_client_is_none(self):
        """Without llm_client, ambiguous lines pass through unchanged."""
        bold_span = span("Bold Concept", size=12.0, font="Helvetica-Bold")
        pg = page(1, [bold_span, span("Body text follows.", size=12.0, y=70)])
        detector = StructureDetector(llm_client=None)
        tree = detector.detect([pg])
        # Should still build a tree (bold heading detected by heuristic)
        assert tree is not None
        assert isinstance(tree.root_nodes, list)
