"""
Unit tests for StructureReconciler.

Covers the cases from plans/structure_reconciler.md verification plan:
- exact match (candidate text == TOC title after normalize)
- punctuation variant match ("Chapter 1. Intro" vs "Chapter 1: Intro")
- false-positive rejection (symbols, short strings, fonts on < min_heading_pages pages)
- unmatched TOC entry stays as anchor node (reconciler doesn't remove it)
- fallback: no TOC (reconciler not called; existing Tier-3 behavior unchanged)
- subsection propagated to Chunk by HierarchicalChunker after reconciliation
"""
from __future__ import annotations

import pytest

from bookmind_tutor.ingestion.models import DocumentNode, DocumentTree, PageContent, TextSpan
from bookmind_tutor.ingestion.structure_reconciler import (
    HeadingCandidate,
    StructureReconciler,
    _flatten_tree,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _page(number: int, text: str, spans: list[TextSpan] | None = None) -> PageContent:
    return PageContent(page_number=number, text=text, spans=spans or [])


def _anchor(title: str, level: int, start: int, end: int, raw: str = "") -> DocumentNode:
    return DocumentNode(
        title=title,
        level=level,
        page_range=(start, end),
        children=[],
        raw_text=raw,
    )


def _cand(
    text: str,
    page: int,
    font_size: float = 10.0,
    bold: bool = False,
    y_pos: float = 100.0,
    level: int = 2,
) -> HeadingCandidate:
    return HeadingCandidate(
        text=text, page=page, font_size=font_size,
        bold=bold, y_pos=y_pos, level=level,
    )


def _reconciler(min_heading_pages: int = 1, total_pages: int = 10) -> StructureReconciler:
    """Reconciler with relaxed min_heading_pages for testing."""
    return StructureReconciler(min_heading_pages=min_heading_pages, total_pages=total_pages)


def _simple_tree(*anchors: DocumentNode) -> DocumentTree:
    return DocumentTree(root_nodes=list(anchors))


# ---------------------------------------------------------------------------
# False-positive filter
# ---------------------------------------------------------------------------

class TestFalsePositiveFilter:
    def test_symbol_candidate_rejected(self):
        """Candidate with no letters (like '∑') must be filtered out."""
        anchor = _anchor("Chapter 1", 0, 1, 10, raw="Some text.\n∑\nMore text.")
        tree = _simple_tree(anchor)
        toc = [[1, "Chapter 1", 1]]
        pages = [_page(1, "Some text.\n∑\nMore text.")]

        # '∑' has no letters, so no_letter filter rejects it
        cand = _cand("∑", page=1)
        r = _reconciler()
        result = r.reconcile(toc, [cand], tree, pages)

        # Tree unchanged - no child injected
        all_nodes = _flatten_tree(result.root_nodes)
        titles = [n.title for n in all_nodes]
        assert "∑" not in titles

    def test_single_char_candidate_rejected(self):
        """Single-character candidates are rejected by the len > 1 filter."""
        anchor = _anchor("Part I", 0, 1, 5, raw="Some content.\nA\nMore.")
        tree = _simple_tree(anchor)
        cand = _cand("A", page=1)
        pages = [_page(1, "Some content.\nA\nMore.")]

        result = _reconciler().reconcile([[1, "Part I", 1]], [cand], tree, pages)
        assert len(result.root_nodes[0].children) == 0

    def test_rare_font_rejected(self):
        """A font size that appears on fewer than min_heading_pages pages is filtered."""
        # Two pages, but the heading font only appears on page 1
        # With min_heading_pages=2, it should be rejected
        anchor = _anchor("Chapter", 0, 1, 2, raw="Laws text.\nFoo bar.")
        tree = _simple_tree(anchor)
        toc = [[1, "Chapter", 1]]
        pages = [
            _page(1, "Laws text."),
            _page(2, "Foo bar."),
        ]
        cand = _cand("Laws of Architecture", page=1, font_size=14.0)
        r = StructureReconciler(min_heading_pages=2, total_pages=2)
        result = r.reconcile(toc, [cand], tree, pages)

        assert result.root_nodes[0].children == []


# ---------------------------------------------------------------------------
# TOC-matched candidates are excluded from sub-heading injection
# ---------------------------------------------------------------------------

class TestTocMatching:
    def test_toc_matched_heading_not_injected(self):
        """A candidate whose text matches a TOC entry is NOT injected as a sub-heading."""
        # TOC: "Chapter 1. Introduction"
        # Candidate: "Chapter 1. Introduction" (same after normalize)
        # Expected: not injected as L2 child
        anchor = _anchor("Preface", 0, 1, 20, raw="Chapter 1. Introduction\nBody text.")
        tree = _simple_tree(anchor)
        toc = [[1, "Chapter 1. Introduction", 2], [1, "Preface", 1]]
        pages = [_page(1, "Chapter 1. Introduction\nBody text.")]

        cand = _cand("Chapter 1. Introduction", page=1)
        result = _reconciler().reconcile(toc, [cand], tree, pages)

        # No L2 child should appear
        assert len(result.root_nodes[0].children) == 0

    def test_punctuation_variant_also_excluded(self):
        """
        A candidate whose text matches a TOC entry after normalization is excluded.
        "Chapter 1: Introduction" normalizes to the same string as
        "Chapter 1. Introduction".
        """
        anchor = _anchor("Book", 0, 1, 20, raw="Chapter 1: Introduction\nBody.")
        tree = _simple_tree(anchor)
        toc = [[1, "Chapter 1. Introduction", 1]]  # period in TOC
        pages = [_page(1, "Chapter 1: Introduction\nBody.")]

        cand = _cand("Chapter 1: Introduction", page=1)  # colon in body
        result = _reconciler().reconcile(toc, [cand], tree, pages)

        assert len(result.root_nodes[0].children) == 0

    def test_unmatched_toc_entry_preserved(self):
        """A TOC entry with NO body match must still exist in the anchor tree."""
        # Build a tree where TOC has a chapter that has no body heading candidate
        ch1 = _anchor("Chapter 1", 1, 1, 5, raw="Body text only, no heading.")
        ch2 = _anchor("Chapter 2", 1, 6, 10, raw="More body text.")
        tree = _simple_tree(ch1, ch2)
        toc = [[1, "Chapter 1", 1], [1, "Chapter 2", 6]]
        pages = [
            _page(1, "Body text only, no heading."),
            _page(6, "More body text."),
        ]
        # No candidates at all
        result = _reconciler().reconcile(toc, [], tree, pages)

        titles = [n.title for n in result.root_nodes]
        assert "Chapter 1" in titles
        assert "Chapter 2" in titles


# ---------------------------------------------------------------------------
# Sub-heading injection
# ---------------------------------------------------------------------------

class TestSubHeadingInjection:
    def test_l2_candidate_injected_as_child(self):
        """An unmatched candidate inside an anchor's page range becomes a child."""
        # Anchor: Chapter 1 (pages 1-5)
        # Candidate on page 2: "Laws of Software Architecture"
        raw = "Intro text.\nLaws of Software Architecture\nLaw 1 text.\nLaw 2 text."
        anchor = _anchor("Chapter 1", 1, 1, 5, raw=raw)
        tree = _simple_tree(anchor)
        toc = [[1, "Chapter 1", 1]]
        pages = [_page(1, raw), _page(2, "")]

        cand = _cand("Laws of Software Architecture", page=1, font_size=12.0)
        result = _reconciler().reconcile(toc, [cand], tree, pages)

        ch = result.root_nodes[0]
        assert len(ch.children) == 1
        assert ch.children[0].title == "Laws of Software Architecture"

    def test_child_level_is_parent_plus_one(self):
        """Injected child level = parent.level + 1 (local ranking, single size)."""
        raw = "Preamble.\nSection A\nSection A body."
        anchor = _anchor("Chapter 1", 1, 1, 3, raw=raw)
        tree = _simple_tree(anchor)
        toc = [[1, "Chapter 1", 1]]
        pages = [_page(1, raw)]

        cand = _cand("Section A", page=1, font_size=11.0)
        result = _reconciler().reconcile(toc, [cand], tree, pages)

        child = result.root_nodes[0].children[0]
        assert child.level == 2  # parent level 1 + 1

    def test_text_redistributed_after_injection(self):
        """Parent raw_text is split: heading + after goes to child, before stays in parent."""
        raw = "Parent intro.\nSection Heading\nSection body text."
        anchor = _anchor("Chapter 1", 1, 1, 3, raw=raw)
        tree = _simple_tree(anchor)
        toc = [[1, "Chapter 1", 1]]
        pages = [_page(1, raw)]

        cand = _cand("Section Heading", page=1)
        result = _reconciler().reconcile(toc, [cand], tree, pages)

        ch = result.root_nodes[0]
        # Parent keeps text before the heading
        assert "Parent intro." in ch.raw_text
        # Heading itself should NOT be in parent's raw_text
        assert "Section Heading" not in ch.raw_text
        # Child has text after the heading
        assert "Section body text." in ch.children[0].raw_text

    def test_multiple_candidates_create_multiple_children(self):
        """Multiple unmatched candidates in the same anchor produce multiple children."""
        raw = (
            "Intro.\n"
            "Topic A\nA content.\n"
            "Topic B\nB content."
        )
        anchor = _anchor("Chapter 1", 1, 1, 5, raw=raw)
        tree = _simple_tree(anchor)
        toc = [[1, "Chapter 1", 1]]
        pages = [_page(1, raw)]

        cands = [
            _cand("Topic A", page=1, y_pos=100.0),
            _cand("Topic B", page=1, y_pos=200.0),
        ]
        result = _reconciler().reconcile(toc, cands, tree, pages)

        ch = result.root_nodes[0]
        assert len(ch.children) == 2
        assert ch.children[0].title == "Topic A"
        assert ch.children[1].title == "Topic B"
        assert "A content." in ch.children[0].raw_text
        assert "B content." in ch.children[1].raw_text

    def test_candidate_outside_anchor_range_not_injected(self):
        """Candidate on a page outside the anchor's range is ignored for that anchor."""
        anchor = _anchor("Chapter 1", 1, 1, 3, raw="Body text.")
        tree = _simple_tree(anchor)
        toc = [[1, "Chapter 1", 1]]
        pages = [_page(1, "Body text.")]

        # Candidate is on page 10, outside anchor range (1-3)
        cand = _cand("Some Heading", page=10)
        result = _reconciler().reconcile(toc, [cand], tree, pages)

        assert result.root_nodes[0].children == []

    def test_empty_tree_returned_unchanged(self):
        """If the anchor tree has no root nodes, return it unchanged."""
        tree = DocumentTree(root_nodes=[])
        result = _reconciler().reconcile([[1, "Ch1", 1]], [], tree, [])
        assert result.root_nodes == []

    def test_no_candidates_tree_unchanged(self):
        """With zero candidates after filtering, tree must be returned unchanged."""
        anchor = _anchor("Chapter 1", 1, 1, 5, raw="Body text.")
        tree = _simple_tree(anchor)
        result = _reconciler().reconcile([[1, "Chapter 1", 1]], [], tree, [_page(1, "Body text.")])
        assert result.root_nodes[0].children == []
        assert result.root_nodes[0].raw_text == "Body text."


# ---------------------------------------------------------------------------
# Chunk subsection propagation via HierarchicalChunker
# ---------------------------------------------------------------------------

class TestChunkSubsectionPropagation:
    def test_subsection_set_on_chunks_from_l2_node(self):
        """HierarchicalChunker sets chunk.subsection from level-2 node title."""
        from bookmind_tutor.ingestion.hierarchical_chunker import HierarchicalChunker

        chapter = DocumentNode(
            title="Chapter 1", level=1,
            page_range=(1, 10), children=[], raw_text=""
        )
        subsec = DocumentNode(
            title="Laws of Architecture", level=2,
            page_range=(2, 5), children=[], raw_text="First law. Second law."
        )
        chapter.children.append(subsec)
        tree = DocumentTree(root_nodes=[chapter])

        chunks = HierarchicalChunker().chunk(tree)
        subsec_chunks = [c for c in chunks if c.subsection == "Laws of Architecture"]
        assert len(subsec_chunks) > 0
        assert all(c.chapter is None for c in subsec_chunks)
        assert all(c.section == "Chapter 1" for c in subsec_chunks)

    def test_subsection_none_for_l1_node_chunks(self):
        """Chunks from a level-1 node have subsection=None."""
        from bookmind_tutor.ingestion.hierarchical_chunker import HierarchicalChunker

        chapter = DocumentNode(
            title="Chapter 1", level=1,
            page_range=(1, 5), children=[], raw_text="Chapter body text."
        )
        tree = DocumentTree(root_nodes=[chapter])
        chunks = HierarchicalChunker().chunk(tree)
        assert all(c.subsection is None for c in chunks)
