"""Tests for BookOutlineTool, including subsection-level output."""
from __future__ import annotations

from bookmind_tutor.agents.tools.book_outline import BookOutlineTool
from bookmind_tutor.ingestion.models import Chunk
from bookmind_tutor.retrieval.vector_store import VectorStore


def _chunk(chunk_id: str, chapter: str, section: str | None = None,
           subsection: str | None = None, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, text=f"Text of {chunk_id}.",
        chapter=chapter, section=section,
        page_range=(page, page), char_offset_start=0, char_offset_end=10,
        subsection=subsection,
    )


def _store_with(chunks: list[Chunk]) -> VectorStore:
    store = VectorStore(persist_dir=None)
    store.index_chunks(chunks)
    return store


class TestBookOutlineTool:
    def test_empty_store_message(self):
        tool = BookOutlineTool(VectorStore(persist_dir=None))
        assert "No book content" in tool.execute()

    def test_chapters_sorted_by_page(self):
        store = _store_with([
            _chunk("c2", "Part II", page=100),
            _chunk("c1", "Part I", page=10),
        ])
        result = BookOutlineTool(store).execute()
        assert result.index("Part I") < result.index("Part II")

    def test_sections_nested_under_chapter(self):
        store = _store_with([
            _chunk("c1", "Part I", page=10),
            _chunk("c2", "Part I", section="Chapter 1", page=12),
        ])
        result = BookOutlineTool(store).execute()
        assert "**Part I** (p. 10)" in result
        assert "  - Chapter 1 (p. 12)" in result

    def test_subsections_nested_under_section(self):
        store = _store_with([
            _chunk("c1", "Part I", section="Chapter 14", page=370),
            _chunk("c2", "Part I", section="Chapter 14",
                   subsection="Broker Topology", page=380),
            _chunk("c3", "Part I", section="Chapter 14",
                   subsection="Mediator Topology", page=390),
        ])
        result = BookOutlineTool(store).execute()
        assert "  - Chapter 14 (p. 370)" in result
        assert "    - Broker Topology (p. 380)" in result
        assert "    - Mediator Topology (p. 390)" in result
        # Reading order preserved
        assert result.index("Broker") < result.index("Mediator")

    def test_subsection_without_section_renders_at_section_level(self):
        """Chunks with a subsection but empty section nest directly under chapter."""
        store = _store_with([
            _chunk("c1", "Preface", subsection="Invalidating Axioms", page=5),
        ])
        result = BookOutlineTool(store).execute()
        assert "**Preface** (p. 5)" in result
        assert "  - Invalidating Axioms (p. 5)" in result

    def test_skip_labels_excluded(self):
        store = _store_with([
            _chunk("c1", "Table of Contents", page=3),
            _chunk("c2", "Chapter 1", page=10),
        ])
        result = BookOutlineTool(store).execute()
        assert "Table of Contents" not in result
        assert "Chapter 1" in result
