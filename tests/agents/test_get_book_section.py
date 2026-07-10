"""Tests for GetBookSectionTool."""
import pytest

from bookmind_tutor.agents.tools.get_book_section import GetBookSectionTool, _MAX_CHUNKS
from bookmind_tutor.ingestion.models import Chunk
from bookmind_tutor.retrieval.vector_store import VectorStore


def _make_chunk(
    chunk_id: str,
    text: str,
    chapter: str | None = None,
    section: str | None = None,
    page: int = 1,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        chapter=chapter,
        section=section,
        page_range=(page, page),
        char_offset_start=0,
        char_offset_end=len(text),
    )


@pytest.fixture
def store():
    s = VectorStore(persist_dir=None)
    s.index_chunks([
        _make_chunk("p1c1", "Part I intro text.",
                    chapter="Part I. Foundations", section="", page=1),
        _make_chunk("p1c2", "Replication keeps copies on multiple machines.",
                    chapter="Part I. Foundations", section="Chapter 5. Replication", page=151),
        _make_chunk("p1c3", "Sharding splits data across partitions.",
                    chapter="Part I. Foundations", section="Chapter 6. Partitioning", page=199),
        _make_chunk("p2c1", "Event-driven systems use async messaging.",
                    chapter="Part II. Architecture Styles", section="Chapter 14. Event-Driven", page=181),
    ])
    return s


class TestGetBookSectionTool:
    def test_spec_name(self, store):
        assert GetBookSectionTool(store).spec.name == "get_book_section"

    def test_spec_requires_field_and_contains(self, store):
        schema = GetBookSectionTool(store).spec.input_schema
        assert "field" in schema["properties"]
        assert "contains" in schema["properties"]
        assert schema["required"] == ["field", "contains"]

    def test_filter_by_section_returns_matching_chunks(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(field="section", contains="Chapter 5")
        assert "Replication keeps copies" in result
        assert "Sharding" not in result
        assert "Event-driven" not in result

    def test_filter_by_chapter_returns_all_chunks_in_part(self, store):
        tool = GetBookSectionTool(store)
        # Use "Part I." (with period) to avoid matching "Part II." as a substring.
        # In practice the LLM copies the exact name from book_outline output.
        result = tool.execute(field="chapter", contains="Part I.")
        assert "Part I intro text" in result
        assert "Replication keeps copies" in result
        assert "Sharding splits" in result
        assert "Event-driven" not in result

    def test_filter_is_case_insensitive(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(field="section", contains="chapter 5")
        assert "Replication" in result

    def test_filter_is_substring_match(self, store):
        tool = GetBookSectionTool(store)
        # "Pipeline" is not in our fixture but partial match should still work
        result = tool.execute(field="section", contains="Replication")
        assert "Replication keeps copies" in result

    def test_results_sorted_by_page(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(field="chapter", contains="Part I")
        idx_intro = result.index("Part I intro text")
        idx_replication = result.index("Replication keeps copies")
        idx_sharding = result.index("Sharding splits")
        assert idx_intro < idx_replication < idx_sharding

    def test_no_match_returns_helpful_message(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(field="section", contains="Chapter 99")
        assert "No chunks found" in result
        assert "book_outline" in result

    def test_empty_store_returns_message(self):
        s = VectorStore(persist_dir=None)
        tool = GetBookSectionTool(s)
        result = tool.execute(field="section", contains="Chapter 5")
        assert "No book content indexed yet" in result

    def test_invalid_field_returns_error(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(field="page_start", contains="anything")
        assert "Invalid field" in result

    def test_page_from_filter(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(field="chapter", contains="Part I", page_from=180)
        assert "Sharding splits" in result
        assert "Part I intro text" not in result
        assert "Replication keeps copies" not in result

    def test_page_to_filter(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(field="chapter", contains="Part I", page_to=160)
        assert "Part I intro text" in result
        assert "Replication keeps copies" in result
        assert "Sharding splits" not in result

    def test_truncation_note_when_over_limit(self):
        s = VectorStore(persist_dir=None)
        chunks = [
            _make_chunk(f"c{i}", f"Content of chunk {i}.",
                        chapter="Part I. Big Part", section=f"Chapter {i}", page=i)
            for i in range(_MAX_CHUNKS + 5)
        ]
        s.index_chunks(chunks)
        tool = GetBookSectionTool(s)
        result = tool.execute(field="chapter", contains="Big Part")
        assert "Note:" in result
        assert str(_MAX_CHUNKS) in result

    def test_source_location_in_output(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(field="section", contains="Chapter 5")
        assert "p.151" in result
