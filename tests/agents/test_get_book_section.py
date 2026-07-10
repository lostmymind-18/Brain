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
        # Mirrors the real-book anomaly: a chapter that sits outside any part
        # lands at part level, i.e. in the `chapter` metadata field.
        _make_chunk("ch1a", "Software architecture is defined here.",
                    chapter="Chapter 1. Introduction", section="Defining Architecture", page=21),
        _make_chunk("ch1b", "Architects have broad expectations.",
                    chapter="Chapter 1. Introduction", section="Expectations", page=28),
    ])
    return s


class TestGetBookSectionTool:
    def test_spec_name(self, store):
        assert GetBookSectionTool(store).spec.name == "get_book_section"

    def test_spec_requires_section_name(self, store):
        schema = GetBookSectionTool(store).spec.input_schema
        assert "section_name" in schema["properties"]
        assert schema["required"] == ["section_name"]

    def test_chapter_level_heading_returns_matching_chunks(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(section_name="Chapter 5. Replication")
        assert "Replication keeps copies" in result
        assert "Sharding" not in result
        assert "Event-driven" not in result

    def test_part_level_heading_returns_all_chunks_in_part(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(section_name="Part I. Foundations")
        assert "Part I intro text" in result
        assert "Replication keeps copies" in result
        assert "Sharding splits" in result
        assert "Event-driven" not in result

    def test_finds_chapter_stored_at_part_level(self, store):
        # The real-book failure case: Chapter 1 lives in the `chapter` field
        # because it precedes Part I. A single name must find it regardless
        # of which metadata field it landed in.
        tool = GetBookSectionTool(store)
        result = tool.execute(section_name="Chapter 1. Introduction")
        assert "Software architecture is defined here" in result
        assert "Architects have broad expectations" in result
        assert "Replication" not in result

    def test_matching_is_case_insensitive(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(section_name="chapter 5. replication")
        assert "Replication keeps copies" in result

    def test_ambiguous_name_returns_disambiguation_list(self, store):
        tool = GetBookSectionTool(store)
        # "Part I" substring-matches both "Part I. Foundations" and
        # "Part II. Architecture Styles".
        result = tool.execute(section_name="Part I")
        assert "matches 2 different headings" in result
        assert "Part I. Foundations" in result
        assert "Part II. Architecture Styles" in result
        # Must NOT dump chunk content.
        assert "Replication keeps copies" not in result

    def test_exact_name_wins_over_substring_ambiguity(self, store):
        tool = GetBookSectionTool(store)
        # Exact heading name must return content directly even though it is
        # also a substring of other headings (retry-with-exact-name must work).
        result = tool.execute(section_name="Part I. Foundations")
        assert "matches" not in result.split("\n")[0] or "Part I intro text" in result
        assert "Part I intro text" in result

    def test_no_match_lists_available_headings(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(section_name="Chapter 99")
        assert "No part or chapter named" in result
        assert "Part I. Foundations" in result
        assert "Chapter 5. Replication" in result

    def test_empty_section_name_returns_guidance(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(section_name="")
        assert "must not be empty" in result
        assert "book_outline" in result

    def test_missing_section_name_returns_guidance(self, store):
        # The executor calls execute(**tc.input); if the model omits the
        # param entirely we must return guidance, not raise TypeError.
        tool = GetBookSectionTool(store)
        result = tool.execute()
        assert "must not be empty" in result

    def test_empty_store_returns_message(self):
        s = VectorStore(persist_dir=None)
        tool = GetBookSectionTool(s)
        result = tool.execute(section_name="Chapter 5")
        assert "No book content indexed yet" in result

    def test_results_sorted_by_page(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(section_name="Part I. Foundations")
        idx_intro = result.index("Part I intro text")
        idx_replication = result.index("Replication keeps copies")
        idx_sharding = result.index("Sharding splits")
        assert idx_intro < idx_replication < idx_sharding

    def test_page_from_filter(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(section_name="Part I. Foundations", page_from=180)
        assert "Sharding splits" in result
        assert "Part I intro text" not in result
        assert "Replication keeps copies" not in result

    def test_page_to_filter(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(section_name="Part I. Foundations", page_to=160)
        assert "Part I intro text" in result
        assert "Replication keeps copies" in result
        assert "Sharding splits" not in result

    def test_page_range_excluding_everything_reports_it(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(section_name="Chapter 5. Replication", page_from=500)
        assert "no chunks in the requested page range" in result

    def test_punctuation_variants_group_as_one_heading(self):
        # The same chapter appears as 'Chapter 1. Introduction' (body, many
        # chunks) and 'Chapter\xa01: Introduction' (appendix TOC, 1 chunk).
        # An LLM guessing either punctuation must reach the real chapter, not
        # exact-match only the appendix quiz page (observed gpt-4o-mini failure).
        s = VectorStore(persist_dir=None)
        s.index_chunks([
            _make_chunk("b1", "Real chapter content about architecture.",
                        chapter="Chapter 1. Introduction", section="Defining", page=21),
            _make_chunk("b2", "More real chapter content.",
                        chapter="Chapter 1. Introduction", section="Expectations", page=28),
            _make_chunk("t1", "Quiz questions for chapter one.",
                        chapter="Appendix", section="Chapter\xa01: Introduction", page=393),
        ])
        tool = GetBookSectionTool(s)
        result = tool.execute(section_name="Chapter 1: Introduction")
        assert "Real chapter content" in result
        assert "More real chapter content" in result
        # The appendix variant is part of the same heading group.
        assert "Quiz questions" in result

    def test_truncation_note_when_over_limit(self):
        s = VectorStore(persist_dir=None)
        chunks = [
            _make_chunk(f"c{i}", f"Content of chunk {i}.",
                        chapter="Part IV. Big Part", section=f"Topic {i}", page=i + 1)
            for i in range(_MAX_CHUNKS + 5)
        ]
        s.index_chunks(chunks)
        tool = GetBookSectionTool(s)
        result = tool.execute(section_name="Part IV. Big Part")
        assert "Note:" in result
        assert str(_MAX_CHUNKS) in result
        assert "DO NOT call this tool again" in result

    def test_source_location_in_output(self, store):
        tool = GetBookSectionTool(store)
        result = tool.execute(section_name="Chapter 5. Replication")
        assert "p.151" in result

    def test_last_sources_recorded_on_success(self, store):
        tool = GetBookSectionTool(store)
        tool.execute(section_name="Part I. Foundations")
        assert len(tool.last_sources) == 1
        ref = tool.last_sources[0]
        assert ref.chapter == "Part I. Foundations"
        assert ref.page_range == (1, 199)

    def test_last_sources_not_recorded_on_disambiguation(self, store):
        tool = GetBookSectionTool(store)
        tool.execute(section_name="Part I")  # ambiguous
        assert tool.last_sources == []

    def test_last_sources_accumulate_across_calls(self, store):
        # The UI clears last_sources once per turn; within a turn every
        # successful retrieval must be kept (e.g. comparing two chapters).
        tool = GetBookSectionTool(store)
        tool.execute(section_name="Chapter 5. Replication")
        tool.execute(section_name="Chapter 6. Partitioning")
        assert [r.chapter for r in tool.last_sources] == [
            "Chapter 5. Replication",
            "Chapter 6. Partitioning",
        ]
