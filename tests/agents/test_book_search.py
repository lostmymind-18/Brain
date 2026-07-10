"""Tests for BookSearchTool."""
import pytest

from bookmind_tutor.agents.tools.book_search import BookSearchTool
from bookmind_tutor.ingestion.models import Chunk
from bookmind_tutor.retrieval.vector_store import VectorStore


def _make_chunk(chunk_id: str, text: str, chapter: str | None = None,
                section: str | None = None, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, text=text, chapter=chapter, section=section,
        page_range=(page, page), char_offset_start=0, char_offset_end=len(text),
    )


@pytest.fixture
def store_with_data():
    store = VectorStore(persist_dir=None)
    store.index_chunks([
        _make_chunk("c1", "Replication means keeping a copy of the same data on multiple machines.",
                    chapter="Chapter 5", section="Replication", page=151),
        _make_chunk("c2", "Sharding splits a large dataset across multiple partitions.",
                    chapter="Chapter 6", section="Partitioning", page=199),
    ])
    return store


class TestBookSearchTool:
    def test_spec_name_is_book_search(self):
        store = VectorStore(persist_dir=None)
        tool = BookSearchTool(store)
        assert tool.spec.name == "book_search"

    def test_spec_has_query_parameter(self):
        store = VectorStore(persist_dir=None)
        tool = BookSearchTool(store)
        props = tool.spec.input_schema["properties"]
        assert "query" in props

    def test_execute_returns_formatted_results(self, store_with_data):
        tool = BookSearchTool(store_with_data, k=1)
        result = tool.execute(query="how does replication work")
        assert "[1]" in result
        assert "Chapter 5" in result
        assert "p.151" in result

    def test_execute_empty_store_returns_no_results_message(self):
        store = VectorStore(persist_dir=None)
        tool = BookSearchTool(store)
        result = tool.execute(query="anything")
        assert "No relevant" in result

    def test_execute_section_in_output(self, store_with_data):
        tool = BookSearchTool(store_with_data, k=1)
        result = tool.execute(query="replication copy data")
        assert "Replication" in result  # section name

    def test_execute_null_section_still_works(self):
        store = VectorStore(persist_dir=None)
        store.index_chunks([_make_chunk("c1", "Some content.", chapter="Intro", section=None)])
        tool = BookSearchTool(store, k=1)
        result = tool.execute(query="content")
        assert "Intro" in result
        assert "[1]" in result

    def test_execute_subsection_in_location_label(self):
        store = VectorStore(persist_dir=None)
        chunk = Chunk(
            chunk_id="c1", text="Events flow between processors.",
            chapter="Chapter 14", section="Topologies",
            page_range=(380, 380), char_offset_start=0, char_offset_end=31,
            subsection="Broker Topology",
        )
        store.index_chunks([chunk])
        tool = BookSearchTool(store, k=1)
        result = tool.execute(query="events processors")
        assert "Chapter 14 / Topologies / Broker Topology" in result
