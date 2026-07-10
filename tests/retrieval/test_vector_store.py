"""Tests for VectorStore."""
import pytest

from bookmind_tutor.ingestion.models import Chunk
from bookmind_tutor.retrieval.vector_store import VectorStore, _context_text


def _make_chunk(chunk_id: str, text: str, chapter: str | None = None,
                section: str | None = None, page: int = 1,
                subsection: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        chapter=chapter,
        section=section,
        page_range=(page, page),
        char_offset_start=0,
        char_offset_end=len(text),
        subsection=subsection,
    )


@pytest.fixture
def store():
    return VectorStore(persist_dir=None)  # ephemeral


class TestVectorStoreBasics:
    def test_empty_store_count_is_zero(self, store):
        assert store.count() == 0

    def test_index_increments_count(self, store):
        chunks = [_make_chunk("c1", "Some text about databases.")]
        store.index_chunks(chunks)
        assert store.count() == 1

    def test_upsert_is_idempotent(self, store):
        chunks = [_make_chunk("c1", "Some text.")]
        store.index_chunks(chunks)
        store.index_chunks(chunks)
        assert store.count() == 1

    def test_clear_resets_count(self, store):
        store.index_chunks([_make_chunk("c1", "Some text.")])
        store.clear()
        assert store.count() == 0

    def test_index_no_chunks_is_noop(self, store):
        store.index_chunks([])
        assert store.count() == 0


class TestVectorStoreSearch:
    def test_search_returns_results(self, store):
        chunks = [
            _make_chunk("c1", "Replication keeps data copies on multiple machines."),
            _make_chunk("c2", "Indexing allows fast lookup by key."),
            _make_chunk("c3", "Consensus algorithms agree on a single value."),
        ]
        store.index_chunks(chunks)
        results = store.search("how does replication work", k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_search_empty_store_returns_empty(self, store):
        results = store.search("anything")
        assert results == []

    def test_search_k_limits_results(self, store):
        chunks = [_make_chunk(f"c{i}", f"Document number {i}.") for i in range(10)]
        store.index_chunks(chunks)
        results = store.search("document", k=3)
        assert len(results) <= 3

    def test_search_result_has_metadata(self, store):
        chunk = _make_chunk("c1", "Partitioning splits data across nodes.",
                            chapter="Chapter 6", section="Partitioning", page=200)
        store.index_chunks([chunk])
        results = store.search("partitioning", k=1)
        r = results[0]
        assert r.chunk_id == "c1"
        assert r.chapter == "Chapter 6"
        assert r.section == "Partitioning"
        assert r.page_range == (200, 200)

    def test_search_score_is_float_between_0_and_1(self, store):
        store.index_chunks([_make_chunk("c1", "Transactions provide ACID guarantees.")])
        results = store.search("ACID properties", k=1)
        assert 0.0 <= results[0].score <= 1.0

    def test_chapter_none_when_not_provided(self, store):
        store.index_chunks([_make_chunk("c1", "Some text.", chapter=None)])
        results = store.search("text", k=1)
        assert results[0].chapter is None

    def test_results_sorted_by_score_descending(self, store):
        chunks = [
            _make_chunk("high", "Replication: keeping a copy of the same data on multiple machines."),
            _make_chunk("low", "The quick brown fox jumps over the lazy dog."),
        ]
        store.index_chunks(chunks)
        results = store.search("data replication on multiple machines", k=2)
        if len(results) == 2:
            assert results[0].score >= results[1].score

    def test_search_result_has_subsection(self, store):
        chunk = _make_chunk("c1", "Events flow through the mesh.",
                            chapter="Chapter 14", section="Topologies",
                            subsection="Broker Topology", page=380)
        store.index_chunks([chunk])
        results = store.search("broker", k=1)
        assert results[0].subsection == "Broker Topology"


class TestContextualEmbedding:
    def test_context_text_prepends_breadcrumb(self):
        out = _context_text("Body.", "Chapter 14", "Topologies", "Broker Topology")
        assert out == "Chapter 14 > Topologies > Broker Topology\nBody."

    def test_context_text_skips_missing_levels(self):
        assert _context_text("Body.", "Chapter 1", None, None) == "Chapter 1\nBody."
        assert _context_text("Body.", None, None, None) == "Body."

    def test_heading_query_finds_chunk_without_heading_words(self, store):
        """
        The core value of contextual embedding: a chunk whose BODY never
        mentions its heading must still be findable by the heading name.
        """
        chunks = [
            _make_chunk(
                "target",
                "The event flows to the next processor in the chain, and each "
                "processor advertises what it has done to the rest of the system.",
                chapter="Chapter 14. Event-Driven Architecture",
                subsection="Broker Topology",
            ),
            _make_chunk("distractor", "Databases store rows in tables with indexes."),
        ]
        store.index_chunks(chunks)
        results = store.search("broker topology", k=1)
        assert results[0].chunk_id == "target"

    def test_stored_document_text_stays_clean(self, store):
        """The breadcrumb must NOT leak into the text shown to the LLM/user."""
        chunk = _make_chunk("c1", "Plain body text.",
                            chapter="Chapter 1", subsection="Some Heading")
        store.index_chunks([chunk])
        results = store.search("plain body", k=1)
        assert results[0].text == "Plain body text."
        assert "Chapter 1 >" not in results[0].text
