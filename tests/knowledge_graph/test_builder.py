"""Tests for KnowledgeGraphBuilder."""
from unittest.mock import MagicMock, call

from bookmind_tutor.ingestion.models import Chunk
from bookmind_tutor.knowledge_graph.builder import KnowledgeGraphBuilder
from bookmind_tutor.knowledge_graph.models import Entity, Relation


def _make_chunk(i: int) -> Chunk:
    text = f"Text of chunk {i}"
    return Chunk(
        chunk_id=f"chunk_{i:03d}",
        text=text,
        chapter="Chapter 1",
        section=None,
        page_range=(i, i),
        char_offset_start=0,
        char_offset_end=len(text),
    )


def _make_entity(name: str) -> Entity:
    return Entity(name=name, type="CONCEPT", description="...", chunk_id="chunk_001")


def _make_relation(src: str, tgt: str) -> Relation:
    return Relation(source=src, relation_type="RELATED_TO", target=tgt)


def test_all_chunks_processed():
    extractor = MagicMock()
    extractor.extract.return_value = ([], [])
    store = MagicMock()

    chunks = [_make_chunk(i) for i in range(5)]
    builder = KnowledgeGraphBuilder(graph_store=store, extractor=extractor)
    stats = builder.build_from_chunks(chunks)

    assert extractor.extract.call_count == 5
    assert stats.chunks_processed == 5
    assert stats.chunks_failed == 0


def test_entities_and_relations_upserted():
    entities = [_make_entity("raft"), _make_entity("paxos")]
    relations = [_make_relation("raft", "paxos")]

    extractor = MagicMock()
    extractor.extract.return_value = (entities, relations)
    store = MagicMock()

    builder = KnowledgeGraphBuilder(graph_store=store, extractor=extractor)
    stats = builder.build_from_chunks([_make_chunk(1)])

    assert store.upsert_entity.call_count == 2
    assert store.upsert_relation.call_count == 1
    assert stats.total_entities == 2
    assert stats.total_relations == 1


def test_failed_chunk_does_not_abort():
    extractor = MagicMock()
    extractor.extract.side_effect = [
        ([_make_entity("kafka")], []),
        Exception("LLM timeout"),
        ([_make_entity("raft")], []),
    ]
    store = MagicMock()

    chunks = [_make_chunk(i) for i in range(3)]
    builder = KnowledgeGraphBuilder(graph_store=store, extractor=extractor)
    stats = builder.build_from_chunks(chunks)

    assert stats.chunks_processed == 2
    assert stats.chunks_failed == 1
    assert stats.total_entities == 2


def test_max_chunks_limits_processing():
    extractor = MagicMock()
    extractor.extract.return_value = ([], [])
    store = MagicMock()

    chunks = [_make_chunk(i) for i in range(10)]
    builder = KnowledgeGraphBuilder(graph_store=store, extractor=extractor)
    stats = builder.build_from_chunks(chunks, max_chunks=3)

    assert extractor.extract.call_count == 3
    assert stats.total_chunks == 3


def test_stats_elapsed_seconds_populated():
    extractor = MagicMock()
    extractor.extract.return_value = ([], [])
    store = MagicMock()

    builder = KnowledgeGraphBuilder(graph_store=store, extractor=extractor)
    stats = builder.build_from_chunks([_make_chunk(1)])
    assert stats.elapsed_seconds >= 0
