"""Tests for GraphStore — all Neo4j calls are mocked."""
from unittest.mock import MagicMock, patch, call

import pytest

from bookmind_tutor.knowledge_graph.graph_store import GraphStore
from bookmind_tutor.knowledge_graph.models import Entity, Relation


def _make_store() -> tuple[GraphStore, MagicMock]:
    """Return a GraphStore with a mocked Neo4j driver."""
    with patch("bookmind_tutor.knowledge_graph.graph_store.GraphDatabase") as mock_gdb:
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver

        # Mock session as context manager
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store = GraphStore(uri="bolt://localhost:7687", user="neo4j", password="test")
        return store, mock_session


def test_upsert_entity_calls_merge():
    store, session = _make_store()
    entity = Entity(name="raft", type="SYSTEM", description="A consensus algo.", chunk_id="c1")
    store.upsert_entity(entity)
    session.run.assert_called()
    cypher = session.run.call_args[0][0]
    assert "MERGE" in cypher
    assert "Entity" in cypher


def test_upsert_relation_uses_correct_type():
    store, session = _make_store()
    relation = Relation(source="raft", relation_type="IS_A", target="consensus algorithm")
    store.upsert_relation(relation)
    cypher = session.run.call_args[0][0]
    assert "IS_A" in cypher


def test_upsert_unknown_relation_type_skips():
    store, session = _make_store()
    relation = Relation(source="raft", relation_type="INVENTED_BY", target="leslie lamport")
    # Reset call count after __init__ calls
    session.run.reset_mock()
    store.upsert_relation(relation)
    session.run.assert_not_called()


def test_get_related_entities_runs_match_query():
    store, session = _make_store()
    mock_record = MagicMock()
    mock_node = {
        "name": "paxos",
        "type": "SYSTEM",
        "description": "Another consensus algo.",
        "chunk_id": "c2",
    }
    mock_record.__getitem__ = lambda self, key: mock_node if key == "related" else None
    session.run.return_value = [mock_record]

    results = store.get_related_entities("raft", depth=1)
    cypher = session.run.call_args[0][0]
    assert "MATCH" in cypher
    assert len(results) == 1
    assert results[0].name == "paxos"


def test_clear_deletes_all_nodes():
    store, session = _make_store()
    store.clear()
    cypher = session.run.call_args[0][0]
    assert "DELETE" in cypher


def test_entity_name_cache_invalidated_on_upsert():
    store, session = _make_store()
    # Seed cache
    session.run.return_value = [{"name": "raft"}]
    names = store.get_all_entity_names()
    assert store._entity_name_cache is not None

    # Upsert should clear cache
    store.upsert_entity(Entity(name="paxos", type="SYSTEM", description="...", chunk_id="c1"))
    assert store._entity_name_cache is None
