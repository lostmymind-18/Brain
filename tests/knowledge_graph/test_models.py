"""Tests for knowledge graph data models."""
import pytest

from bookmind_tutor.knowledge_graph.models import Entity, Relation


def test_entity_name_normalized_to_lowercase():
    e = Entity(name="Raft", type="SYSTEM", description="...", chunk_id="c1")
    assert e.name == "raft"


def test_entity_name_stripped():
    e = Entity(name="  eventual consistency  ", type="CONCEPT", description="...", chunk_id="c1")
    assert e.name == "eventual consistency"


def test_entity_valid():
    e = Entity(name="kafka", type="SYSTEM", description="...", chunk_id="c1")
    assert e.is_valid()


def test_entity_invalid_type():
    e = Entity(name="kafka", type="UNKNOWN_TYPE", description="...", chunk_id="c1")
    assert not e.is_valid()


def test_entity_invalid_empty_name():
    e = Entity(name="", type="SYSTEM", description="...", chunk_id="c1")
    assert not e.is_valid()


def test_relation_names_normalized():
    r = Relation(source="Raft", relation_type="IS_A", target="Consensus Algorithm")
    assert r.source == "raft"
    assert r.target == "consensus algorithm"


def test_relation_valid():
    r = Relation(source="raft", relation_type="IS_A", target="consensus algorithm")
    assert r.is_valid()


def test_relation_invalid_type():
    r = Relation(source="raft", relation_type="INVENTED_BY", target="leslie lamport")
    assert not r.is_valid()


def test_relation_invalid_self_loop():
    r = Relation(source="raft", relation_type="IS_A", target="raft")
    assert not r.is_valid()


def test_relation_invalid_empty_source():
    r = Relation(source="", relation_type="IS_A", target="consensus algorithm")
    assert not r.is_valid()
