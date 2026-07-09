"""Tests for EntityExtractor."""
import json
from unittest.mock import MagicMock

import pytest

from bookmind_tutor.knowledge_graph.extractor import EntityExtractor
from bookmind_tutor.ingestion.models import Chunk
from bookmind_tutor.llm.base import CompletionResponse, TokenUsage


def _make_chunk(text="Raft is a consensus algorithm.", chunk_id="chunk_001"):
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        chapter="Chapter 9",
        section=None,
        page_range=(380, 390),
        char_offset_start=0,
        char_offset_end=len(text),
    )


def _completion(text: str) -> CompletionResponse:
    return CompletionResponse(
        text=text,
        stop_reason="end_turn",
        tool_calls=[],
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        raw_message={"role": "assistant", "content": text},
    )


def _make_extractor(llm_response: str) -> EntityExtractor:
    client = MagicMock()
    client.complete.return_value = _completion(llm_response)
    return EntityExtractor(llm_client=client)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_json_parsed_correctly():
    payload = json.dumps({
        "entities": [
            {"name": "raft", "type": "SYSTEM", "description": "A consensus algorithm."},
            {"name": "consensus algorithm", "type": "CONCEPT", "description": "Agreement protocol."},
        ],
        "relations": [
            {"source": "raft", "type": "IS_A", "target": "consensus algorithm", "description": ""},
        ],
    })
    extractor = _make_extractor(payload)
    entities, relations = extractor.extract(_make_chunk())

    assert len(entities) == 2
    assert len(relations) == 1
    assert entities[0].name == "raft"
    assert relations[0].relation_type == "IS_A"


def test_entity_names_lowercased_by_extractor():
    payload = json.dumps({
        "entities": [{"name": "Kafka", "type": "SYSTEM", "description": "A streaming platform."}],
        "relations": [],
    })
    extractor = _make_extractor(payload)
    entities, _ = extractor.extract(_make_chunk())
    assert entities[0].name == "kafka"


def test_markdown_fence_stripped():
    payload = '```json\n' + json.dumps({
        "entities": [{"name": "zookeeper", "type": "SYSTEM", "description": "Coordination service."}],
        "relations": [],
    }) + '\n```'
    extractor = _make_extractor(payload)
    entities, _ = extractor.extract(_make_chunk())
    assert len(entities) == 1


# ---------------------------------------------------------------------------
# Validation filtering
# ---------------------------------------------------------------------------

def test_invalid_entity_type_dropped():
    payload = json.dumps({
        "entities": [{"name": "raft", "type": "MADE_UP_TYPE", "description": "..."}],
        "relations": [],
    })
    extractor = _make_extractor(payload)
    entities, _ = extractor.extract(_make_chunk())
    assert entities == []


def test_relation_with_unknown_endpoint_dropped():
    """Relation referencing an entity not in the extracted set is dropped."""
    payload = json.dumps({
        "entities": [{"name": "raft", "type": "SYSTEM", "description": "..."}],
        "relations": [
            {"source": "raft", "type": "IS_A", "target": "ghost entity", "description": ""},
        ],
    })
    extractor = _make_extractor(payload)
    _, relations = extractor.extract(_make_chunk())
    assert relations == []


def test_self_loop_relation_dropped():
    payload = json.dumps({
        "entities": [{"name": "raft", "type": "SYSTEM", "description": "..."}],
        "relations": [{"source": "raft", "type": "IS_A", "target": "raft", "description": ""}],
    })
    extractor = _make_extractor(payload)
    _, relations = extractor.extract(_make_chunk())
    assert relations == []


# ---------------------------------------------------------------------------
# Fallback on failure
# ---------------------------------------------------------------------------

def test_invalid_json_returns_empty():
    extractor = _make_extractor("this is not json at all")
    entities, relations = extractor.extract(_make_chunk())
    assert entities == []
    assert relations == []


def test_llm_exception_returns_empty():
    client = MagicMock()
    client.complete.side_effect = ConnectionError("network down")
    extractor = EntityExtractor(llm_client=client)
    entities, relations = extractor.extract(_make_chunk())
    assert entities == []
    assert relations == []
