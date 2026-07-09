"""Tests for KGUpdateSuggester."""
from unittest.mock import MagicMock, call

from bookmind_tutor.knowledge_graph.models import Entity, KGUpdateResult, Relation
from bookmind_tutor.knowledge_graph.suggester import KGUpdateSuggester
from bookmind_tutor.retrieval.models import SearchResult


def _make_result(chunk_id: str, text: str = "some text", page: int = 10) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=text,
        chapter="Chapter 5",
        section=None,
        page_range=(page, page + 2),
        score=0.9,
    )


def _make_entity(name: str, entity_type: str = "CONCEPT") -> Entity:
    return Entity(name=name, type=entity_type, description="desc", chunk_id="c1")


def _make_suggester(extracted_entities: list[Entity]) -> tuple[KGUpdateSuggester, MagicMock]:
    extractor = MagicMock()
    extractor.extract.return_value = (extracted_entities, [])
    return KGUpdateSuggester(extractor=extractor), extractor


def _make_graph(known_names: list[str]) -> MagicMock:
    graph = MagicMock()
    graph.get_user_entity_names.return_value = known_names
    return graph


def test_suggests_new_entities():
    suggester, _ = _make_suggester([_make_entity("raft"), _make_entity("consensus")])
    graph = _make_graph([])
    result = suggester.suggest([_make_result("c1")], graph)
    names = [s.entity.name for s in result.suggestions]
    assert "raft" in names
    assert "consensus" in names


def test_excludes_already_known_entities():
    suggester, _ = _make_suggester([_make_entity("raft"), _make_entity("consensus")])
    graph = _make_graph(["raft"])
    result = suggester.suggest([_make_result("c1")], graph)
    names = [s.entity.name for s in result.suggestions]
    assert "raft" not in names
    assert "consensus" in names
    assert "raft" in result.already_known


def test_caps_suggestions_at_max():
    entities = [_make_entity(f"concept_{i}") for i in range(10)]
    suggester = KGUpdateSuggester(extractor=MagicMock(), max_suggestions=3)
    suggester._extract_cached = lambda r: (entities, [])
    graph = _make_graph([])
    result = suggester.suggest([_make_result("c1")], graph)
    assert len(result.suggestions) == 3


def test_no_duplicate_suggestions_across_chunks():
    # Same entity extracted from two different chunks — should appear only once.
    extractor = MagicMock()
    extractor.extract.return_value = ([_make_entity("raft")], [])
    suggester = KGUpdateSuggester(extractor=extractor)
    graph = _make_graph([])
    result = suggester.suggest([_make_result("c1"), _make_result("c2")], graph)
    names = [s.entity.name for s in result.suggestions]
    assert names.count("raft") == 1


def test_empty_results_returns_empty_suggestions():
    suggester, _ = _make_suggester([])
    graph = _make_graph([])
    result = suggester.suggest([], graph)
    assert result.suggestions == []
    assert result.already_known == []


def test_caches_extraction_by_chunk_id():
    extractor = MagicMock()
    extractor.extract.return_value = ([_make_entity("raft")], [])
    suggester = KGUpdateSuggester(extractor=extractor)
    graph = _make_graph([])
    result1 = _make_result("same_chunk")
    result2 = _make_result("same_chunk")

    # Call twice with the same chunk_id
    suggester.suggest([result1], graph)
    graph.get_user_entity_names.return_value = []  # reset known
    suggester.suggest([result2], graph)

    # Extractor should only be called once (cache hit on second call)
    assert extractor.extract.call_count == 1


def test_suggestion_includes_reason_with_page():
    suggester, _ = _make_suggester([_make_entity("raft")])
    graph = _make_graph([])
    result = suggester.suggest([_make_result("c1", page=174)], graph)
    assert result.suggestions
    assert "174" in result.suggestions[0].reason


def test_upserts_entities_to_book_kg():
    # Entities should be written to the book KG (upsert_entity) regardless of
    # whether they are new to the user KG or already known.
    entity_new = _make_entity("raft")
    entity_known = _make_entity("consensus")
    extractor = MagicMock()
    extractor.extract.return_value = ([entity_new, entity_known], [])
    suggester = KGUpdateSuggester(extractor=extractor)
    graph = _make_graph(["consensus"])  # consensus already known to user

    suggester.suggest([_make_result("c1")], graph)

    # Both entities written to book KG regardless of user KG status
    upserted_names = {c.args[0].name for c in graph.upsert_entity.call_args_list}
    assert "raft" in upserted_names
    assert "consensus" in upserted_names


def test_upserts_relations_to_book_kg():
    # Relations extracted from chunks should be persisted to the book KG.
    entity_a = _make_entity("raft")
    entity_b = _make_entity("consensus")
    relation = Relation(
        source="raft",
        relation_type="IS_A",
        target="consensus",
        description="Raft is a consensus algorithm",
    )
    extractor = MagicMock()
    extractor.extract.return_value = ([entity_a, entity_b], [relation])
    suggester = KGUpdateSuggester(extractor=extractor)
    graph = _make_graph([])

    suggester.suggest([_make_result("c1")], graph)

    graph.upsert_relation.assert_called_once()
    stored_relation = graph.upsert_relation.call_args.args[0]
    assert stored_relation.source == "raft"
    assert stored_relation.relation_type == "IS_A"
    assert stored_relation.target == "consensus"
