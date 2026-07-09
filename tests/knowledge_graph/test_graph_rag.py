"""Tests for GraphRAGRetriever."""
from unittest.mock import MagicMock

from bookmind_tutor.knowledge_graph.graph_rag import GraphRAGRetriever
from bookmind_tutor.knowledge_graph.models import Entity
from bookmind_tutor.retrieval.models import SearchResult


def _make_result(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=f"text of {chunk_id}",
        chapter="Ch1",
        section=None,
        page_range=(1, 2),
        score=score,
    )


def _make_retriever(
    vector_results: list,
    entity_names: list[str],
    related_entities: list[Entity],
    expanded_results: list | None = None,
    user_entity_names: list[str] | None = None,
) -> GraphRAGRetriever:
    vector_store = MagicMock()
    if expanded_results is not None:
        vector_store.search.side_effect = [vector_results, expanded_results]
    else:
        vector_store.search.return_value = vector_results

    graph_store = MagicMock()
    graph_store.get_all_entity_names.return_value = entity_names
    graph_store.get_related_entities.return_value = related_entities
    graph_store.get_user_entity_names.return_value = user_entity_names or []

    return GraphRAGRetriever(vector_store=vector_store, graph_store=graph_store, k=5)


def test_no_entity_match_returns_vector_results_only():
    results = [_make_result("c1", 0.9)]
    retriever = _make_retriever(
        vector_results=results,
        entity_names=["kafka", "raft"],  # not in query
        related_entities=[],
    )
    output = retriever.search("What is eventual consistency?")
    assert output == results
    # Second vector search should not have been called
    retriever._vector_store.search.assert_called_once()


def test_entity_match_triggers_graph_expansion():
    vector_results = [_make_result("c1", 0.9)]
    expanded_results = [_make_result("c2", 0.7)]
    raft_entity = Entity(name="paxos", type="SYSTEM", description="...", chunk_id="c2")

    retriever = _make_retriever(
        vector_results=vector_results,
        entity_names=["raft"],  # "raft" appears in query
        related_entities=[raft_entity],
        expanded_results=expanded_results,
    )
    output = retriever.search("How does raft achieve consensus?")

    assert retriever._vector_store.search.call_count == 2
    assert len(output) == 2  # both chunks returned


def test_deduplication_keeps_highest_score():
    """Same chunk returned by both searches — keep the higher score."""
    r1 = _make_result("c1", 0.9)
    r1_lower = _make_result("c1", 0.5)

    retriever = _make_retriever(
        vector_results=[r1],
        entity_names=["raft"],
        related_entities=[Entity(name="paxos", type="SYSTEM", description="...", chunk_id="c1")],
        expanded_results=[r1_lower],
    )
    output = retriever.search("How does raft work?")

    c1_results = [r for r in output if r.chunk_id == "c1"]
    assert len(c1_results) == 1
    assert c1_results[0].score == 0.9


def test_results_sorted_by_score_descending():
    r_low = _make_result("c1", 0.5)
    r_high = _make_result("c2", 0.9)
    raft_entity = Entity(name="paxos", type="SYSTEM", description="...", chunk_id="c3")
    r_mid = _make_result("c3", 0.7)

    retriever = _make_retriever(
        vector_results=[r_low, r_high],
        entity_names=["raft"],
        related_entities=[raft_entity],
        expanded_results=[r_mid],
    )
    output = retriever.search("raft consensus")
    scores = [r.score for r in output]
    assert scores == sorted(scores, reverse=True)


def test_results_capped_at_k():
    many_results = [_make_result(f"c{i}", 1.0 - i * 0.05) for i in range(8)]
    retriever = _make_retriever(
        vector_results=many_results[:4],
        entity_names=["raft"],
        related_entities=[Entity(name="paxos", type="SYSTEM", description="...", chunk_id="cx")],
        expanded_results=many_results[4:],
    )
    retriever._k = 5
    output = retriever.search("raft")
    assert len(output) <= 5


def test_match_query_entities_longest_first():
    """'eventual consistency' should match before 'consistency'."""
    retriever = _make_retriever(
        vector_results=[],
        entity_names=["consistency", "eventual consistency"],
        related_entities=[],
    )
    matched = retriever._match_book_entities("explain eventual consistency trade-offs")
    assert matched[0] == "eventual consistency"


def test_user_entity_match_triggers_expansion():
    """User-KG entities appearing in query cause a second vector search."""
    vector_results = [_make_result("c1", 0.9)]
    expanded_results = [_make_result("c2", 0.7)]

    retriever = _make_retriever(
        vector_results=vector_results,
        entity_names=[],           # no book entities match
        related_entities=[],
        expanded_results=expanded_results,
        user_entity_names=["architecture pattern"],  # appears in query
    )
    output = retriever.search("What is event sourcing as an architecture pattern?")

    assert retriever._vector_store.search.call_count == 2
    assert len(output) == 2


def test_user_entity_no_match_skips_expansion():
    """User-KG entities NOT in query do not trigger expansion."""
    vector_results = [_make_result("c1", 0.9)]

    retriever = _make_retriever(
        vector_results=vector_results,
        entity_names=[],
        related_entities=[],
        user_entity_names=["paxos"],   # not in query
    )
    output = retriever.search("What is event sourcing?")

    retriever._vector_store.search.assert_called_once()
    assert output == vector_results


def test_user_and_book_expansion_combined():
    """Both book-graph and user-KG terms end up in the expanded query."""
    vector_results = [_make_result("c1", 0.9)]
    expanded_results = [_make_result("c2", 0.8)]
    book_related = Entity(name="cqrs", type="SYSTEM", description="...", chunk_id="cx")

    retriever = _make_retriever(
        vector_results=vector_results,
        entity_names=["event sourcing"],   # matches query
        related_entities=[book_related],   # book expansion: "cqrs"
        expanded_results=expanded_results,
        user_entity_names=["architecture pattern"],  # user expansion: appears in query
    )
    output = retriever.search("explain event sourcing as an architecture pattern")

    # Expanded query must contain both "cqrs" (book) and "architecture pattern" (user)
    _, call_kwargs = retriever._vector_store.search.call_args_list[1]
    expanded_query: str = retriever._vector_store.search.call_args_list[1].args[0]
    assert "cqrs" in expanded_query
    assert "architecture pattern" in expanded_query
    assert len(output) == 2
