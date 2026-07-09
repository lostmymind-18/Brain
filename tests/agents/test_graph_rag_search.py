"""Tests for GraphRAGSearchTool."""
from unittest.mock import MagicMock

from bookmind_tutor.agents.tools.graph_rag_search import GraphRAGSearchTool
from bookmind_tutor.retrieval.models import SearchResult


def _make_result(chunk_id: str, chapter: str, page: int, text: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=text,
        chapter=chapter,
        section=None,
        page_range=(page, page + 2),
        score=score,
    )


def _make_tool(results: list) -> GraphRAGSearchTool:
    retriever = MagicMock()
    retriever.search.return_value = results
    return GraphRAGSearchTool(retriever=retriever)


def test_formats_results_with_citation():
    tool = _make_tool([
        _make_result("c1", "Chapter 5. Replication", 174, "Raft is a consensus algorithm.")
    ])
    output = tool.execute("What is Raft?")
    assert "[1]" in output
    assert "Chapter 5. Replication" in output
    assert "p.174" in output
    assert "Raft is a consensus algorithm." in output


def test_empty_results_returns_fallback():
    tool = _make_tool([])
    output = tool.execute("something obscure")
    assert "No relevant passages found" in output


def test_multiple_results_numbered():
    tool = _make_tool([
        _make_result("c1", "Ch5", 100, "text one"),
        _make_result("c2", "Ch6", 200, "text two"),
    ])
    output = tool.execute("query")
    assert "[1]" in output
    assert "[2]" in output


def test_tool_name_is_book_search():
    tool = _make_tool([])
    assert tool.spec.name == "book_search"


def test_delegates_query_to_retriever():
    retriever = MagicMock()
    retriever.search.return_value = []
    tool = GraphRAGSearchTool(retriever=retriever)
    tool.execute("consensus algorithms")
    retriever.search.assert_called_once_with("consensus algorithms")
