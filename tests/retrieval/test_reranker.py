"""Tests for CrossEncoderReranker."""
from unittest.mock import MagicMock, patch

import pytest

from bookmind_tutor.retrieval.models import SearchResult
from bookmind_tutor.retrieval.reranker import CrossEncoderReranker


def _make_result(chunk_id: str, text: str, score: float = 0.5) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=text,
        chapter="Chapter 1",
        section="Section 1",
        page_range=(1, 5),
        score=score,
    )


class TestCrossEncoderReranker:
    def _reranker_with_mock(self, scores: list[float]) -> CrossEncoderReranker:
        """Return a reranker whose model.predict() returns the given scores."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = scores
        reranker._model = mock_model
        return reranker

    def test_rerank_reverses_order_when_scores_say_so(self):
        """Result ranked last by retrieval score can be ranked first by reranker."""
        reranker = self._reranker_with_mock(scores=[1.0, 5.0, 2.0])
        results = [
            _make_result("a", "low relevance text", score=0.9),
            _make_result("b", "most relevant text", score=0.5),
            _make_result("c", "medium relevance text", score=0.7),
        ]
        reranked = reranker.rerank("query", results)
        assert reranked[0].chunk_id == "b"
        assert reranked[1].chunk_id == "c"
        assert reranked[2].chunk_id == "a"

    def test_rerank_replaces_score_with_cross_encoder_score(self):
        """The .score field after reranking is the cross-encoder logit, not the RRF score."""
        reranker = self._reranker_with_mock(scores=[3.7, 0.2])
        results = [
            _make_result("a", "text a", score=0.033),
            _make_result("b", "text b", score=0.031),
        ]
        reranked = reranker.rerank("query", results)
        assert reranked[0].score == pytest.approx(3.7)
        assert reranked[1].score == pytest.approx(0.2)

    def test_top_k_limits_output(self):
        reranker = self._reranker_with_mock(scores=[3.0, 2.0, 1.0])
        results = [_make_result(str(i), f"text {i}") for i in range(3)]
        reranked = reranker.rerank("query", results, top_k=2)
        assert len(reranked) == 2

    def test_empty_input_returns_empty(self):
        reranker = CrossEncoderReranker()
        assert reranker.rerank("query", []) == []

    def test_preserves_all_result_fields(self):
        """Reranking must not lose metadata (chapter, section, page_range, subsection)."""
        reranker = self._reranker_with_mock(scores=[1.0])
        original = SearchResult(
            chunk_id="x",
            text="some text",
            chapter="Chapter 14",
            section="Event-Driven",
            page_range=(200, 215),
            score=0.031,
            subsection="Broker Topology",
        )
        reranked = reranker.rerank("query", [original])
        r = reranked[0]
        assert r.chunk_id == "x"
        assert r.chapter == "Chapter 14"
        assert r.section == "Event-Driven"
        assert r.page_range == (200, 215)
        assert r.subsection == "Broker Topology"
        assert r.text == "some text"

    def test_model_loaded_lazily(self):
        """Model should not be loaded at construction time."""
        reranker = CrossEncoderReranker()
        assert reranker._model is None

    def test_predict_called_with_query_doc_pairs(self):
        """predict() must receive (query, text) pairs, not just texts."""
        reranker = self._reranker_with_mock(scores=[1.0, 2.0])
        results = [_make_result("a", "alpha"), _make_result("b", "beta")]
        reranker.rerank("my query", results)
        call_args = reranker._model.predict.call_args[0][0]
        assert call_args == [("my query", "alpha"), ("my query", "beta")]


class TestVectorStoreWithReranker:
    """Integration: VectorStore.search() delegates to reranker when one is attached."""

    def test_search_uses_reranker_when_attached(self):
        from bookmind_tutor.retrieval.vector_store import VectorStore

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            _make_result("reranked_top", "reranked text", score=9.9)
        ]

        vs = VectorStore(reranker=mock_reranker)
        # Seed with a couple of chunks
        from bookmind_tutor.ingestion.models import Chunk
        chunks = [
            Chunk(
                chunk_id=f"c{i}",
                text=f"text about topic {i}",
                chapter="Ch1",
                section="Sec1",
                page_range=(i, i + 1),
                char_offset_start=0,
                char_offset_end=20,
            )
            for i in range(5)
        ]
        vs.index_chunks(chunks)

        results = vs.search("topic", k=2)

        assert mock_reranker.rerank.called
        call_kwargs = mock_reranker.rerank.call_args
        assert call_kwargs[1]["top_k"] == 2 or call_kwargs[0][2] == 2
        # The returned results are exactly what the reranker returned
        assert results[0].chunk_id == "reranked_top"

    def test_search_without_reranker_skips_rerank(self):
        """No reranker → no rerank call, just RRF results."""
        from bookmind_tutor.retrieval.vector_store import VectorStore
        from bookmind_tutor.ingestion.models import Chunk

        vs = VectorStore()
        chunks = [
            Chunk(
                chunk_id=f"c{i}",
                text=f"text {i}",
                chapter="Ch1",
                section="Sec1",
                page_range=(i, i + 1),
                char_offset_start=0,
                char_offset_end=10,
            )
            for i in range(5)
        ]
        vs.index_chunks(chunks)
        results = vs.search("text", k=3)
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)
