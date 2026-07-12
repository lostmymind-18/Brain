"""
Cross-encoder reranker for the two-stage retrieval pipeline.

Stage 1 (VectorStore.search): bi-encoder retrieval — fast, recall-optimized.
  Retrieve a large candidate set (k*8) using RRF-fused BM25 + dense vectors.

Stage 2 (CrossEncoderReranker.rerank): cross-encoder scoring — slower, precision-optimized.
  Score each (query, document) pair jointly; tokens attend across both sides,
  capturing fine-grained relevance that independent bi-encoder encoding misses.

Why two stages: bi-encoder can pre-compute all document vectors at index time (O(1)
per query). Cross-encoder cannot — it must run a forward pass for every (query, doc)
pair, so it is only practical on a small candidate set (30-50 docs, not 2000+).

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking (web search queries + passages).
  - 22M parameters, runs on CPU in ~5-10ms per pair.
  - Works well for technical book QA: it generalises to domain text well enough
    for our recall numbers to improve meaningfully (Anthropic data: -49% → -67%).
  - Local, no API key, no cost at query time.

The model is downloaded once to ~/.cache/huggingface/ on first use.
"""
from __future__ import annotations

import logging

from bookmind_tutor.retrieval.models import SearchResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Reranks a list of SearchResult objects using a cross-encoder model.

    The cross-encoder reads (query, document_text) pairs jointly and assigns
    a relevance score. Results are re-sorted by this score.

    Args:
        model_name: HuggingFace model ID.
            Defaults to ms-marco-MiniLM-L-6-v2 (fast, CPU-friendly, no API needed).
        max_length: Token budget for the (query + document) pair.
            The model supports up to 512. 384 is a safe default that avoids
            truncation for most 160-word chunks plus a typical query.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        max_length: int = 384,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._model = None  # lazy-loaded on first rerank() call

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder model: %s", self._model_name)
        self._model = CrossEncoder(self._model_name, max_length=self._max_length)
        logger.info("Cross-encoder ready")

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Re-score and re-sort results by cross-encoder relevance.

        Args:
            query: The user's query string.
            results: Candidate SearchResults from Stage 1 retrieval.
            top_k: How many to return after reranking. None = return all re-sorted.

        Returns:
            SearchResults sorted by cross-encoder score descending.
            The .score field is replaced with the cross-encoder logit score
            (higher = more relevant; scale is unbounded, typically -10 to +10).
        """
        if not results:
            return results

        self._load()

        pairs = [(query, r.text) for r in results]
        scores = self._model.predict(pairs)

        reranked = [
            SearchResult(
                chunk_id=r.chunk_id,
                text=r.text,
                chapter=r.chapter,
                section=r.section,
                page_range=r.page_range,
                score=float(s),
                subsection=r.subsection,
            )
            for r, s in zip(results, scores)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        logger.debug(
            "Reranked %d candidates → top-%d | top score=%.3f",
            len(results),
            top_k or len(reranked),
            reranked[0].score if reranked else 0.0,
        )
        return reranked
