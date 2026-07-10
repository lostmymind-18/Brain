"""
ChromaDB-backed vector store for book chunks.

Uses ChromaDB's DefaultEmbeddingFunction (sentence-transformers/all-MiniLM-L6-v2)
so no external embedding API key is required.

Search uses hybrid BM25 + dense vector fusion (rank_bm25 package required).
Falls back to pure cosine similarity if rank_bm25 is not installed.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from bookmind_tutor.ingestion.models import Chunk
from bookmind_tutor.retrieval.models import SearchResult

logger = logging.getLogger(__name__)

_DEFAULT_EF = embedding_functions.DefaultEmbeddingFunction()


class VectorStore:
    """
    Stores and retrieves Chunk objects by semantic similarity.

    Args:
        persist_dir: Path to ChromaDB data directory.
            None → ephemeral in-memory collection (useful for tests).
        collection_name: ChromaDB collection name.
            For ephemeral stores (persist_dir=None), a unique UUID name is generated
            automatically to ensure test isolation even if ChromaDB uses a shared
            process-level in-memory store.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        if persist_dir is None:
            self._client = chromadb.EphemeralClient()
            # Use a unique name per instance so multiple ephemeral VectorStores
            # in the same process don't share collection state.
            self._collection_name = collection_name or f"chunks_{uuid.uuid4().hex}"
        else:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection_name = collection_name or "chunks"

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=_DEFAULT_EF,
            metadata={"hnsw:space": "cosine"},
        )
        # BM25 index — built lazily on first search, invalidated when chunks change.
        self._bm25 = None
        self._bm25_ids: list[str] = []
        logger.debug("VectorStore ready: collection=%s, count=%d", self._collection_name, self.count())

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """
        Add chunks to the collection.

        Uses upsert so re-indexing the same PDF is idempotent.
        Chunks are batched in groups of 100 to stay within ChromaDB limits.
        """
        if not chunks:
            return

        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "chapter": c.chapter or "",
                        "section": c.section or "",
                        "page_start": c.page_range[0],
                        "page_end": c.page_range[1],
                    }
                    for c in batch
                ],
            )
        # Invalidate BM25 cache — corpus has changed.
        self._bm25 = None
        self._bm25_ids = []
        logger.info("Indexed %d chunks (total in store: %d)", len(chunks), self.count())

    def search(self, query: str, k: int = 5, alpha: float = 0.5) -> list[SearchResult]:
        """
        Return up to k most relevant chunks for query.

        Uses hybrid BM25 + dense vector fusion when rank_bm25 is installed.
        alpha controls the blend: 1.0 = pure dense, 0.0 = pure BM25.
        Falls back to pure dense cosine similarity if rank_bm25 is not available.

        Results are sorted by fused score descending.
        Returns empty list if the collection is empty.
        """
        total = self.count()
        if total == 0:
            return []

        bm25 = self._get_bm25()
        if bm25 is None:
            return self._search_dense(query, k)
        return self._search_hybrid(query, k, alpha, bm25)

    def _search_dense(self, query: str, k: int) -> list[SearchResult]:
        """Pure dense cosine similarity search."""
        n = min(k, self.count())
        result = self._collection.query(query_texts=[query], n_results=n)

        results: list[SearchResult] = []
        for cid, doc, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            results.append(SearchResult(
                chunk_id=cid,
                text=doc,
                chapter=meta.get("chapter") or None,
                section=meta.get("section") or None,
                page_range=(meta.get("page_start", 0), meta.get("page_end", 0)),
                score=1.0 - dist,
            ))
        return results

    def _search_hybrid(self, query: str, k: int, alpha: float, bm25) -> list[SearchResult]:
        """
        Fused BM25 + dense search using Reciprocal Rank Fusion-style score combination.

        Fetches k*3 candidates from each source, then merges and re-ranks.
        BM25 excels at exact-term/proper-noun matches; dense covers semantic paraphrases.
        """
        total = self.count()
        n_candidates = min(k * 3, total)

        # --- Dense candidates ---
        dense_result = self._collection.query(query_texts=[query], n_results=n_candidates)
        dense_ids = dense_result["ids"][0]
        dense_scores: dict[str, float] = {
            cid: 1.0 - dist
            for cid, dist in zip(dense_ids, dense_result["distances"][0])
        }
        doc_map: dict[str, str] = dict(zip(dense_ids, dense_result["documents"][0]))
        meta_map: dict[str, dict] = dict(zip(dense_ids, dense_result["metadatas"][0]))

        # --- BM25 candidates ---
        raw_scores = bm25.get_scores(query.lower().split())
        max_s = float(max(raw_scores)) if len(raw_scores) > 0 else 0.0
        if max_s > 0:
            normalized = [float(s) / max_s for s in raw_scores]
        else:
            normalized = [0.0] * len(raw_scores)

        # Take top n_candidates from BM25 by descending score
        top_indices = sorted(range(len(normalized)), key=lambda i: normalized[i], reverse=True)[:n_candidates]
        bm25_scores: dict[str, float] = {
            self._bm25_ids[i]: normalized[i] for i in top_indices
        }

        # Fetch docs/metas for BM25 candidates not already in dense results
        missing = [cid for cid in bm25_scores if cid not in doc_map]
        if missing:
            fetched = self._collection.get(ids=missing, include=["documents", "metadatas"])
            for cid, doc, meta in zip(
                fetched["ids"], fetched["documents"], fetched["metadatas"]
            ):
                doc_map[cid] = doc
                meta_map[cid] = meta

        # --- Fuse scores ---
        all_ids = set(dense_scores.keys()) | set(bm25_scores.keys())
        results: list[SearchResult] = []
        for cid in all_ids:
            d = dense_scores.get(cid, 0.0)
            b = bm25_scores.get(cid, 0.0)
            hybrid = alpha * d + (1.0 - alpha) * b
            meta = meta_map.get(cid, {})
            results.append(SearchResult(
                chunk_id=cid,
                text=doc_map.get(cid, ""),
                chapter=meta.get("chapter") or None,
                section=meta.get("section") or None,
                page_range=(meta.get("page_start", 0), meta.get("page_end", 0)),
                score=hybrid,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def _get_bm25(self):
        """
        Build and cache the BM25 index from all documents in the collection.

        Lazy init: runs once on the first search after indexing, then cached
        until index_chunks() or clear() invalidates it.
        Returns None if rank_bm25 is not installed.
        """
        if self._bm25 is not None:
            return self._bm25

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.debug("rank_bm25 not installed — falling back to pure dense search")
            return None

        raw = self._collection.get(include=["documents"])
        if not raw["ids"]:
            return None

        self._bm25_ids = raw["ids"]
        tokenized = [doc.lower().split() for doc in raw["documents"]]
        self._bm25 = BM25Okapi(tokenized)
        logger.debug("BM25 index built: %d documents", len(self._bm25_ids))
        return self._bm25

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return self._collection.count()

    def clear(self) -> None:
        """Remove all chunks from the collection (keeps the collection itself)."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=_DEFAULT_EF,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = None
        self._bm25_ids = []
        logger.debug("VectorStore cleared.")

    def drop(self) -> None:
        """Delete the collection entirely and release the ChromaDB handle."""
        try:
            self._client.delete_collection(self._collection_name)
            logger.debug("VectorStore dropped: %s", self._collection_name)
        except Exception as exc:
            logger.warning("Could not drop collection %s: %s", self._collection_name, exc)
        self._bm25 = None
        self._bm25_ids = []
