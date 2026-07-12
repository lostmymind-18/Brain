"""
ChromaDB-backed vector store for book chunks.

Uses ChromaDB's DefaultEmbeddingFunction (sentence-transformers/all-MiniLM-L6-v2)
so no external embedding API key is required.

Search uses hybrid BM25 + dense vector fusion (rank_bm25 package required).
Falls back to pure cosine similarity if rank_bm25 is not installed.

Contextual embedding: chunks are EMBEDDED (and BM25-tokenized) with their heading
breadcrumb prepended ("Chapter > Section > Subsection\\n<text>"), while the STORED
document text stays clean for display. This makes a chunk inside "Broker Topology"
findable by the query "broker topology" even when the chunk text never contains
those words. A lightweight version of Anthropic's contextual retrieval technique -
the context comes free from structure detection instead of an LLM call.
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


def _context_text(
    text: str,
    chapter: str | None,
    section: str | None,
    subsection: str | None,
) -> str:
    """
    Prepend the heading breadcrumb to text for embedding / BM25 tokenization.

    The breadcrumb (~10-20 words) plus the chunk text must stay under the
    embedding model's ~195-word truncation point; the chunker's max_tokens
    default (160) leaves headroom for this.
    """
    crumbs = " > ".join(x for x in (chapter, section, subsection) if x)
    return f"{crumbs}\n{text}" if crumbs else text


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
        reranker: Optional CrossEncoderReranker. When provided, search() fetches
            n_candidates = min(k * 8, total) via RRF and then reranks them to
            return the top-k most relevant results. Without a reranker, search()
            returns the top-k RRF results directly.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        reranker=None,
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
        self._reranker = reranker
        logger.debug("VectorStore ready: collection=%s, count=%d", self._collection_name, self.count())

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """
        Add chunks to the collection.

        Uses upsert so re-indexing the same PDF is idempotent.
        Chunks are batched in groups of 100 to stay within ChromaDB limits.

        Embeddings are computed from the CONTEXT text (heading breadcrumb +
        chunk text) and passed explicitly, while `documents` stores the clean
        chunk text. ChromaDB uses provided embeddings as-is and does not
        re-embed the documents.
        """
        if not chunks:
            return

        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            context_docs = [
                _context_text(c.text, c.chapter, c.section, c.subsection)
                for c in batch
            ]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                embeddings=_DEFAULT_EF(context_docs),
                metadatas=[
                    {
                        "chapter": c.chapter or "",
                        "section": c.section or "",
                        "subsection": c.subsection or "",
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

        Two-stage when a reranker is attached:
          Stage 1: RRF hybrid retrieval of min(k*8, total) candidates.
          Stage 2: Cross-encoder reranking → top-k returned.

        Without a reranker: single-stage RRF hybrid (or pure dense if rank_bm25
        is not installed). Falls back to pure dense cosine if rank_bm25 unavailable.

        Results are sorted by score descending. Returns [] if collection is empty.
        """
        total = self.count()
        if total == 0:
            return []

        if self._reranker is not None:
            # Fetch a larger candidate set for the reranker to select from.
            n_candidates = min(k * 8, total)
            bm25 = self._get_bm25()
            candidates = (
                self._search_hybrid(query, n_candidates, alpha, bm25)
                if bm25 is not None
                else self._search_dense(query, n_candidates)
            )
            return self._reranker.rerank(query, candidates, top_k=k)

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
                subsection=meta.get("subsection") or None,
            ))
        return results

    def _search_hybrid(self, query: str, k: int, alpha: float, bm25) -> list[SearchResult]:
        """
        Fused BM25 + dense search using Reciprocal Rank Fusion (RRF).

        RRF score: 1/(rrf_k + rank_dense) + 1/(rrf_k + rank_bm25)
        rrf_k=60 is the standard constant from Cormack et al. 2009.

        Advantages over the previous max-norm alpha blend:
        - Scale-free: BM25 raw scores are not comparable to cosine similarities,
          so normalizing by max-score made the top BM25 doc always dominate.
          (Fingerprint of the old bug: exactly-0.500 hybrid score when BM25 ranked
          a doc first but dense did not surface it at all.)
        - k-independent: RRF rank is stable regardless of how many candidates
          are fetched; alpha blend depended on the distribution of the candidate set.

        The alpha parameter is kept in the signature for API compatibility but is
        no longer used in fusion. It may be repurposed later (e.g. to weight the
        two rank lists differently via 1/(rrf_k + rank_dense) * alpha).
        """
        total = self.count()
        n_candidates = min(k * 3, total)
        rrf_k = 60  # standard Cormack et al. 2009 constant

        # --- Dense candidates ---
        dense_result = self._collection.query(query_texts=[query], n_results=n_candidates)
        dense_ids = dense_result["ids"][0]
        # rank is 0-based; lower rank = better
        dense_rank: dict[str, int] = {cid: rank for rank, cid in enumerate(dense_ids)}
        doc_map: dict[str, str] = dict(zip(dense_ids, dense_result["documents"][0]))
        meta_map: dict[str, dict] = dict(zip(dense_ids, dense_result["metadatas"][0]))

        # --- BM25 candidates ---
        raw_scores = bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)[:n_candidates]
        bm25_rank: dict[str, int] = {
            self._bm25_ids[i]: rank for rank, i in enumerate(top_indices)
        }

        # Fetch docs/metas for BM25 candidates not already in dense results
        missing = [cid for cid in bm25_rank if cid not in doc_map]
        if missing:
            fetched = self._collection.get(ids=missing, include=["documents", "metadatas"])
            for cid, doc, meta in zip(
                fetched["ids"], fetched["documents"], fetched["metadatas"]
            ):
                doc_map[cid] = doc
                meta_map[cid] = meta

        # --- RRF fusion ---
        all_ids = set(dense_rank.keys()) | set(bm25_rank.keys())
        results: list[SearchResult] = []
        for cid in all_ids:
            # Docs absent from one list get rank = n_candidates (worst possible)
            dr = dense_rank.get(cid, n_candidates)
            br = bm25_rank.get(cid, n_candidates)
            rrf_score = 1.0 / (rrf_k + dr) + 1.0 / (rrf_k + br)
            meta = meta_map.get(cid, {})
            results.append(SearchResult(
                chunk_id=cid,
                text=doc_map.get(cid, ""),
                chapter=meta.get("chapter") or None,
                section=meta.get("section") or None,
                page_range=(meta.get("page_start", 0), meta.get("page_end", 0)),
                score=rrf_score,
                subsection=meta.get("subsection") or None,
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

        raw = self._collection.get(include=["documents", "metadatas"])
        if not raw["ids"]:
            return None

        self._bm25_ids = raw["ids"]
        # Tokenize the context text (breadcrumb + body) so heading words are
        # searchable even when they never appear in the chunk body.
        tokenized = [
            _context_text(
                doc,
                meta.get("chapter"),
                meta.get("section"),
                meta.get("subsection"),
            ).lower().split()
            for doc, meta in zip(raw["documents"], raw["metadatas"])
        ]
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
