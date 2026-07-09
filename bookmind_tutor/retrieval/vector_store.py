"""
ChromaDB-backed vector store for book chunks.

Uses ChromaDB's DefaultEmbeddingFunction (sentence-transformers/all-MiniLM-L6-v2)
so no external embedding API key is required.
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
        logger.info("Indexed %d chunks (total in store: %d)", len(chunks), self.count())

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """
        Return up to k most relevant chunks for query.

        Results are sorted by cosine similarity descending.
        Returns empty list if the collection is empty.
        """
        n = min(k, self.count())
        if n == 0:
            return []

        result = self._collection.query(query_texts=[query], n_results=n)

        results: list[SearchResult] = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        # ChromaDB returns distances (lower = more similar for cosine); convert to similarity score
        distances = result["distances"][0]

        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            score = 1.0 - dist  # cosine distance → cosine similarity
            page_start = meta.get("page_start", 0)
            page_end = meta.get("page_end", 0)
            results.append(
                SearchResult(
                    chunk_id=cid,
                    text=doc,
                    chapter=meta.get("chapter") or None,
                    section=meta.get("section") or None,
                    page_range=(page_start, page_end),
                    score=score,
                )
            )
        return results

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
        logger.debug("VectorStore cleared.")

    def drop(self) -> None:
        """Delete the collection entirely and release the ChromaDB handle."""
        try:
            self._client.delete_collection(self._collection_name)
            logger.debug("VectorStore dropped: %s", self._collection_name)
        except Exception as exc:
            logger.warning("Could not drop collection %s: %s", self._collection_name, exc)
