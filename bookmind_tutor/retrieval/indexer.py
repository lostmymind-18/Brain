"""
ChunkIndexer: orchestrates the full ingestion-to-retrieval pipeline.

PdfExtractor → StructureDetector → HierarchicalChunker → VectorStore
"""
from __future__ import annotations

import logging

from bookmind_tutor.ingestion import HierarchicalChunker, PdfExtractor, StructureDetector
from bookmind_tutor.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ChunkIndexer:
    """
    Convenience wrapper that takes a PDF path and indexes it into a VectorStore.

    Args:
        vector_store: The VectorStore instance to write chunks into.
        llm_client: Optional anthropic.Anthropic instance.
            If provided, enables Tier 2 (LLM TOC parse) and Tier 3 LLM verification.
            If None, falls back to heuristic-only structure detection (Tier 3).
    """

    def __init__(self, vector_store: VectorStore, llm_client=None) -> None:
        self._store = vector_store
        self._llm_client = llm_client
        self._extractor = PdfExtractor()
        self._chunker = HierarchicalChunker()

    def index_pdf(self, pdf_path: str) -> int:
        """
        Parse the PDF, chunk it, and upsert all chunks into the vector store.

        Returns the number of chunks indexed.
        Raises FileNotFoundError if pdf_path does not exist.
        """
        logger.info("Indexing PDF: %s", pdf_path)
        pages = self._extractor.extract(pdf_path)
        toc = self._extractor.extract_toc(pdf_path)

        detector = StructureDetector(llm_client=self._llm_client)
        tree = detector.detect(pages, toc=toc)

        chunks = self._chunker.chunk(tree)
        logger.info("Generated %d chunks from %d pages", len(chunks), len(pages))

        self._store.index_chunks(chunks)
        return len(chunks)
