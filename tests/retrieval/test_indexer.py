"""Tests for ChunkIndexer."""
from unittest.mock import MagicMock, patch

import pytest

from bookmind_tutor.ingestion.models import Chunk, DocumentNode, DocumentTree
from bookmind_tutor.retrieval.indexer import ChunkIndexer
from bookmind_tutor.retrieval.vector_store import VectorStore


def _fake_chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"chunk-{i}",
        text=f"Text content {i}",
        chapter="Chapter 1",
        section=None,
        page_range=(i, i),
        char_offset_start=0,
        char_offset_end=10,
    )


def _fake_tree() -> DocumentTree:
    node = DocumentNode(title="Chapter 1", level=0, page_range=(1, 5),
                        children=[], raw_text="Content here.")
    return DocumentTree(root_nodes=[node])


class TestChunkIndexer:
    @patch("bookmind_tutor.retrieval.indexer.HierarchicalChunker")
    @patch("bookmind_tutor.retrieval.indexer.StructureDetector")
    @patch("bookmind_tutor.retrieval.indexer.PdfExtractor")
    def test_index_pdf_calls_pipeline(self, MockExtractor, MockDetector, MockChunker):
        fake_pages = [MagicMock()]
        fake_toc = [MagicMock()]
        fake_tree = _fake_tree()
        fake_chunks = [_fake_chunk(i) for i in range(3)]

        MockExtractor.return_value.extract.return_value = fake_pages
        MockExtractor.return_value.extract_toc.return_value = fake_toc
        MockDetector.return_value.detect.return_value = fake_tree
        MockChunker.return_value.chunk.return_value = fake_chunks

        store = VectorStore(persist_dir=None)
        indexer = ChunkIndexer(vector_store=store)
        count = indexer.index_pdf("/fake/path.pdf")

        assert count == 3
        assert store.count() == 3

    @patch("bookmind_tutor.retrieval.indexer.HierarchicalChunker")
    @patch("bookmind_tutor.retrieval.indexer.StructureDetector")
    @patch("bookmind_tutor.retrieval.indexer.PdfExtractor")
    def test_index_pdf_returns_chunk_count(self, MockExtractor, MockDetector, MockChunker):
        MockExtractor.return_value.extract.return_value = []
        MockExtractor.return_value.extract_toc.return_value = []
        MockDetector.return_value.detect.return_value = _fake_tree()
        MockChunker.return_value.chunk.return_value = [_fake_chunk(0)]

        store = VectorStore(persist_dir=None)
        indexer = ChunkIndexer(store)
        assert indexer.index_pdf("/fake/path.pdf") == 1
