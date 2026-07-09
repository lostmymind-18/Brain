"""
Unit tests for HierarchicalChunker using synthetic DocumentTree objects.

The most critical requirement - no mid-sentence cuts - is tested explicitly
with a crafted example. All tests use fake DocumentNode objects (no PDF needed).
"""
from __future__ import annotations

import hashlib

import pytest

from bookmind_tutor.ingestion import (
    Chunk,
    DocumentNode,
    DocumentTree,
    HierarchicalChunker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def leaf_node(
    title: str | None,
    level: int,
    raw_text: str,
    page_range: tuple[int, int] = (1, 1),
) -> DocumentNode:
    return DocumentNode(
        title=title,
        level=level,
        page_range=page_range,
        children=[],
        raw_text=raw_text,
    )


MULTI_SENTENCE_TEXT = (
    "The mitochondria is the powerhouse of the cell. "
    "It produces ATP through oxidative phosphorylation. "
    "This process occurs in the inner mitochondrial membrane. "
    "The electron transport chain is a key component. "
    "Protons flow through ATP synthase to generate energy. "
    "This is one of the most important processes in biology. "
    "Without it, eukaryotic life would not be possible. "
    "The discovery of this mechanism earned a Nobel Prize. "
    "Peter Mitchell proposed the chemiosmotic theory in 1961. "
    "His work was initially controversial but later accepted."
)


# ---------------------------------------------------------------------------
# Basic chunking
# ---------------------------------------------------------------------------

class TestBasicChunking:
    def test_returns_list_of_chunks(self):
        node = leaf_node("Chapter 1", 0, "Short text.")
        tree = DocumentTree(root_nodes=[node])
        chunks = HierarchicalChunker().chunk(tree)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_empty_tree_returns_empty_list(self):
        chunks = HierarchicalChunker().chunk(DocumentTree(root_nodes=[]))
        assert chunks == []

    def test_empty_raw_text_produces_no_chunks(self):
        node = leaf_node("Chapter", 0, "   \n  ")
        chunks = HierarchicalChunker().chunk(DocumentTree(root_nodes=[node]))
        assert chunks == []

    def test_short_text_is_single_chunk(self):
        text = "One sentence only."
        node = leaf_node("Intro", 0, text)
        chunks = HierarchicalChunker(max_tokens=512).chunk(DocumentTree(root_nodes=[node]))
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_chunk_text_preserves_content(self):
        node = leaf_node("Ch", 0, MULTI_SENTENCE_TEXT)
        tree = DocumentTree(root_nodes=[node])
        chunks = HierarchicalChunker(max_tokens=512).chunk(tree)
        # All non-overlapping content should appear somewhere in chunks
        assert any("mitochondria" in c.text for c in chunks)
        assert any("Nobel" in c.text for c in chunks)


# ---------------------------------------------------------------------------
# No mid-sentence cuts (critical requirement)
# ---------------------------------------------------------------------------

class TestNoMidSentenceCuts:
    def test_chunks_end_at_sentence_boundaries(self):
        """
        With a low max_tokens, the chunker MUST split. Each chunk boundary
        must align with a sentence end (period/question mark/exclamation).
        """
        node = leaf_node("Section", 1, MULTI_SENTENCE_TEXT)
        tree = DocumentTree(root_nodes=[node])
        # Set a small window to force multiple chunks
        chunks = HierarchicalChunker(max_tokens=20, overlap_tokens=5).chunk(tree)

        assert len(chunks) > 1, "should have produced multiple chunks with max_tokens=20"
        for chunk in chunks:
            stripped = chunk.text.strip()
            # Each chunk must end with a sentence-terminating character
            assert stripped[-1] in ".!?", (
                f"Chunk does not end at a sentence boundary: ...{stripped[-30:]!r}"
            )

    def test_no_sentence_appears_split_across_chunks(self):
        """
        Verify by reconstruction: joining consecutive chunks (ignoring overlap)
        should contain all original sentences completely.
        """
        import nltk
        sentences = nltk.sent_tokenize(MULTI_SENTENCE_TEXT)

        node = leaf_node("Section", 1, MULTI_SENTENCE_TEXT)
        chunks = HierarchicalChunker(max_tokens=25, overlap_tokens=5).chunk(
            DocumentTree(root_nodes=[node])
        )

        # Every sentence must appear fully (not truncated) in at least one chunk
        for sent in sentences:
            found = any(sent in chunk.text for chunk in chunks)
            assert found, f"Sentence not found intact in any chunk: {sent!r}"

    def test_single_very_long_sentence_becomes_its_own_chunk(self):
        """If a sentence alone exceeds max_tokens, it must still be emitted as-is."""
        long_sent = "word " * 100 + "end."
        short_sent = "Short one."
        text = long_sent + " " + short_sent
        node = leaf_node("Ch", 0, text)
        chunks = HierarchicalChunker(max_tokens=30, overlap_tokens=5).chunk(
            DocumentTree(root_nodes=[node])
        )
        # The long sentence must appear as a complete chunk, not split
        assert any(long_sent.strip() in c.text for c in chunks)


# ---------------------------------------------------------------------------
# Metadata correctness
# ---------------------------------------------------------------------------

class TestChunkMetadata:
    def test_chapter_metadata_from_level_0_node(self):
        node = leaf_node("Chapter 3", 0, "Some content here.")
        chunks = HierarchicalChunker().chunk(DocumentTree(root_nodes=[node]))
        assert all(c.chapter == "Chapter 3" for c in chunks)
        assert all(c.section is None for c in chunks)

    def test_section_inherits_parent_chapter(self):
        chapter = DocumentNode(
            title="Chapter 1", level=0,
            page_range=(1, 5), children=[], raw_text=""
        )
        section = leaf_node("Section 1.1", 1, "Section body text.", page_range=(2, 3))
        chapter.children.append(section)

        chunks = HierarchicalChunker().chunk(DocumentTree(root_nodes=[chapter]))
        section_chunks = [c for c in chunks if c.section == "Section 1.1"]
        assert section_chunks
        assert all(c.chapter == "Chapter 1" for c in section_chunks)

    def test_page_range_passed_through(self):
        node = leaf_node("Ch", 0, "Text.", page_range=(7, 9))
        chunks = HierarchicalChunker().chunk(DocumentTree(root_nodes=[node]))
        assert all(c.page_range == (7, 9) for c in chunks)

    def test_char_offsets_within_raw_text(self):
        text = "First sentence here. Second sentence there."
        node = leaf_node("Ch", 0, text, page_range=(1, 1))
        chunks = HierarchicalChunker(max_tokens=512).chunk(DocumentTree(root_nodes=[node]))
        for chunk in chunks:
            assert chunk.char_offset_start >= 0
            assert chunk.char_offset_end <= len(text)
            assert chunk.char_offset_start < chunk.char_offset_end
            # Offsets must point to the chunk text within raw_text
            assert text[chunk.char_offset_start:chunk.char_offset_end] == chunk.text

    def test_chunk_ids_are_deterministic(self):
        node = leaf_node("Ch", 0, MULTI_SENTENCE_TEXT)
        tree = DocumentTree(root_nodes=[node])
        chunks_1 = HierarchicalChunker(max_tokens=30).chunk(tree)
        chunks_2 = HierarchicalChunker(max_tokens=30).chunk(tree)
        assert [c.chunk_id for c in chunks_1] == [c.chunk_id for c in chunks_2]

    def test_chunk_ids_are_unique(self):
        node = leaf_node("Ch", 0, MULTI_SENTENCE_TEXT)
        chunks = HierarchicalChunker(max_tokens=30).chunk(DocumentTree(root_nodes=[node]))
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "chunk IDs should be unique"


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

class TestOverlap:
    def test_overlap_sentences_repeated_in_next_chunk(self):
        """The last sentence(s) of chunk N should appear at the start of chunk N+1."""
        node = leaf_node("Ch", 0, MULTI_SENTENCE_TEXT)
        chunks = HierarchicalChunker(max_tokens=20, overlap_tokens=10).chunk(
            DocumentTree(root_nodes=[node])
        )
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")

        for i in range(len(chunks) - 1):
            # The last sentence of chunk[i] should appear somewhere in chunk[i+1]
            import nltk
            last_sent = nltk.sent_tokenize(chunks[i].text)[-1]
            assert last_sent in chunks[i + 1].text, (
                f"No overlap found between chunk {i} and {i + 1}"
            )


# ---------------------------------------------------------------------------
# Multiple nodes / tree traversal
# ---------------------------------------------------------------------------

class TestTreeTraversal:
    def test_chunks_from_all_nodes(self):
        ch1 = DocumentNode("Chapter 1", 0, (1, 3), [], "Chapter one body text.")
        ch2 = DocumentNode("Chapter 2", 0, (4, 6), [], "Chapter two body text.")
        tree = DocumentTree(root_nodes=[ch1, ch2])
        chunks = HierarchicalChunker().chunk(tree)
        chapters = {c.chapter for c in chunks}
        assert "Chapter 1" in chapters
        assert "Chapter 2" in chapters

    def test_deep_nesting_does_not_crash(self):
        # Three levels: chapter -> section -> subsection
        chapter = DocumentNode("Ch", 0, (1, 10), [], "Intro text.")
        section = DocumentNode("Sec", 1, (2, 8), [], "Section text.")
        subsection = DocumentNode("Sub", 2, (3, 5), [], "Subsection text.")
        section.children.append(subsection)
        chapter.children.append(section)

        chunks = HierarchicalChunker().chunk(DocumentTree(root_nodes=[chapter]))
        assert len(chunks) >= 3  # one per node (each has content)
