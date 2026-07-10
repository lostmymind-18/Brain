"""
HierarchicalChunker: splits a DocumentTree into retrieval-ready Chunks.

Design decisions:
- Sentence boundaries: NLTK's sent_tokenize is used so chunks never break
  mid-sentence. This adds a small startup cost (NLTK model download on first
  run) but is lighter than spaCy for this use case.
- Token counting: we use whitespace-split word count as an approximation for
  BPE tokens. Actual BPE token counts are ~1.3x word count for English prose.
  all-MiniLM-L6-v2 truncates at 256 BPE (~195 prose words), measured
  empirically: embeddings of texts that differ only after that point are
  byte-identical, i.e. the tail is silently discarded. The original default
  of 512 caused 52.8% of FoSA chunks to have un-embedded tails.
  Default max_tokens=160 leaves ~35 words of headroom for the heading
  breadcrumb that VectorStore prepends at embedding time (contextual
  embedding), keeping breadcrumb + text under the truncation point.
- Overlap: the last overlap_tokens words of each chunk are repeated at the
  start of the next chunk so that context that spans a chunk boundary is still
  retrievable in full.
- Chunk IDs: MD5 hash of (text prefix + position metadata). Stable as long as
  the document and chunking parameters don't change - sufficient for caching
  and dedup in Week 4's Knowledge Graph work.
"""
from __future__ import annotations

import hashlib

import nltk

from .models import Chunk, DocumentNode, DocumentTree


def _ensure_nltk_punkt() -> None:
    """Download the NLTK sentence tokenizer data if not already present."""
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            return
        except LookupError:
            pass
    # Try downloading punkt_tab first (NLTK >= 3.8.1), fall back to punkt
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        nltk.download("punkt", quiet=True)


class HierarchicalChunker:
    """
    Chunks a DocumentTree into a flat list of Chunks.

    The chunker recurses through the tree, chunking each node's raw_text
    independently, then chunks children nodes with the correct chapter/section
    metadata inherited from their ancestors.

    Args:
        max_tokens: Maximum whitespace-token count per chunk (default 160).
        overlap_tokens: Token overlap between consecutive chunks (default 30).
    """

    def __init__(self, max_tokens: int = 160, overlap_tokens: int = 30) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        _ensure_nltk_punkt()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, tree: DocumentTree) -> list[Chunk]:
        """Produce a flat list of Chunks from the document tree."""
        chunks: list[Chunk] = []
        for node in tree.root_nodes:
            chapter = node.title if node.level == 0 else None
            section = node.title if node.level == 1 else None
            subsection = node.title if node.level == 2 else None
            chunks.extend(self._chunk_node(node, chapter=chapter, section=section, subsection=subsection))
        return chunks

    # ------------------------------------------------------------------
    # Recursive tree traversal
    # ------------------------------------------------------------------

    def _chunk_node(
        self,
        node: DocumentNode,
        chapter: str | None,
        section: str | None,
        subsection: str | None = None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        # Chunk this node's own text (text that belongs directly to it, not to children)
        if node.raw_text.strip():
            chunks.extend(
                self._split_text(node.raw_text, chapter, section, subsection, node.page_range)
            )

        # Recurse into children, propagating chapter/section/subsection metadata
        for child in node.children:
            if child.level == 0:
                child_chapter, child_section, child_subsection = child.title, None, None
            elif child.level == 1:
                child_chapter = chapter  # inherit ancestor chapter
                child_section = child.title
                child_subsection = None
            elif child.level == 2:
                child_chapter = chapter
                child_section = section
                child_subsection = child.title
            else:
                # Level 3+: inherit all from ancestor
                child_chapter, child_section, child_subsection = chapter, section, subsection

            chunks.extend(self._chunk_node(child, child_chapter, child_section, child_subsection))

        return chunks

    # ------------------------------------------------------------------
    # Text splitting
    # ------------------------------------------------------------------

    def _split_text(
        self,
        text: str,
        chapter: str | None,
        section: str | None,
        subsection: str | None,
        page_range: tuple[int, int],
    ) -> list[Chunk]:
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return []

        positions = self._locate_sentences(text, sentences)
        chunks: list[Chunk] = []
        chunk_start_idx = 0
        running_tokens = 0

        for i, sent in enumerate(sentences):
            sent_tokens = self._count_tokens(sent)

            # Flush when the window is full and we have at least one sentence in it
            if running_tokens + sent_tokens > self.max_tokens and i > chunk_start_idx:
                chunks.append(
                    self._make_chunk(
                        text, sentences, positions,
                        chunk_start_idx, i - 1,
                        chapter, section, subsection, page_range,
                    )
                )
                chunk_start_idx = self._compute_overlap_start(
                    sentences, chunk_start_idx, i - 1
                )
                running_tokens = sum(
                    self._count_tokens(sentences[j])
                    for j in range(chunk_start_idx, i)
                )

            running_tokens += sent_tokens

        # Final chunk - whatever is left in the window
        if chunk_start_idx < len(sentences):
            chunks.append(
                self._make_chunk(
                    text, sentences, positions,
                    chunk_start_idx, len(sentences) - 1,
                    chapter, section, subsection, page_range,
                )
            )

        return chunks

    def _compute_overlap_start(
        self,
        sentences: list[str],
        chunk_start: int,
        chunk_end: int,
    ) -> int:
        """
        Find the sentence index where the next chunk should begin so that the
        last ~overlap_tokens of the current chunk are repeated as context.
        """
        accumulated = 0
        idx = chunk_end
        while idx > chunk_start:
            accumulated += self._count_tokens(sentences[idx])
            if accumulated >= self.overlap_tokens:
                return idx
            idx -= 1
        return chunk_start

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Approximate BPE token count using whitespace split word count."""
        return len(text.split())

    @staticmethod
    def _locate_sentences(
        text: str, sentences: list[str]
    ) -> list[tuple[int, int]]:
        """
        Find the (start, end) character offsets of each sentence within text.

        Uses a linear scan so the positions stay consistent even when the same
        sentence appears multiple times in a node.
        """
        positions: list[tuple[int, int]] = []
        cursor = 0
        for sent in sentences:
            start = text.find(sent, cursor)
            if start == -1:
                # Fallback: search from the very beginning (shouldn't happen in practice)
                start = text.find(sent)
            if start == -1:
                start = cursor  # last resort - keeps the list length consistent
            end = start + len(sent)
            positions.append((start, end))
            cursor = end
        return positions

    def _make_chunk(
        self,
        text: str,
        sentences: list[str],
        positions: list[tuple[int, int]],
        start_idx: int,
        end_idx: int,
        chapter: str | None,
        section: str | None,
        subsection: str | None,
        page_range: tuple[int, int],
    ) -> Chunk:
        char_start = positions[start_idx][0]
        char_end = positions[end_idx][1]
        chunk_text = text[char_start:char_end]

        return Chunk(
            chunk_id=self._make_chunk_id(chunk_text, chapter, section, page_range, char_start, char_end),
            text=chunk_text,
            chapter=chapter,
            section=section,
            page_range=page_range,
            char_offset_start=char_start,
            char_offset_end=char_end,
            subsection=subsection,
        )

    @staticmethod
    def _make_chunk_id(
        text: str,
        chapter: str | None,
        section: str | None,
        page_range: tuple[int, int],
        char_start: int,
        char_end: int,
    ) -> str:
        """
        Deterministic ID based on content + position so the same chunk always
        gets the same ID regardless of re-processing order.
        """
        # Include both endpoints so chunks that share a start (due to heavy overlap)
        # still get distinct IDs.
        fingerprint = (
            f"{chapter or ''}|{section or ''}|"
            f"{page_range[0]}-{page_range[1]}|{char_start}-{char_end}|"
            f"{text[:200]}"
        )
        return hashlib.md5(fingerprint.encode("utf-8")).hexdigest()
