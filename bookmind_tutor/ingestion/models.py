"""
Data models for the ingestion pipeline.

All models use @dataclass for easy inspection during debugging.
Metadata fields (page_range, char offsets, source) are kept rich
because downstream modules (Week 4 KG, Week 7 evaluation) will need
to trace chunks back to their source.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextSpan:
    """A run of text with uniform font properties, as extracted from PDF."""
    text: str
    font: str
    size: float
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in page units


@dataclass
class PageContent:
    """All text content from a single PDF page."""
    page_number: int       # 1-indexed
    text: str              # full page text, blocks separated by "\n\n"
    spans: list[TextSpan]  # all spans in reading order, used for heading detection


@dataclass
class DocumentNode:
    """
    A node in the document hierarchy (chapter, section, subsection, ...).

    raw_text holds only the text that belongs DIRECTLY to this node - not
    the text of its children. This makes recursive chunking straightforward.
    """
    title: str | None                 # None for preamble nodes (text before first heading)
    level: int                        # 0 = chapter, 1 = section, 2 = subsection, -1 = preamble
    page_range: tuple[int, int]       # (first_page, last_page), 1-indexed, inclusive
    children: list[DocumentNode]
    raw_text: str


@dataclass
class DocumentTree:
    """The parsed structure of a document."""
    root_nodes: list[DocumentNode]


@dataclass
class Chunk:
    """
    A text chunk ready for embedding / retrieval.

    char_offset_start and char_offset_end are byte offsets into the
    PARENT DocumentNode's raw_text (not the PDF page text). Combined
    with page_range, this gives enough info to trace back to the source
    document for Knowledge Graph construction in Week 4.
    """
    chunk_id: str           # stable MD5 hash of content + position metadata
    text: str
    chapter: str | None
    section: str | None
    page_range: tuple[int, int]
    char_offset_start: int
    char_offset_end: int
