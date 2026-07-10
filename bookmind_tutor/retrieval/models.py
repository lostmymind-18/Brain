"""Data models for the retrieval module."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchResult:
    """A chunk returned by a vector store search."""
    chunk_id: str
    text: str
    chapter: str | None
    section: str | None
    page_range: tuple[int, int]
    score: float  # cosine similarity; higher = more relevant
    subsection: str | None = None  # L2 heading; None for older indexes without this field
