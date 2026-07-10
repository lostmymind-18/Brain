"""BookOutlineTool: returns the book's chapter structure from indexed metadata.

This tool solves the "list all chapters" hallucination problem: vector search
cannot reliably surface table-of-contents pages because the TOC embeddings
represent chapter topics, not the concept "table of contents". This tool
derives the outline directly from ChromaDB metadata (chapter + section fields
set during ingestion), which is deterministic and always accurate.
"""
from __future__ import annotations

from bookmind_tutor.agents.models import ToolSpec
from bookmind_tutor.agents.tools.base import Tool
from bookmind_tutor.retrieval.vector_store import VectorStore


class BookOutlineTool(Tool):
    """
    Return the complete table of contents derived from indexed chunk metadata.

    Reconstructs the part -> chapter hierarchy from the `chapter` and `section`
    metadata fields stored in ChromaDB during ingestion. Results are sorted by
    first page number so the outline preserves book order.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="book_outline",
            description=(
                "Returns the complete table of contents of the book — all parts and "
                "chapters in order, with page numbers. Use this tool (instead of "
                "book_search) whenever the user asks about: book structure, chapter "
                "list, what topics are covered, which part contains which chapters, "
                "or how many chapters the book has."
            ),
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )

    def execute(self) -> str:
        """Build and return the outline from metadata. No LLM call, no embeddings."""
        raw = self._store._collection.get(include=["metadatas"])
        metadatas = raw.get("metadatas", [])

        if not metadatas:
            return "No book content indexed yet."

        # Collect unique (chapter, section) pairs with the earliest page seen.
        # chapter="" means the chunk belongs to no named part (skip those).
        seen: dict[tuple[str, str], int] = {}
        for m in metadatas:
            chapter = (m.get("chapter") or "").strip()
            section = (m.get("section") or "").strip()
            page = m.get("page_start", 0)
            if not chapter:
                continue
            key = (chapter, section)
            if key not in seen or page < seen[key]:
                seen[key] = page

        if not seen:
            return "Book structure could not be determined from indexed metadata."

        # Sort by page number to restore reading order.
        ordered = sorted(seen.items(), key=lambda kv: kv[1])

        # Group sections under their parent chapter/part.
        # A row with section=="" is a top-level entry (part header or solo chapter).
        groups: dict[str, list[tuple[str, int]]] = {}  # parent -> [(section, page)]
        group_order: list[str] = []
        group_first_page: dict[str, int] = {}

        for (chapter, section), page in ordered:
            if chapter not in groups:
                groups[chapter] = []
                group_order.append(chapter)
                group_first_page[chapter] = page
            if section:
                groups[chapter].append((section, page))

        lines: list[str] = []
        for chapter in group_order:
            # Skip internal/appendix sections (TOC page, Copyright, etc.)
            skip_labels = {"Table of Contents", "Copyright", "Cover", "Colophon", "Index"}
            if chapter in skip_labels:
                continue
            first_page = group_first_page[chapter]
            lines.append(f"\n**{chapter}** (p. {first_page})")
            for section, pg in groups[chapter]:
                lines.append(f"  - {section} (p. {pg})")

        return "\n".join(lines).strip()
