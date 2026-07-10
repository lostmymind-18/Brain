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



# Internal/appendix labels excluded from the outline (TOC page, Copyright, etc.)
_SKIP_LABELS = {"Table of Contents", "Copyright", "Cover", "Colophon", "Index"}


class BookOutlineTool(Tool):
    """
    Return the complete table of contents derived from indexed chunk metadata.

    Reconstructs the part -> chapter -> subsection hierarchy from the `chapter`,
    `section`, and `subsection` metadata fields stored in ChromaDB during
    ingestion. Results are sorted by first page number so the outline preserves
    book order.

    Subsections are included so the LLM can discover exact heading names to pass
    to get_book_section - without them, sub-chapter headings exist in the index
    but the model has no way to learn their names.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="book_outline",
            description=(
                "Returns the complete table of contents of the book — all parts, "
                "chapters, and section headings in order, with page numbers. Use "
                "this tool (instead of book_search) whenever the user asks about: "
                "book structure, chapter list, what topics are covered, which part "
                "contains which chapters, or how many chapters the book has. Also "
                "useful to find the exact heading name to pass to get_book_section."
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

        # Nested min-page maps: chapter -> sections -> subsections.
        # chapter="" means the chunk belongs to no named part (skip those).
        # section="" with a non-empty subsection still nests the subsection
        # under the chapter (the "" section bucket is not rendered itself).
        chapters: dict[str, dict] = {}
        for m in metadatas:
            chapter = (m.get("chapter") or "").strip()
            section = (m.get("section") or "").strip()
            subsection = (m.get("subsection") or "").strip()
            page = m.get("page_start", 0)
            if not chapter:
                continue

            ch = chapters.setdefault(chapter, {"page": page, "sections": {}})
            ch["page"] = min(ch["page"], page)

            sec = ch["sections"].setdefault(section, {"page": page, "subs": {}})
            sec["page"] = min(sec["page"], page)

            if subsection:
                sec["subs"][subsection] = min(
                    sec["subs"].get(subsection, page), page
                )

        if not chapters:
            return "Book structure could not be determined from indexed metadata."

        lines: list[str] = []
        for chapter, ch in sorted(chapters.items(), key=lambda kv: kv[1]["page"]):
            if chapter in _SKIP_LABELS:
                continue
            lines.append(f"\n**{chapter}** (p. {ch['page']})")
            for section, sec in sorted(
                ch["sections"].items(), key=lambda kv: kv[1]["page"]
            ):
                if section:
                    lines.append(f"  - {section} (p. {sec['page']})")
                    sub_indent = "    - "
                else:
                    # Unnamed section bucket: render its subsections one level up
                    sub_indent = "  - "
                for sub, pg in sorted(sec["subs"].items(), key=lambda kv: kv[1]):
                    lines.append(f"{sub_indent}{sub} (p. {pg})")

        return "\n".join(lines).strip()
