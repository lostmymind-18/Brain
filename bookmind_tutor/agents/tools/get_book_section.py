"""GetBookSectionTool: fetch ALL chunks matching a metadata filter.

Complements book_search (top-k semantic) with structured retrieval:
given a part name, chapter name, or page range, return every chunk
that belongs to that scope — not just the top-k most similar ones.

Use cases:
- Summarize a chapter   → field="section", contains="Chapter 5"
- Summarize a part      → field="chapter", contains="Part I"
- Retrieve by page range → page_from / page_to

The LLM should call book_outline first to discover exact field values,
then call this tool with the value it found.
"""
from __future__ import annotations

import logging

from bookmind_tutor.agents.models import ToolSpec
from bookmind_tutor.agents.tools.base import Tool
from bookmind_tutor.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Summarizing a full part (multiple chapters) can easily exceed 50k tokens.
# Cap chunks returned so the LLM context stays manageable.
# The tool notes truncation in its output so the LLM can tell the user.
_MAX_CHUNKS = 30


class GetBookSectionTool(Tool):
    """
    Return all book chunks that belong to a specific part, chapter, or page range.

    Unlike book_search (which returns top-k by semantic similarity), this tool
    uses metadata filtering to guarantee complete coverage of the requested scope.
    Ideal for summarization tasks where missing chunks would produce an incomplete answer.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="get_book_section",
            description=(
                "Retrieve ALL content from a specific part, chapter, or page range of "
                "the book — not just the most relevant excerpts. Use this tool (instead "
                "of book_search) when the user asks to summarize, overview, or explain "
                "an entire chapter or part. "
                "Call book_outline first to find the exact chapter/part names, then pass "
                "those names here. "
                "Fields: 'section' = chapter level (e.g. 'Chapter 5. Pipeline Architecture'), "
                "'chapter' = part level (e.g. 'Part II. Architecture Styles'). "
                "All string matching is case-insensitive substring match."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": ["chapter", "section"],
                        "description": (
                            "Metadata field to filter on. "
                            "'section' matches chapter-level headings; "
                            "'chapter' matches part-level headings."
                        ),
                    },
                    "contains": {
                        "type": "string",
                        "description": (
                            "Substring to match against the field value (case-insensitive). "
                            "Example: 'Chapter 5' or 'Part I' or 'Pipeline'."
                        ),
                    },
                    "page_from": {
                        "type": "integer",
                        "description": "Optional: only include chunks starting at or after this page.",
                    },
                    "page_to": {
                        "type": "integer",
                        "description": "Optional: only include chunks starting at or before this page.",
                    },
                },
                "required": ["field", "contains"],
            },
        )

    def execute(
        self,
        field: str,
        contains: str,
        page_from: int | None = None,
        page_to: int | None = None,
    ) -> str:
        if field not in ("chapter", "section"):
            return f"Invalid field '{field}'. Must be 'chapter' or 'section'."

        raw = self._store._collection.get(include=["documents", "metadatas"])
        if not raw["ids"]:
            return "No book content indexed yet."

        needle = contains.lower()
        matched: list[tuple[int, str, dict]] = []  # (page_start, doc, meta)

        for doc, meta in zip(raw["documents"], raw["metadatas"]):
            field_value = (meta.get(field) or "").lower()
            if needle not in field_value:
                continue

            page_start = meta.get("page_start", 0)
            if page_from is not None and page_start < page_from:
                continue
            if page_to is not None and page_start > page_to:
                continue

            matched.append((page_start, doc, meta))

        if not matched:
            return (
                f"No chunks found where {field} contains '{contains}'. "
                "Try calling book_outline to see exact chapter/part names."
            )

        matched.sort(key=lambda t: t[0])

        truncated = len(matched) > _MAX_CHUNKS
        chunks_to_show = matched[:_MAX_CHUNKS]

        parts: list[str] = []
        if truncated:
            parts.append(
                f"[Note: {len(matched)} chunks matched. Showing first {_MAX_CHUNKS} "
                f"(pages {chunks_to_show[0][0]}–{chunks_to_show[-1][0]}) to fit context. "
                "Summarize from these excerpts; mention that the summary covers the opening "
                "portion of this section if relevant.]"
            )

        for page_start, doc, meta in chunks_to_show:
            section = meta.get("section") or meta.get("chapter") or ""
            loc = f"{section}, p.{page_start}" if section else f"p.{page_start}"
            parts.append(f"[{loc}]\n{doc}")

        logger.debug(
            "get_book_section: field=%s contains=%r → %d/%d chunks returned",
            field, contains, len(chunks_to_show), len(matched),
        )
        return "\n\n".join(parts)
