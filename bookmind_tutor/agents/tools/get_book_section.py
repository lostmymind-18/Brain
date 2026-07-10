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
                "Retrieve ALL content from a specific chapter or part of the book. "
                "Use when the user asks to summarize or overview an entire chapter or part.\n\n"
                "ALWAYS provide BOTH required parameters: `field` AND `contains`.\n"
                "  - To get a chapter: field=\"section\", contains=\"Chapter 1\"\n"
                "  - To get a part:    field=\"chapter\", contains=\"Part II\"\n\n"
                "The `field` parameter selects which metadata field to search:\n"
                "  - \"section\": chapter-level headings (e.g. 'Chapter 1. Introduction')\n"
                "  - \"chapter\": part-level headings (e.g. 'Part I. Foundations')\n\n"
                "The `contains` parameter is the search string (case-insensitive substring). "
                "Call book_outline first to find exact names. "
                "WARNING: `contains` must not be empty — always provide a search term."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": ["chapter", "section"],
                        "description": (
                            "Metadata field to filter on: "
                            "'section' for chapter-level headings, "
                            "'chapter' for part-level headings."
                        ),
                    },
                    "contains": {
                        "type": "string",
                        "description": (
                            "Non-empty substring to match (case-insensitive). "
                            "Examples: 'Chapter 1', 'Part II', 'Pipeline Architecture'."
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
        field: str = "",
        contains: str = "",
        page_from: int | None = None,
        page_to: int | None = None,
    ) -> str:
        if field not in ("chapter", "section"):
            return (
                f"Invalid field '{field}'. Must be 'chapter' or 'section'. "
                "Example: field=\"section\", contains=\"Chapter 1\""
            )
        if not contains or not contains.strip():
            return (
                "The 'contains' parameter must not be empty. "
                "Provide a chapter or part name to match. "
                "Example: field=\"section\", contains=\"Chapter 1\""
            )

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
            # Count distinct section names to detect substring ambiguity.
            distinct = {
                (meta.get(field) or "").strip()
                for _, _, meta in matched
                if (meta.get(field) or "").strip()
            }
            ambiguity_hint = ""
            if len(distinct) > 1 and not contains.endswith("."):
                ambiguity_hint = (
                    f" WARNING: '{contains}' matched {len(distinct)} different sections "
                    f"(substring is too broad, e.g. 'Chapter 1' also matches Chapter 10, 11...). "
                    f"Try adding a period: contains='{contains}.' to match exactly one section."
                )
            parts.append(
                f"[Note: {len(matched)} chunks across {len(distinct)} section(s) matched. "
                f"Showing first {_MAX_CHUNKS} (pages {chunks_to_show[0][0]}-{chunks_to_show[-1][0]})."
                + ambiguity_hint
                + " DO NOT call this tool again with the same parameters - "
                "calling again returns the same result. Fix the 'contains' value if needed, "
                "or summarize from the excerpts provided here.]"
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
