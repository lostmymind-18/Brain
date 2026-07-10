"""GetBookSectionTool: fetch ALL chunks belonging to a named part or chapter.

Complements book_search (top-k semantic) with structured retrieval:
given a part/chapter name, return every chunk that belongs to that
scope, not just the top-k most similar ones.

Design note (why a single `section_name` parameter):
An earlier version asked the LLM to pick a metadata field
("chapter" = part level, "section" = chapter level) plus a substring.
That leaked the internal metadata schema to the model, and the correct
field choice depends on book structure the model cannot see. Real
failure observed: in "Fundamentals of Software Architecture",
Chapter 1 sits BEFORE Part I, so the structure detector stored it at
part level (in the `chapter` field). The model asked for
field='section', contains='Chapter 1', which substring-matched
Chapters 10-19 (73k chars of wrong content) and never the real
Chapter 1. The fix: the tool matches the requested name against
DISTINCT heading values from BOTH fields and disambiguates itself.

Matching behavior:
- Exactly one distinct heading matches -> return all its chunks.
- Multiple headings match -> return a short disambiguation list
  (name, chunk count, page range) so the model can call again with
  the exact name. A few hundred tokens instead of 73k chars of
  possibly-wrong content.
- Zero headings match -> return the list of available headings.
"""
from __future__ import annotations

import logging
from collections import Counter

from bookmind_tutor.agents.models import ToolSpec
from bookmind_tutor.agents.tools.base import SourceRef, Tool
from bookmind_tutor.ingestion.utils import normalize_heading as _normalize
from bookmind_tutor.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Summarizing a full part (100+ chunks) can easily exceed 50k tokens.
# Cap chunks returned so the LLM context stays manageable.
# The tool notes truncation in its output so the LLM can tell the user.
_MAX_CHUNKS = 30

# Metadata fields that hold heading names. Searched in order of specificity.
# `chapter` = part-level, `section` = chapter-level, `subsection` = L2 section.
# Front matter and out-of-part chapters can land in any field, so all three
# are searched.
_HEADING_FIELDS = ("chapter", "section", "subsection")


class GetBookSectionTool(Tool):
    """
    Return all book chunks that belong to a named part, chapter, or page range.

    Unlike book_search (which returns top-k by semantic similarity), this tool
    uses metadata filtering to guarantee complete coverage of the requested scope.
    Ideal for summarization tasks where missing chunks would produce an incomplete answer.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store
        self.last_sources: list[SourceRef] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="get_book_section",
            description=(
                "Retrieve ALL content of a specific part or chapter of the book. "
                "Use when the user asks to summarize, overview, or explain an entire "
                "chapter or part (instead of book_search, which returns only the "
                "top-k excerpts).\n\n"
                "Pass the part/chapter name in `section_name`. Matching is a "
                "case-insensitive substring match against the book's heading names. "
                "If the name matches several headings (e.g. 'Chapter 1' also matches "
                "'Chapter 14'), the tool returns the list of matching headings so you "
                "can call again with the exact one. Call book_outline first if you "
                "want to see the available names up front."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "section_name": {
                        "type": "string",
                        "description": (
                            "Name of the part or chapter to retrieve, e.g. "
                            "'Chapter 1. Introduction' or 'Part II'. "
                            "Must not be empty."
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
                "required": ["section_name"],
            },
        )

    def execute(
        self,
        section_name: str = "",
        page_from: int | None = None,
        page_to: int | None = None,
    ) -> str:
        if not section_name or not section_name.strip():
            return (
                "The 'section_name' parameter must not be empty. "
                "Provide a part or chapter name, e.g. section_name=\"Chapter 1. Introduction\". "
                "Call book_outline to see the available names."
            )

        raw = self._store._collection.get(include=["documents", "metadatas"])
        if not raw["ids"]:
            return "No book content indexed yet."

        # Group chunks by distinct heading value across both metadata fields.
        # A chunk belongs to the heading in its `chapter` field AND the one in
        # its `section` field, so it appears under both groups. Dedup by chunk
        # id at output time in case both match the query.
        headings: dict[str, dict] = {}  # normalized name -> {variants, chunks}
        for chunk_id, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"]):
            for field in _HEADING_FIELDS:
                value = (meta.get(field) or "").strip()
                if not value:
                    continue
                norm = _normalize(value)
                if not norm:
                    continue
                entry = headings.setdefault(norm, {"variants": Counter(), "chunks": []})
                entry["variants"][value] += 1
                entry["chunks"].append((meta.get("page_start", 0), chunk_id, doc, meta))
        # Display name per heading: the most frequent original spelling. Body
        # headings cover many chunks; TOC/appendix variants of the same name
        # cover one, so the real heading wins.
        for entry in headings.values():
            entry["display"] = entry["variants"].most_common(1)[0][0]

        needle = _normalize(section_name)
        matched_names = [norm for norm in headings if needle in norm]
        # An exact name always wins over substring hits, so calling again with
        # a name copied from the disambiguation list cannot re-trigger it.
        if needle in headings:
            matched_names = [needle]

        if not matched_names:
            available = self._format_heading_list(headings)
            return (
                f"No part or chapter named '{section_name}' found. "
                f"Available headings:\n{available}\n"
                "Call again with one of these exact names."
            )

        if len(matched_names) > 1:
            listing = self._format_heading_list(headings, only=matched_names)
            return (
                f"'{section_name}' matches {len(matched_names)} different headings. "
                f"Call again with the exact name of the one you want:\n{listing}"
            )

        entry = headings[matched_names[0]]
        seen_ids: set[str] = set()
        matched: list[tuple[int, str, dict]] = []  # (page_start, doc, meta)
        for page_start, chunk_id, doc, meta in entry["chunks"]:
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            if page_from is not None and page_start < page_from:
                continue
            if page_to is not None and page_start > page_to:
                continue
            matched.append((page_start, doc, meta))

        if not matched:
            return (
                f"'{entry['display']}' exists but has no chunks in the requested "
                f"page range ({page_from}-{page_to})."
            )

        matched.sort(key=lambda t: t[0])

        # One SourceRef covering the whole heading is more readable in the
        # Sources panel than one line per chunk.
        first_page = matched[0][0]
        last_page = max(m.get("page_end", p) for p, _, m in matched)
        self.last_sources.append(
            SourceRef(
                chapter=entry["display"],
                section=None,
                page_range=(first_page, last_page),
            )
        )

        truncated = len(matched) > _MAX_CHUNKS
        chunks_to_show = matched[:_MAX_CHUNKS]

        parts: list[str] = []
        if truncated:
            parts.append(
                f"[Note: '{entry['display']}' has {len(matched)} chunks. Showing first "
                f"{_MAX_CHUNKS} (pages {chunks_to_show[0][0]}-{chunks_to_show[-1][0]}). "
                "DO NOT call this tool again with the same parameters - calling again "
                "returns the same result. Use page_from/page_to to fetch a later page "
                "range, or summarize from the excerpts provided here.]"
            )

        for page_start, doc, meta in chunks_to_show:
            heading = (
                meta.get("subsection")
                or meta.get("section")
                or meta.get("chapter")
                or ""
            )
            loc = f"{heading}, p.{page_start}" if heading else f"p.{page_start}"
            parts.append(f"[{loc}]\n{doc}")

        logger.debug(
            "get_book_section: name=%r -> heading=%r, %d/%d chunks returned",
            section_name, entry["display"], len(chunks_to_show), len(matched),
        )
        return "\n\n".join(parts)

    @staticmethod
    def _format_heading_list(
        headings: dict[str, dict], only: list[str] | None = None
    ) -> str:
        """One line per heading: name, chunk count, page range. Sorted by first page."""
        names = only if only is not None else list(headings)
        rows = []
        for norm in names:
            entry = headings[norm]
            pages = [p for p, _, _, _ in entry["chunks"]]
            rows.append((min(pages), entry["display"], len(entry["chunks"]), max(pages)))
        rows.sort()
        return "\n".join(
            f"- {display} ({count} chunks, pages {first}-{last})"
            for first, display, count, last in rows
        )
