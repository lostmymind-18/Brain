"""BookSearchTool: semantic search over an indexed book."""
from __future__ import annotations

from bookmind_tutor.agents.models import ToolSpec
from bookmind_tutor.agents.tools.base import Tool
from bookmind_tutor.retrieval.vector_store import VectorStore


class BookSearchTool(Tool):
    """
    Search the indexed book for passages relevant to a query.

    Returns up to k results formatted as numbered, annotated excerpts.
    Each result includes the chapter, section, and page number so Claude
    can cite sources in its answers.
    """

    def __init__(self, vector_store: VectorStore, k: int = 5) -> None:
        self._store = vector_store
        self._k = k

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="book_search",
            description=(
                "Search the book for passages relevant to a question or topic. "
                "Use this whenever you need to find information from the book. "
                "Returns up to 5 relevant excerpts with source location."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to search for.",
                    }
                },
                "required": ["query"],
            },
        )

    def execute(self, query: str) -> str:
        results = self._store.search(query, k=self._k)
        if not results:
            return "No relevant passages found in the book."

        parts: list[str] = []
        for i, r in enumerate(results, 1):
            chapter = r.chapter or "unknown chapter"
            section = r.section
            loc = f"{chapter}" + (f" / {section}" if section else "")
            loc += f", p.{r.page_range[0]}"
            parts.append(f"[{i}] ({loc})\n{r.text}")

        return "\n\n".join(parts)
