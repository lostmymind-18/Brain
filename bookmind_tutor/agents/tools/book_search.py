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
                "Returns up to 5 relevant excerpts with source location. "
                "IMPORTANT: Always write the query in English, regardless of the "
                "language the user is speaking. The book content is in English and "
                "the search index only matches English text."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to search for, written in English.",
                    }
                },
                "required": ["query"],
            },
        )

    # Cosine similarity below this threshold means the top result is a poor match.
    # At 0.25, irrelevant queries (e.g. "table of contents" vs English chunks) score
    # ~0.22 while typical on-topic queries score 0.35+.
    _LOW_RELEVANCE_THRESHOLD = 0.25

    def execute(self, query: str) -> str:
        results = self._store.search(query, k=self._k)
        if not results:
            return "No relevant passages found in the book."

        best_score = results[0].score
        parts: list[str] = []

        if best_score < self._LOW_RELEVANCE_THRESHOLD:
            parts.append(
                f"[Warning: low relevance — best match score {best_score:.2f}. "
                "The book may not contain specific information about this query. "
                "Do not invent details not present in the excerpts below.]"
            )

        for i, r in enumerate(results, 1):
            chapter = r.chapter or "unknown chapter"
            loc = chapter + (f" / {r.section}" if r.section else "")
            if r.subsection:
                loc += f" / {r.subsection}"
            loc += f", p.{r.page_range[0]}"
            parts.append(f"[{i}] ({loc})\n{r.text}")

        return "\n\n".join(parts)
