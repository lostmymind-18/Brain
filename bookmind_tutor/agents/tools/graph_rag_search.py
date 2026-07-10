"""GraphRAGSearchTool: semantic search enhanced with knowledge graph expansion."""
from __future__ import annotations

from bookmind_tutor.agents.models import ToolSpec
from bookmind_tutor.agents.tools.base import SourceRef, Tool
from bookmind_tutor.knowledge_graph.graph_rag import GraphRAGRetriever


class GraphRAGSearchTool(Tool):
    """
    Search the indexed book using GraphRAG — vector search enriched by KG traversal.

    Compared to BookSearchTool (plain vector search), this tool also expands
    the query with entities related to those found in the question, surfacing
    chunks that vector search alone would miss.

    Use this tool instead of BookSearchTool when a knowledge graph has been built
    for the book. Falls back gracefully to pure vector results when the graph is
    empty or no query entities match.
    """

    def __init__(self, retriever: GraphRAGRetriever, k: int = 5) -> None:
        self._retriever = retriever
        self._k = k
        self.last_sources: list[SourceRef] = []

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

    _LOW_RELEVANCE_THRESHOLD = 0.25

    def execute(self, query: str) -> str:
        results = self._retriever.search(query)
        if not results:
            return "No relevant passages found in the book."

        self.last_sources.extend(
            SourceRef(chapter=r.chapter, section=r.section, page_range=r.page_range)
            for r in results
        )

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
            loc += f", p.{r.page_range[0]}"
            parts.append(f"[{i}] ({loc})\n{r.text}")

        return "\n\n".join(parts)
