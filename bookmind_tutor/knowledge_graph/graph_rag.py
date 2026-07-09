"""
GraphRAGRetriever: combines vector search with knowledge graph expansion.

Why two sources?
- Vector search: good at finding chunks that paraphrase the query or use
  synonymous language. Works even when the user doesn't know exact entity names.
- Graph traversal: good at finding chunks about entities *related* to what the
  user asked about — things the user may not have thought to search for directly.

Approach (query expansion, not replacement):
  1. Run vector search with the original query → top-k results.
  2. Find entity names from the query that exist in the book graph → traverse 1
     hop → collect related entity names as book-side expansion terms.
  3. Find entity names from the query that exist in the user's personal KG →
     add directly as user-side expansion terms (no traversal — user KG has no
     edges yet, but the names themselves anchor retrieval to familiar vocabulary).
  4. If any expansion terms found, append them to the query and run a second
     vector search.
  5. Merge both result sets, deduplicate by chunk_id, re-rank by score.

User KG personalization (step 3) means: when a user already knows "architecture
pattern" and asks about "event sourcing", the expanded query pulls in chunks that
discuss event sourcing *in the context of* architecture patterns — i.e. content
that bridges what the user knows to what they're learning.

Cold start: empty user KG → step 3 adds nothing → behaviour identical to before.
"""
from __future__ import annotations

import logging

from bookmind_tutor.knowledge_graph.graph_store import GraphStore
from bookmind_tutor.retrieval.models import SearchResult
from bookmind_tutor.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class GraphRAGRetriever:
    """
    Retriever that blends semantic vector search with KG-guided query expansion.

    Args:
        vector_store: VectorStore (ChromaDB) for chunk retrieval.
        graph_store: GraphStore (Neo4j) for entity traversal.
        k: Number of results to return after merging.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        k: int = 5,
    ) -> None:
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._k = k

    def search(self, query: str) -> list[SearchResult]:
        """
        Search for relevant chunks using vector search + two expansion sources:
        the book graph (entity traversal) and the user's personal KG (vocabulary
        anchoring).

        Falls back to plain vector search when both expansion sources are empty.
        """
        query_lower = query.lower()

        # Step 1: vector search with original query
        vector_results = self._vector_store.search(query, k=self._k)

        # Step 2: book-graph expansion — traverse 1 hop from matched book entities
        book_entity_names = self._match_book_entities(query_lower)
        book_expansion: set[str] = set()
        for name in book_entity_names:
            related = self._graph_store.get_related_entities(name, depth=1)
            book_expansion.update(e.name for e in related)
        book_expansion = {n for n in book_expansion if n not in query_lower}

        # Step 3: user-KG expansion — entity names the user already knows that
        # appear in the query are added directly as expansion terms.
        # No traversal needed: the names themselves anchor the second vector
        # search to content that bridges the user's existing knowledge to the
        # query topic.
        user_expansion = set(self._match_user_entities(query_lower))

        expansion_terms = book_expansion | user_expansion
        if not expansion_terms:
            logger.debug("No expansion terms from book or user KG — returning vector results only")
            return vector_results

        logger.debug(
            "Expansion: %d book terms, %d user-KG terms",
            len(book_expansion), len(user_expansion),
        )

        # Step 4: second vector search with expanded query
        expanded_query = query + " " + " ".join(expansion_terms)
        expanded_results = self._vector_store.search(expanded_query, k=self._k)

        # Step 5: merge and deduplicate — keep highest score per chunk
        merged: dict[str, SearchResult] = {}
        for result in vector_results + expanded_results:
            if result.chunk_id not in merged or result.score > merged[result.chunk_id].score:
                merged[result.chunk_id] = result

        final = sorted(merged.values(), key=lambda r: r.score, reverse=True)[: self._k]
        logger.debug(
            "GraphRAG: %d vector + %d expanded → %d merged",
            len(vector_results), len(expanded_results), len(final),
        )
        return final

    def _match_book_entities(self, query_lower: str) -> list[str]:
        """
        Return book-graph entity names that appear as substrings in the query.

        Sorted longest-first so "eventual consistency" matches before "consistency".
        """
        all_names = self._graph_store.get_all_entity_names()
        return [
            name for name in sorted(all_names, key=len, reverse=True)
            if name in query_lower
        ]

    def _match_user_entities(self, query_lower: str) -> list[str]:
        """
        Return user-KG entity names that appear as substrings in the query.

        These are concepts the user has already encountered and approved. Adding
        them to the expansion query pulls in chunks that discuss the query topic
        *using vocabulary familiar to the user*, making the retrieved content
        easier to connect to their existing knowledge.
        """
        user_names = self._graph_store.get_user_entity_names()
        return [name for name in user_names if name in query_lower]
