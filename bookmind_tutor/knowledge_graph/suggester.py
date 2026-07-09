"""
KGUpdateSuggester: suggests knowledge graph updates after each conversation turn.

After the agent answers a question, the suggester inspects the retrieved chunks
and extracts entities the user hasn't encountered yet. These are presented to
the user as suggestions ("Would you like to add these to your knowledge graph?").

This is what makes the KG a personal "brain" rather than a static book index:
it grows lazily, one conversation turn at a time, driven by what the user
actually explores.

Cost note: EntityExtractor makes 1 LLM call per chunk. To avoid re-extracting
the same chunk across turns, results are cached by chunk_id in memory. In a
Streamlit session this cache lives in session_state so it persists across reruns.
"""
from __future__ import annotations

import logging

from bookmind_tutor.ingestion.models import Chunk
from bookmind_tutor.knowledge_graph.extractor import EntityExtractor
from bookmind_tutor.knowledge_graph.graph_store import GraphStore
from bookmind_tutor.knowledge_graph.models import (
    Entity,
    KGUpdateResult,
    KGUpdateSuggestion,
    Relation,
)
from bookmind_tutor.retrieval.models import SearchResult

logger = logging.getLogger(__name__)

# Cap suggestions per turn to avoid overwhelming the user.
_MAX_SUGGESTIONS = 5


class KGUpdateSuggester:
    """
    Proposes new entities for the user's knowledge graph after each turn.

    Args:
        extractor: EntityExtractor instance for LLM-based entity extraction.
        max_suggestions: Maximum number of suggestions returned per turn.
    """

    def __init__(
        self,
        extractor: EntityExtractor,
        max_suggestions: int = _MAX_SUGGESTIONS,
    ) -> None:
        self._extractor = extractor
        self._max_suggestions = max_suggestions
        # chunk_id → (entities, relations), persists across turns to avoid re-extraction
        self._cache: dict[str, tuple[list[Entity], list[Relation]]] = {}

    def suggest(
        self,
        results: list[SearchResult],
        user_graph: GraphStore,
    ) -> KGUpdateResult:
        """
        Return entity suggestions from retrieved search results.

        Entities already in the user's KG are excluded. Results are capped at
        max_suggestions to keep the UI manageable.

        Args:
            results: Search results from the current conversation turn.
            user_graph: The user's personal knowledge graph (for deduplication).
        """
        known_names = set(user_graph.get_user_entity_names())
        already_known: list[str] = []
        candidates: list[KGUpdateSuggestion] = []

        for result in results:
            entities, relations = self._extract_cached(result)

            # Persist to book KG so relations accumulate as user explores.
            # This is the lazy build: we only extract chunks the user actually
            # retrieves, so the book KG grows in step with their reading.
            for entity in entities:
                user_graph.upsert_entity(entity)
            for relation in relations:
                user_graph.upsert_relation(relation)

            for entity in entities:
                if entity.name in known_names:
                    already_known.append(entity.name)
                else:
                    candidates.append(
                        KGUpdateSuggestion(
                            entity=entity,
                            reason=f"appeared in retrieved content from {result.chapter or 'the book'}, p.{result.page_range[0]}",
                            source_chunk_id=result.chunk_id,
                        )
                    )
                    # Track locally so we don't suggest the same entity twice
                    # from different chunks in the same turn.
                    known_names.add(entity.name)

        # Deduplicate by entity name (keep first occurrence) and cap.
        suggestions = candidates[: self._max_suggestions]

        logger.debug(
            "KGUpdateSuggester: %d suggestions, %d already known",
            len(suggestions),
            len(already_known),
        )
        return KGUpdateResult(suggestions=suggestions, already_known=already_known)

    def _extract_cached(self, result: SearchResult) -> tuple[list[Entity], list[Relation]]:
        """Extract entities and relations from a SearchResult, caching by chunk_id."""
        if result.chunk_id in self._cache:
            return self._cache[result.chunk_id]

        # EntityExtractor expects a Chunk; we build a minimal one from SearchResult.
        # Only chunk_id and text are used internally by EntityExtractor.
        chunk = Chunk(
            chunk_id=result.chunk_id,
            text=result.text,
            chapter=result.chapter,
            section=result.section,
            page_range=result.page_range,
            char_offset_start=0,
            char_offset_end=len(result.text),
        )
        entities, relations = self._extractor.extract(chunk)
        self._cache[result.chunk_id] = (entities, relations)
        logger.debug(
            "Extracted %d entities, %d relations from chunk %s (cached)",
            len(entities), len(relations), result.chunk_id[:8],
        )
        return entities, relations
