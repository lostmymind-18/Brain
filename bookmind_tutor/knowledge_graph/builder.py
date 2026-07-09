"""
KnowledgeGraphBuilder: orchestrates chunk → entity extraction → Neo4j indexing.

Processes chunks sequentially (not in parallel) to avoid overwhelming the LLM
rate limit. Each chunk failure is logged and skipped — one bad chunk does not
abort the rest of the build.
"""
from __future__ import annotations

import logging
import time

from bookmind_tutor.ingestion.models import Chunk
from bookmind_tutor.knowledge_graph.extractor import EntityExtractor
from bookmind_tutor.knowledge_graph.graph_store import GraphStore
from bookmind_tutor.knowledge_graph.models import BuildStats

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Extracts entities from chunks and persists them to the graph store.

    Args:
        graph_store: GraphStore instance (Neo4j connection).
        extractor: EntityExtractor instance (LLM-based extraction).
    """

    def __init__(self, graph_store: GraphStore, extractor: EntityExtractor) -> None:
        self._store = graph_store
        self._extractor = extractor

    def build_from_chunks(
        self,
        chunks: list[Chunk],
        max_chunks: int | None = None,
    ) -> BuildStats:
        """
        Extract entities and relations from chunks and upsert into Neo4j.

        Args:
            chunks: List of Chunk objects from the ingestion pipeline.
            max_chunks: If set, only process the first N chunks. Useful for
                        testing extraction quality before running the full book.

        Returns:
            BuildStats with counts of processed/failed chunks and entities.
        """
        target = chunks[:max_chunks] if max_chunks else chunks
        stats = BuildStats(total_chunks=len(target))
        t0 = time.monotonic()

        for i, chunk in enumerate(target, 1):
            logger.info(
                "Processing chunk %d/%d: %s p.%d",
                i, len(target), chunk.chapter or "?", chunk.page_range[0],
            )
            try:
                entities, relations = self._extractor.extract(chunk)

                for entity in entities:
                    self._store.upsert_entity(entity)

                for relation in relations:
                    self._store.upsert_relation(relation)

                stats.chunks_processed += 1
                stats.total_entities += len(entities)
                stats.total_relations += len(relations)

            except Exception as exc:
                logger.warning("Failed chunk %s: %s", chunk.chunk_id, exc)
                stats.chunks_failed += 1

        stats.elapsed_seconds = time.monotonic() - t0
        logger.info(
            "Build complete: %d/%d chunks ok, %d entities, %d relations, %.1fs",
            stats.chunks_processed, stats.total_chunks,
            stats.total_entities, stats.total_relations, stats.elapsed_seconds,
        )
        return stats
