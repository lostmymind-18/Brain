"""Knowledge graph module — entity extraction, Neo4j storage, and Graph RAG."""
from bookmind_tutor.knowledge_graph.builder import KnowledgeGraphBuilder
from bookmind_tutor.knowledge_graph.extractor import EntityExtractor
from bookmind_tutor.knowledge_graph.graph_rag import GraphRAGRetriever
from bookmind_tutor.knowledge_graph.graph_store import GraphStore
from bookmind_tutor.knowledge_graph.models import (
    ENTITY_TYPES,
    RELATION_TYPES,
    BuildStats,
    Entity,
    KGUpdateResult,
    KGUpdateSuggestion,
    Relation,
)
from bookmind_tutor.knowledge_graph.suggester import KGUpdateSuggester

__all__ = [
    "BuildStats",
    "ENTITY_TYPES",
    "EntityExtractor",
    "GraphRAGRetriever",
    "GraphStore",
    "KGUpdateResult",
    "KGUpdateSuggester",
    "KGUpdateSuggestion",
    "KnowledgeGraphBuilder",
    "RELATION_TYPES",
    "Entity",
    "Relation",
]
