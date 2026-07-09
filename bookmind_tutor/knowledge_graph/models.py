"""Data models for the knowledge graph module."""
from __future__ import annotations

from dataclasses import dataclass, field

# Valid entity types — fixed set to reduce LLM hallucination and simplify validation.
# Open-ended types were considered but ruled out: harder to validate, noisier output.
ENTITY_TYPES = frozenset({"CONCEPT", "SYSTEM", "PRINCIPLE", "DEFINITION", "PERSON"})

# Valid relation types between entities.
RELATION_TYPES = frozenset({"IS_A", "RELATED_TO", "CONTRADICTS", "PART_OF", "DEFINED_IN"})


@dataclass
class Entity:
    """
    A technical entity extracted from a book chunk.

    name is normalized to lowercase so "Raft", "raft", "RAFT" all merge into
    one node in Neo4j (the unique constraint is on name).
    """
    name: str        # lowercase, e.g. "raft", "eventual consistency"
    type: str        # one of ENTITY_TYPES
    description: str # 1-sentence description from the source text
    chunk_id: str    # chunk where this entity was first extracted

    def __post_init__(self) -> None:
        self.name = self.name.strip().lower()

    def is_valid(self) -> bool:
        return bool(self.name) and self.type in ENTITY_TYPES


@dataclass
class Relation:
    """
    A directed relationship between two entities.

    source and target are entity names (lowercase).
    description is optional context, most useful for RELATED_TO.
    """
    source: str        # entity name
    relation_type: str # one of RELATION_TYPES
    target: str        # entity name
    description: str = ""

    def __post_init__(self) -> None:
        self.source = self.source.strip().lower()
        self.target = self.target.strip().lower()

    def is_valid(self) -> bool:
        return (
            bool(self.source)
            and bool(self.target)
            and self.source != self.target
            and self.relation_type in RELATION_TYPES
        )


@dataclass
class KGUpdateSuggestion:
    """
    A single entity suggested to be added to the user's knowledge graph.

    Produced by KGUpdateSuggester after each conversation turn.
    The user decides whether to accept or ignore each suggestion.
    """
    entity: Entity
    reason: str       # human-readable explanation, shown in the UI
    source_chunk_id: str


@dataclass
class KGUpdateResult:
    """Output of KGUpdateSuggester.suggest()."""
    suggestions: list[KGUpdateSuggestion]
    already_known: list[str]  # entity names already in user KG (for debugging)


@dataclass
class BuildStats:
    """Summary of a KnowledgeGraphBuilder.build_from_chunks() run."""
    total_chunks: int
    chunks_processed: int = 0
    chunks_failed: int = 0
    total_entities: int = 0
    total_relations: int = 0
    elapsed_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.chunks_processed / self.total_chunks
