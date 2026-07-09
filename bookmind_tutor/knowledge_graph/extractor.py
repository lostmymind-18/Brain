"""
EntityExtractor: extract entities and relations from book chunks via LLM.

Uses fixed entity types (CONCEPT, SYSTEM, PRINCIPLE, DEFINITION, PERSON) and
fixed relation types (IS_A, RELATED_TO, CONTRADICTS, PART_OF, DEFINED_IN).

Fixed types were chosen over open-ended extraction because:
- Technical books like DDIA have predictable entity categories.
- Open-ended types require a second LLM pass to normalize, adding latency and cost.
- Validation is straightforward: any entity/relation not in the fixed set is dropped.

Fallback: if JSON parsing fails for any chunk, ([], []) is returned so the
builder can continue processing other chunks without crashing.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from bookmind_tutor.ingestion.models import Chunk
from bookmind_tutor.llm.base import LLMClient
from bookmind_tutor.knowledge_graph.models import (
    ENTITY_TYPES,
    RELATION_TYPES,
    Entity,
    Relation,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = f"""\
You are an entity extraction assistant for technical books.
Extract technical entities and their relationships from the given excerpt.

Entity types — use EXACTLY these strings:
- CONCEPT: abstract technical idea (e.g. "eventual consistency", "replication lag")
- SYSTEM: concrete software, algorithm, or protocol (e.g. "kafka", "raft", "b-tree")
- PRINCIPLE: fundamental rule or theorem (e.g. "cap theorem", "linearizability")
- DEFINITION: a term being formally defined in the text
- PERSON: researcher or author mentioned by name

Relation types — use EXACTLY these strings:
- IS_A: source is a type or instance of target
- RELATED_TO: source is meaningfully related to target (explain in description)
- CONTRADICTS: source contradicts or trades off against target
- PART_OF: source is a component or sub-concept of target
- DEFINED_IN: source entity is defined in the given text (target = chunk location, skip this type)

Rules:
- All entity names must be LOWERCASE.
- Only extract entities clearly present in the excerpt — do not hallucinate.
- Only create relations between entities you have already extracted.
- If you are unsure about an entity's type, omit it.
- Return ONLY raw JSON — no markdown fences, no explanation.

Output format:
{{
  "entities": [
    {{"name": "<lowercase name>", "type": "<TYPE>", "description": "<1 sentence from text>"}}
  ],
  "relations": [
    {{"source": "<name>", "type": "<TYPE>", "target": "<name>", "description": ""}}
  ]
}}
"""

_USER_PROMPT = "Extract entities and relations from this excerpt:\n\n{text}"


class EntityExtractor:
    """
    Extracts Entity and Relation objects from a single Chunk using an LLM.

    Args:
        llm_client: LLMClient instance.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "",  # ignored — model is part of llm_client
    ) -> None:
        self._client = llm_client

    def extract(self, chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
        """
        Extract entities and relations from one chunk.

        Returns ([], []) on any failure so callers can continue without crashing.
        All returned Entity and Relation objects pass their is_valid() check.
        """
        try:
            raw = self._call_llm(chunk.text)
            return self._parse(raw, chunk.chunk_id)
        except Exception as exc:
            logger.warning(
                "Extraction failed for chunk %s: %s", chunk.chunk_id, exc
            )
            return [], []

    def _call_llm(self, text: str) -> str:
        response = self._client.complete(
            messages=[{"role": "user", "content": _USER_PROMPT.format(text=text[:3000])}],
            system=_SYSTEM_PROMPT,
            max_tokens=2048,
        )
        return response.text

    def _parse(
        self,
        raw: str,
        chunk_id: str,
    ) -> tuple[list[Entity], list[Relation]]:
        """
        Parse LLM JSON output into validated Entity and Relation objects.

        Strips markdown fences if the LLM wraps the JSON despite instructions.
        Drops any entity or relation that fails is_valid() rather than crashing.
        """
        # Strip markdown fences if present
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object found in LLM response: {raw[:100]!r}")

        data = json.loads(raw[start : end + 1])

        # --- entities ---
        entities: list[Entity] = []
        extracted_names: set[str] = set()
        for item in data.get("entities", []):
            try:
                entity = Entity(
                    name=str(item.get("name", "")).strip().lower(),
                    type=str(item.get("type", "")),
                    description=str(item.get("description", "")),
                    chunk_id=chunk_id,
                )
            except Exception:
                continue

            if entity.is_valid():
                entities.append(entity)
                extracted_names.add(entity.name)
            else:
                logger.debug("Dropped invalid entity: %s (type=%s)", entity.name, entity.type)

        # --- relations ---
        relations: list[Relation] = []
        for item in data.get("relations", []):
            try:
                relation = Relation(
                    source=str(item.get("source", "")).strip().lower(),
                    relation_type=str(item.get("type", "")),
                    target=str(item.get("target", "")).strip().lower(),
                    description=str(item.get("description", "")),
                )
            except Exception:
                continue

            # Only keep relations between entities we actually extracted
            if relation.is_valid() and relation.source in extracted_names and relation.target in extracted_names:
                relations.append(relation)
            else:
                logger.debug(
                    "Dropped relation: %s -[%s]-> %s",
                    relation.source, relation.relation_type, relation.target,
                )

        logger.debug(
            "Chunk %s: extracted %d entities, %d relations",
            chunk_id, len(entities), len(relations),
        )
        return entities, relations
