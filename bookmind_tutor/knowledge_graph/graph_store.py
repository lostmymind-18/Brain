"""
GraphStore: Neo4j CRUD operations for the BookMind knowledge graph.

All writes use MERGE (not CREATE) so re-running extraction on the same chunks
is idempotent — no duplicate nodes or edges accumulate.

Design note: dynamic relationship types are not supported in Cypher without APOC,
so each relation type has its own query template. This is verbose but explicit
and avoids a runtime dependency on APOC plugins.
"""
from __future__ import annotations

import logging

from neo4j import GraphDatabase

from bookmind_tutor.knowledge_graph.models import Entity, Relation

logger = logging.getLogger(__name__)

# One query template per relation type.
# Two separate MATCH clauses (not comma-separated) so Neo4j's query planner
# treats each as an independent index lookup on Entity.name rather than a
# Cartesian product join. With the unique constraint on name, the comma form
# works correctly but triggers a performance warning on every call.
_RELATION_QUERIES: dict[str, str] = {
    t: f"""
        MATCH (s:Entity {{name: $source}})
        MATCH (t:Entity {{name: $target}})
        MERGE (s)-[r:{t}]->(t)
        SET r.description = $description
    """
    for t in ("IS_A", "RELATED_TO", "CONTRADICTS", "PART_OF", "DEFINED_IN")
}

# Same pattern for UserEntity — separate label keeps user knowledge distinct
# from the book index so queries can target either independently.
_USER_RELATION_QUERIES: dict[str, str] = {
    t: f"""
        MATCH (s:UserEntity {{name: $source}})
        MATCH (t:UserEntity {{name: $target}})
        MERGE (s)-[r:{t}]->(t)
        SET r.description = $description
    """
    for t in ("IS_A", "RELATED_TO", "CONTRADICTS", "PART_OF", "DEFINED_IN")
}


class GraphStore:
    """
    Thin wrapper around the Neo4j driver for BookMind entity/relation storage.

    Args:
        uri: Bolt URI, e.g. "bolt://localhost:7687".
        user: Neo4j username.
        password: Neo4j password.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "bookmind123",
    ) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._entity_name_cache: list[str] | None = None
        self._user_entity_name_cache: list[str] | None = None
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        """Create uniqueness constraints on Entity.name and UserEntity.name (idempotent)."""
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT user_entity_name IF NOT EXISTS "
                "FOR (e:UserEntity) REQUIRE e.name IS UNIQUE"
            )

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def upsert_entity(self, entity: Entity) -> None:
        """
        Insert or update an Entity node.

        If a node with the same name already exists, its properties are updated.
        This means the first chunk to mention an entity "wins" on description,
        unless a later chunk provides a better one — for now last-write-wins.
        """
        with self._driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                SET e.type        = $type,
                    e.description = $description,
                    e.chunk_id    = $chunk_id
                """,
                name=entity.name,
                type=entity.type,
                description=entity.description,
                chunk_id=entity.chunk_id,
            )
        self._entity_name_cache = None  # invalidate cache on write

    def upsert_relation(self, relation: Relation) -> None:
        """
        Insert or update a directed relation between two Entity nodes.

        Silently skips if either endpoint entity does not exist in the graph —
        this can happen when a relation references an entity that failed validation
        during extraction.
        """
        query = _RELATION_QUERIES.get(relation.relation_type)
        if query is None:
            logger.warning("Unknown relation type '%s' — skipping", relation.relation_type)
            return

        with self._driver.session() as session:
            session.run(
                query,
                source=relation.source,
                target=relation.target,
                description=relation.description,
            )

    # ------------------------------------------------------------------
    # User Knowledge Graph API
    # ------------------------------------------------------------------

    def upsert_user_entity(self, entity: Entity) -> None:
        """
        Insert or update a UserEntity node in the user's personal knowledge graph.

        UserEntity nodes are distinct from book-indexed Entity nodes so queries
        can target either independently.
        """
        with self._driver.session() as session:
            session.run(
                """
                MERGE (e:UserEntity {name: $name})
                SET e.type        = $type,
                    e.description = $description,
                    e.chunk_id    = $chunk_id
                """,
                name=entity.name,
                type=entity.type,
                description=entity.description,
                chunk_id=entity.chunk_id,
            )
        self._user_entity_name_cache = None

    def upsert_user_relation(self, relation: Relation) -> None:
        """Insert or update a relation between two UserEntity nodes."""
        query = _USER_RELATION_QUERIES.get(relation.relation_type)
        if query is None:
            logger.warning("Unknown relation type '%s' — skipping", relation.relation_type)
            return
        with self._driver.session() as session:
            session.run(
                query,
                source=relation.source,
                target=relation.target,
                description=relation.description,
            )

    def get_user_entity_names(self) -> list[str]:
        """Return all entity names in the user's knowledge graph (cached)."""
        if self._user_entity_name_cache is not None:
            return self._user_entity_name_cache
        with self._driver.session() as session:
            result = session.run("MATCH (e:UserEntity) RETURN e.name AS name")
            self._user_entity_name_cache = [r["name"] for r in result]
        return self._user_entity_name_cache

    def get_user_graph_data(self) -> dict:
        """
        Return all UserEntity nodes and their derived edges for graph visualization.

        Edges are not stored on UserEntity nodes directly. Instead they are derived
        from the book KG (Entity relations) by finding all typed relations where
        both endpoints are concepts the user has already approved. This means edges
        appear automatically as the book KG grows (lazily, via KGUpdateSuggester)
        without needing an explicit "add edge" step from the user.

        Returns a dict with:
            "nodes": list of {"name": str, "type": str, "description": str}
            "edges": list of {"from": str, "type": str, "to": str}
        """
        with self._driver.session() as session:
            node_result = session.run(
                """
                MATCH (e:UserEntity)
                RETURN e.name AS name, e.type AS type, e.description AS description
                """
            )
            nodes = [
                {
                    "name": r["name"],
                    "type": r["type"] or "CONCEPT",
                    "description": r["description"] or "",
                }
                for r in node_result
            ]

            if not nodes:
                return {"nodes": [], "edges": []}

            # Derive edges: typed relations in the book KG between user-known concepts.
            user_names = [n["name"] for n in nodes]
            edge_result = session.run(
                """
                MATCH (s:Entity)-[r]->(t:Entity)
                WHERE s.name IN $names AND t.name IN $names
                RETURN s.name AS from_name, type(r) AS rel_type, t.name AS to_name
                """,
                names=user_names,
            )
            edges = [
                {"from": r["from_name"], "type": r["rel_type"], "to": r["to_name"]}
                for r in edge_result
            ]

        return {"nodes": nodes, "edges": edges}

    def get_user_related_entities(
        self,
        entity_name: str,
        depth: int = 1,
    ) -> list[Entity]:
        """
        Return UserEntity nodes reachable from entity_name via book KG relations.

        Traversal goes through the book KG (Entity nodes) but results are filtered
        to concepts the user has already approved (UserEntity nodes). This means:
        - Traversal uses semantically typed edges (IS_A, PART_OF, etc.)
        - Only concepts the user knows appear in the result
        - As the user approves more concepts, the neighborhood automatically expands
        """
        with self._driver.session() as session:
            # Get all UserEntity names to restrict traversal results.
            user_names_result = session.run(
                "MATCH (e:UserEntity) RETURN e.name AS name"
            )
            user_names = [r["name"] for r in user_names_result]

            if not user_names:
                return []

            # Traverse via book KG, filter to user-known concepts.
            traversal_result = session.run(
                f"""
                MATCH (center:Entity {{name: $name}})-[*1..{depth}]-(related:Entity)
                WHERE related.name IN $user_names AND related.name <> $name
                RETURN DISTINCT related.name AS related_name
                LIMIT 20
                """,
                name=entity_name,
                user_names=user_names,
            )
            related_names = [r["related_name"] for r in traversal_result]

            if not related_names:
                return []

            # Fetch UserEntity data for matched names (type, description from user KG).
            user_entity_result = session.run(
                "MATCH (e:UserEntity) WHERE e.name IN $names RETURN e",
                names=related_names,
            )
            return [_record_to_entity(r["e"]) for r in user_entity_result]

    def count_user_entities(self) -> int:
        with self._driver.session() as session:
            result = session.run("MATCH (e:UserEntity) RETURN count(e) AS n")
            return result.single()["n"]

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_related_entities(
        self,
        entity_name: str,
        depth: int = 1,
    ) -> list[Entity]:
        """
        Return entities reachable from entity_name within `depth` hops.

        Uses undirected traversal so both IS_A and inverse IS_A edges count.
        Capped at 20 results to prevent runaway traversal on dense graphs.
        """
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (e:Entity {{name: $name}})-[*1..{depth}]-(related:Entity)
                WHERE related.name <> $name
                RETURN DISTINCT related
                LIMIT 20
                """,
                name=entity_name,
            )
            return [_record_to_entity(r["related"]) for r in result]

    def get_all_entity_names(self) -> list[str]:
        """Return all entity names in the graph (cached after first call)."""
        if self._entity_name_cache is not None:
            return self._entity_name_cache

        with self._driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN e.name AS name")
            self._entity_name_cache = [r["name"] for r in result]

        logger.debug("Entity name cache loaded: %d entities", len(self._entity_name_cache))
        return self._entity_name_cache

    def count_entities(self) -> int:
        with self._driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN count(e) AS n")
            return result.single()["n"]

    def count_relations(self) -> int:
        with self._driver.session() as session:
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS n")
            return result.single()["n"]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Delete all nodes and relationships. Useful for tests and re-runs."""
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        self._entity_name_cache = None
        self._user_entity_name_cache = None
        logger.info("GraphStore cleared.")

    def delete_user_entity(self, name: str) -> None:
        """Delete a single UserEntity node by name."""
        with self._driver.session() as session:
            session.run("MATCH (e:UserEntity {name: $name}) DETACH DELETE e", name=name)
        self._user_entity_name_cache = None

    def clear_user_graph(self) -> None:
        """Delete only UserEntity nodes and their relationships."""
        with self._driver.session() as session:
            session.run("MATCH (e:UserEntity) DETACH DELETE e")
        self._user_entity_name_cache = None
        logger.info("User knowledge graph cleared.")

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> GraphStore:
        return self

    def __exit__(self, *_) -> None:
        self.close()


def _record_to_entity(node) -> Entity:
    """Convert a Neo4j Node object to an Entity dataclass."""
    return Entity(
        name=node["name"],
        type=node["type"],
        description=node.get("description", ""),
        chunk_id=node.get("chunk_id", ""),
    )
