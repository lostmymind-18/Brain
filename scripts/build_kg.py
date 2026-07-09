"""
Build the knowledge graph from a PDF, then run a sample GraphRAG query.

Usage:
    python scripts/build_kg.py                          # build from DDIA, first 20 chunks
    python scripts/build_kg.py --max-chunks 50          # larger sample
    python scripts/build_kg.py --all                    # full book (slow, ~550 LLM calls)
    python scripts/build_kg.py --skip-build             # skip build, query existing graph
    python scripts/build_kg.py --clear                  # clear graph before building
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
import anthropic

from bookmind_tutor.ingestion import HierarchicalChunker, PdfExtractor, StructureDetector
from bookmind_tutor.knowledge_graph import (
    EntityExtractor,
    GraphRAGRetriever,
    GraphStore,
    KnowledgeGraphBuilder,
)
from bookmind_tutor.retrieval.indexer import ChunkIndexer
from bookmind_tutor.retrieval.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_kg")

DEFAULT_PDF = "data/pdf/Designing Data Intensive Applications by Martin Kleppmann.pdf"
DEFAULT_MAX_CHUNKS = 20

SAMPLE_QUERIES = [
    "How does Raft achieve fault tolerance?",
    "What are the trade-offs of eventual consistency?",
    "Compare consensus algorithms",
]


def build_graph(pdf_path: str, max_chunks: int | None, clear: bool) -> tuple[VectorStore, GraphStore]:
    llm_client = anthropic.Anthropic()

    # --- Index chunks into vector store ---
    store = VectorStore()
    indexer = ChunkIndexer(vector_store=store, llm_client=llm_client)
    logger.info("Indexing PDF into vector store...")
    n_chunks = indexer.index_pdf(pdf_path)
    logger.info("Indexed %d chunks", n_chunks)

    # --- Extract raw chunks for KG building ---
    extractor_pdf = PdfExtractor()
    pages = extractor_pdf.extract(pdf_path)
    toc = extractor_pdf.extract_toc(pdf_path)
    detector = StructureDetector(llm_client=llm_client)
    tree = detector.detect(pages, toc=toc)
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(tree)

    # --- Build knowledge graph ---
    graph = GraphStore(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "bookmind123"),
    )

    if clear:
        logger.info("Clearing existing graph...")
        graph.clear()

    kg_extractor = EntityExtractor(llm_client=llm_client)
    builder = KnowledgeGraphBuilder(graph_store=graph, extractor=kg_extractor)

    logger.info("Building knowledge graph (max_chunks=%s)...", max_chunks or "all")
    stats = builder.build_from_chunks(chunks, max_chunks=max_chunks)

    print(f"\n{'='*50}")
    print(f"Build complete:")
    print(f"  Chunks processed : {stats.chunks_processed}/{stats.total_chunks}")
    print(f"  Chunks failed    : {stats.chunks_failed}")
    print(f"  Entities         : {stats.total_entities}")
    print(f"  Relations        : {stats.total_relations}")
    print(f"  Time             : {stats.elapsed_seconds:.1f}s")
    print(f"  Neo4j total      : {graph.count_entities()} entities, {graph.count_relations()} relations")
    print(f"{'='*50}\n")

    return store, graph


def run_queries(vector_store: VectorStore, graph_store: GraphStore) -> None:
    retriever = GraphRAGRetriever(vector_store=vector_store, graph_store=graph_store, k=5)

    print("--- GraphRAG vs Plain Vector Search ---\n")
    for query in SAMPLE_QUERIES:
        print(f"Query: {query}")

        graph_results = retriever.search(query)
        plain_results = vector_store.search(query, k=5)

        graph_ids = {r.chunk_id for r in graph_results}
        plain_ids = {r.chunk_id for r in plain_results}
        extra = graph_ids - plain_ids

        print(f"  Plain vector: {len(plain_results)} results")
        print(f"  GraphRAG    : {len(graph_results)} results  (+{len(extra)} new from graph expansion)")
        if extra:
            for r in graph_results:
                if r.chunk_id in extra:
                    print(f"    [graph-only] {r.chapter} p.{r.page_range[0]} (score={r.score:.3f})")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=DEFAULT_PDF)
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS)
    parser.add_argument("--all", action="store_true", help="Process all chunks")
    parser.add_argument("--skip-build", action="store_true", help="Skip build, query existing graph")
    parser.add_argument("--clear", action="store_true", help="Clear graph before building")
    args = parser.parse_args()

    pdf_path = str(Path(args.pdf).expanduser().resolve())
    if not Path(pdf_path).exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    max_chunks = None if args.all else args.max_chunks

    if args.skip_build:
        graph = GraphStore(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "bookmind123"),
        )
        vector_store = VectorStore()
        llm_client = anthropic.Anthropic()
        indexer = ChunkIndexer(vector_store=vector_store, llm_client=llm_client)
        indexer.index_pdf(pdf_path)
    else:
        vector_store, graph = build_graph(pdf_path, max_chunks, args.clear)

    run_queries(vector_store, graph)
    graph.close()
