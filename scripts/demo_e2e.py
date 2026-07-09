"""
End-to-end integration demo: PDF ingestion → KG → GraphRAGSearchTool → AgentHarness.

This is the Week 4 integration checkpoint: verifies the full pipeline
(ingestion → reasoning harness → knowledge graph) works together before
adding the QA/pedagogical layer in Week 5-6.

Usage:
    python scripts/demo_e2e.py                    # interactive chat with DDIA
    python scripts/demo_e2e.py --strategy react   # force specific strategy
    python scripts/demo_e2e.py --no-kg            # plain vector search (baseline)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import anthropic

from bookmind_tutor.agents.harness import AgentHarness
from bookmind_tutor.agents.strategies import (
    PlanExecuteStrategy,
    ReActStrategy,
    ReflexionStrategy,
    StrategyRouter,
)
from bookmind_tutor.agents.tools.book_search import BookSearchTool
from bookmind_tutor.agents.tools.graph_rag_search import GraphRAGSearchTool
from bookmind_tutor.ingestion import HierarchicalChunker, PdfExtractor, StructureDetector
from bookmind_tutor.knowledge_graph import (
    EntityExtractor,
    GraphRAGRetriever,
    GraphStore,
    KnowledgeGraphBuilder,
)
from bookmind_tutor.retrieval.indexer import ChunkIndexer
from bookmind_tutor.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.WARNING)
logging.getLogger("bookmind_tutor.agents.harness").setLevel(logging.INFO)

DEFAULT_PDF = "data/pdf/Designing Data Intensive Applications by Martin Kleppmann.pdf"
DEFAULT_MAX_CHUNKS = 20


def setup_pipeline(pdf_path: str, max_chunks: int, use_kg: bool) -> StrategyRouter:
    llm_client = anthropic.Anthropic()

    print("Indexing PDF into vector store...")
    t0 = time.time()
    vector_store = VectorStore()
    indexer = ChunkIndexer(vector_store=vector_store, llm_client=llm_client)
    n = indexer.index_pdf(pdf_path)
    print(f"  {n} chunks indexed ({time.time() - t0:.1f}s)")

    if use_kg:
        print(f"Building knowledge graph (first {max_chunks} chunks)...")
        t0 = time.time()

        extractor_pdf = PdfExtractor()
        pages = extractor_pdf.extract(pdf_path)
        toc = extractor_pdf.extract_toc(pdf_path)
        detector = StructureDetector(llm_client=llm_client)
        tree = detector.detect(pages, toc=toc)
        chunks = HierarchicalChunker().chunk(tree)

        graph = GraphStore(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "bookmind123"),
        )
        kg_extractor = EntityExtractor(llm_client=llm_client)
        builder = KnowledgeGraphBuilder(graph_store=graph, extractor=kg_extractor)
        stats = builder.build_from_chunks(chunks, max_chunks=max_chunks)
        elapsed = time.time() - t0
        print(f"  {stats.total_entities} entities, {stats.total_relations} relations ({elapsed:.1f}s)")

        retriever = GraphRAGRetriever(vector_store=vector_store, graph_store=graph)
        tool = GraphRAGSearchTool(retriever=retriever)
        print("  Using GraphRAG search tool\n")
    else:
        tool = BookSearchTool(vector_store=vector_store)
        print("  Using plain vector search tool\n")

    harness = AgentHarness(llm_client=llm_client, tools=[tool])
    react = ReActStrategy(harness=harness)
    plan = PlanExecuteStrategy(harness=harness, llm_client=llm_client)
    reflexion = ReflexionStrategy(harness=harness, llm_client=llm_client)
    return StrategyRouter(react=react, plan_execute=plan, reflexion=reflexion)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=DEFAULT_PDF)
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS)
    parser.add_argument("--no-kg", action="store_true", help="Skip KG, use plain vector search")
    args = parser.parse_args()

    pdf_path = str(Path(args.pdf).expanduser().resolve())
    if not Path(pdf_path).exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    router = setup_pipeline(pdf_path, args.max_chunks, use_kg=not args.no_kg)

    print("BookMind Tutor - end-to-end demo (type 'quit' to exit)\n")
    print("-" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        t0 = time.time()
        answer = router.chat(question)
        elapsed = time.time() - t0

        trace = router.last_trace
        if trace:
            print(f"\n[{trace.strategy_name} | {trace.num_llm_calls} LLM calls | {elapsed:.1f}s]")
        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    main()
