"""
Side-by-side comparison of ReAct, Plan-and-Execute, and Reflexion strategies.

Usage:
    python scripts/demo_strategies.py
    python scripts/demo_strategies.py --pdf data/pdf/my_book.pdf
    python scripts/demo_strategies.py --strategy react       # single strategy
    python scripts/demo_strategies.py --strategy plan
    python scripts/demo_strategies.py --strategy reflexion
    python scripts/demo_strategies.py --strategy auto        # StrategyRouter
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

import anthropic

from bookmind_tutor.agents.harness import AgentHarness
from bookmind_tutor.agents.strategies import (
    PlanExecuteStrategy,
    ReActStrategy,
    ReflexionStrategy,
    StrategyRouter,
)
from bookmind_tutor.agents.tools.book_search import BookSearchTool
from bookmind_tutor.retrieval.indexer import ChunkIndexer
from bookmind_tutor.retrieval.vector_store import VectorStore

logging.basicConfig(
    level=logging.WARNING,  # suppress harness internals; show only demo output
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
# Show router/strategy decisions without full harness noise
logging.getLogger("bookmind_tutor.agents.strategies").setLevel(logging.INFO)
logging.getLogger("demo_strategies").setLevel(logging.INFO)

logger = logging.getLogger("demo_strategies")

DEFAULT_PDF = "data/pdf/Designing Data Intensive Applications by Martin Kleppmann.pdf"

# Three questions designed to stress-test different strategies
COMPARISON_QUESTIONS = [
    {
        "question": "What is eventual consistency?",
        "note": "Simple lookup — expect all strategies to perform similarly",
    },
    {
        "question": "Compare leader-based replication and leaderless replication: mechanisms, trade-offs, and real-world examples",
        "note": "Multi-part comparison — Plan-and-Execute should shine here",
    },
    {
        "question": "Explain in depth how consensus algorithms like Raft achieve fault tolerance",
        "note": "Depth-sensitive — Reflexion may improve the initial answer",
    },
]


def index_book(pdf_path: str, llm_client) -> VectorStore:
    store = VectorStore()  # ephemeral — fresh each run
    logger.info("Indexing %s ...", pdf_path)
    t0 = time.monotonic()
    indexer = ChunkIndexer(vector_store=store, llm_client=llm_client)
    n = indexer.index_pdf(pdf_path)
    logger.info("Indexed %d chunks in %.1fs", n, time.monotonic() - t0)
    return store


def build_strategies(store: VectorStore, llm_client) -> dict:
    harness = AgentHarness(tools=[BookSearchTool(store, k=5)], llm_client=llm_client)
    react = ReActStrategy(harness=harness)
    plan = PlanExecuteStrategy(harness=harness, llm_client=llm_client)
    reflexion = ReflexionStrategy(harness=harness, llm_client=llm_client)
    router = StrategyRouter(react=react, plan_execute=plan, reflexion=reflexion)
    return {"react": react, "plan": plan, "reflexion": reflexion, "auto": router}


def run_comparison(strategies: dict, pdf_name: str) -> None:
    print(f"\n{'='*70}")
    print(f"BookMind Tutor — Strategy Comparison")
    print(f"Book: {pdf_name}")
    print(f"{'='*70}\n")

    for entry in COMPARISON_QUESTIONS:
        question = entry["question"]
        note = entry["note"]

        print(f"QUESTION: {question}")
        print(f"Note:     {note}")
        print()

        for label, strategy in strategies.items():
            print(f"  [{label.upper()}]")
            t0 = time.monotonic()
            answer = strategy.chat(question)
            elapsed = time.monotonic() - t0

            trace = strategy.last_trace
            steps_summary = ""
            if trace and trace.steps:
                steps_summary = f" | steps: {len(trace.steps)}"

            print(f"  Time: {elapsed:.1f}s{steps_summary}")
            print()
            # Print first 600 chars of answer to keep output readable
            preview = answer[:600] + ("..." if len(answer) > 600 else "")
            for line in preview.splitlines():
                print(f"    {line}")
            print()

        print("-" * 70)
        print()


def run_interactive(strategies: dict, strategy_key: str) -> None:
    strategy = strategies[strategy_key]
    print(f"\n{'='*60}")
    print(f"BookMind Tutor — Strategy: {strategy_key.upper()}")
    print("Type your question and press Enter. Type 'quit' to exit.")
    print(f"{'='*60}\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        t0 = time.monotonic()
        answer = strategy.chat(question)
        elapsed = time.monotonic() - t0

        trace = strategy.last_trace
        if trace and trace.steps:
            print(f"\n[Trace] {' → '.join(trace.steps)}")
        print(f"\nAnswer ({elapsed:.1f}s):\n{answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare reasoning strategies on a book.")
    parser.add_argument("--pdf", default=DEFAULT_PDF)
    parser.add_argument(
        "--strategy",
        choices=["react", "plan", "reflexion", "auto", "compare"],
        default="compare",
        help="Which strategy to use. 'compare' runs all three side-by-side (default).",
    )
    args = parser.parse_args()

    pdf_path = str(Path(args.pdf).expanduser().resolve())
    if not Path(pdf_path).exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    llm_client = anthropic.Anthropic()
    store = index_book(pdf_path, llm_client)
    strategies = build_strategies(store, llm_client)

    if args.strategy == "compare":
        run_comparison(
            {k: v for k, v in strategies.items() if k != "auto"},
            Path(pdf_path).name,
        )
    else:
        run_interactive(strategies, args.strategy)
