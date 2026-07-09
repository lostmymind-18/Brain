"""
End-to-end demo: index a PDF then chat with it via AgentHarness.

Usage:
    python scripts/demo_chat.py
    python scripts/demo_chat.py --pdf data/pdf/my_book.pdf
    python scripts/demo_chat.py --pdf data/pdf/my_book.pdf --persist data/chroma

By default uses DDIA and an ephemeral (in-memory) ChromaDB store so each run
starts fresh. Pass --persist <dir> to reuse an existing index across runs
(indexing is skipped if the collection already has chunks).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the package is importable when running from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import anthropic

from bookmind_tutor.agents.harness import AgentHarness
from bookmind_tutor.agents.memory import ConversationMemory
from bookmind_tutor.agents.tools.book_search import BookSearchTool
from bookmind_tutor.retrieval.indexer import ChunkIndexer
from bookmind_tutor.retrieval.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo_chat")

DEFAULT_PDF = "data/pdf/Designing Data Intensive Applications by Martin Kleppmann.pdf"

DEMO_QUESTIONS = [
    "What is the difference between replication and partitioning?",
    "Can you explain what a write-ahead log is and why databases use it?",
    "What are the trade-offs of using eventual consistency?",
]


def index_book(pdf_path: str, persist_dir: str | None, llm_client) -> VectorStore:
    store = VectorStore(persist_dir=persist_dir)

    if store.count() > 0:
        logger.info("Store already has %d chunks — skipping indexing.", store.count())
        return store

    logger.info("Indexing %s ...", pdf_path)
    t0 = time.time()
    indexer = ChunkIndexer(vector_store=store, llm_client=llm_client)
    n = indexer.index_pdf(pdf_path)
    logger.info("Indexed %d chunks in %.1fs", n, time.time() - t0)
    return store


def run_demo(pdf_path: str, persist_dir: str | None) -> None:
    pdf_path = str(Path(pdf_path).expanduser().resolve())
    if not Path(pdf_path).exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    llm_client = anthropic.Anthropic()
    store = index_book(pdf_path, persist_dir, llm_client)

    harness = AgentHarness(
        tools=[BookSearchTool(store, k=5)],
        llm_client=llm_client,
    )
    memory = ConversationMemory()

    print("\n" + "=" * 60)
    print(f"BookMind Tutor — {Path(pdf_path).name}")
    print(f"Store: {store.count()} chunks indexed")
    print("=" * 60)
    print("Type your question and press Enter. Type 'quit' to exit.")
    print("Type 'demo' to run preset demo questions.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.lower() == "demo":
            _run_demo_questions(harness, DEMO_QUESTIONS)
            continue

        t0 = time.time()
        answer = harness.chat(user_input, memory=memory)
        elapsed = time.time() - t0

        print(f"\nAssistant ({elapsed:.1f}s):\n{answer}\n")
        print("-" * 60)


def _run_demo_questions(harness: AgentHarness, questions: list[str]) -> None:
    """Run preset questions without memory (stateless, each is independent)."""
    for i, q in enumerate(questions, 1):
        print(f"\n[Demo {i}/{len(questions)}] {q}")
        t0 = time.time()
        answer = harness.chat(q)  # no memory = stateless
        print(f"Answer ({time.time() - t0:.1f}s):\n{answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a book via BookMind Tutor.")
    parser.add_argument(
        "--pdf",
        default=DEFAULT_PDF,
        help=f"Path to PDF file (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "--persist",
        default=None,
        metavar="DIR",
        help="ChromaDB persist directory. If omitted, uses ephemeral in-memory store.",
    )
    args = parser.parse_args()
    run_demo(args.pdf, args.persist)
