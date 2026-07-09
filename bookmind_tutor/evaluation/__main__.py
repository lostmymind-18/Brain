"""CLI entry point for the evaluation runner.

Usage:
    python -m bookmind_tutor.evaluation \\
        --book-id <id> \\
        --questions data/eval_questions.json \\
        [--provider openai] \\
        [--model gpt-4o-mini] \\
        [--strategy react] \\
        [--label "gpt-4o-mini/react"] \\
        [--eval-provider openai] \\
        [--eval-model gpt-4o-mini]

The book-id must match an existing ChromaDB collection (book was previously indexed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from bookmind_tutor.evaluation.evaluator import AnswerEvaluator
from bookmind_tutor.evaluation.runner import EvalRunner, RunConfig
from bookmind_tutor.knowledge_graph.graph_store import GraphStore
from bookmind_tutor.llm import create_client
from bookmind_tutor.retrieval.vector_store import VectorStore

_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "bookmind123")
_CHROMA_DIR = "data/chroma"
_LLM_BASE_URL = os.getenv("LLM_BASE_URL")


def main() -> None:
    parser = argparse.ArgumentParser(description="BookMind Tutor evaluation runner")
    parser.add_argument("--book-id", required=True, help="Book ID (ChromaDB collection suffix)")
    parser.add_argument("--questions", required=True, help="Path to eval_questions.json")
    parser.add_argument("--provider", default="openai", help="LLM provider for the tutor")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model for the tutor")
    parser.add_argument("--strategy", default="react", help="Strategy: react | plan-execute | reflexion | auto")
    parser.add_argument("--label", default=None, help="Human-readable config label for the report")
    parser.add_argument("--eval-provider", default="openai", help="LLM provider for the evaluator judge")
    parser.add_argument("--eval-model", default="gpt-4o-mini", help="LLM model for the evaluator judge")
    parser.add_argument("--base-url", default=_LLM_BASE_URL, help="Custom API base URL (e.g. http://localhost:11434/v1 for Ollama)")
    args = parser.parse_args()

    q_path = Path(args.questions)
    if not q_path.exists():
        print(f"Error: questions file not found: {q_path}", file=sys.stderr)
        sys.exit(1)

    with q_path.open() as f:
        q_data = json.load(f)

    questions = q_data.get("questions", [])
    if not questions:
        print("Error: no questions found in file", file=sys.stderr)
        sys.exit(1)

    label = args.label or f"{args.model}/{args.strategy}"
    config = RunConfig(
        provider=args.provider,
        model=args.model,
        strategy=args.strategy,
        label=label,
        base_url=args.base_url,
    )

    vs = VectorStore(persist_dir=_CHROMA_DIR, collection_name=f"book_{args.book_id}")
    if vs.count() == 0:
        print(f"Error: book '{args.book_id}' not indexed or collection is empty.", file=sys.stderr)
        sys.exit(1)

    graph_store = GraphStore(uri=_NEO4J_URI, user=_NEO4J_USER, password=_NEO4J_PASSWORD)
    eval_client = create_client(provider=args.eval_provider, model=args.eval_model)
    evaluator = AnswerEvaluator(client=eval_client)

    runner = EvalRunner(
        book_id=args.book_id,
        vector_store=vs,
        graph_store=graph_store,
        evaluator=evaluator,
    )

    book_name = q_data.get("book_name", args.book_id)
    print(f"Running {len(questions)} question(s) with config: {label}")
    report = runner.run(questions=questions, configs=[config], book_name=book_name)

    print("\n" + report.to_markdown_table())
    print("\nSummary:")
    for cfg_label, scores in report.summary_by_config().items():
        print(f"\n  {cfg_label}:")
        for k, v in scores.items():
            print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")


if __name__ == "__main__":
    main()
