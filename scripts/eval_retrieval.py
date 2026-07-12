"""
Retrieval eval: measure recall@k for a query set against a book's vector store.

Recall@k: fraction of queries where at least one expected_section appears
in the top-k search results (comparing normalized section strings).

Usage:
    python scripts/eval_retrieval.py data/eval/deutsch_retrieval.json
    python scripts/eval_retrieval.py data/eval/fosa_retrieval.json --k 5 10 20
    python scripts/eval_retrieval.py data/eval/ddia_retrieval.json --verbose

Requires the book to be indexed in data/chroma first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bookmind_tutor.ingestion.utils import normalize_heading
from bookmind_tutor.retrieval.vector_store import VectorStore


def _section_matches(result_section: str | None, expected: list[str]) -> bool:
    """True if the result section matches any of the expected section names."""
    if not result_section:
        return False
    norm_result = normalize_heading(result_section)
    return any(normalize_heading(e) in norm_result or norm_result in normalize_heading(e)
               for e in expected)


def run_eval(query_file: Path, k_values: list[int], verbose: bool) -> None:
    data = json.loads(query_file.read_text())
    book_id = data["book_id"]
    book_name = data.get("book_name", book_id)
    queries = data["queries"]

    vs = VectorStore(
        persist_dir="data/chroma",
        collection_name=f"book_{book_id}",
    )
    if vs.count() == 0:
        print(f"ERROR: book '{book_id}' is not indexed in data/chroma.")
        print("Index it via the Streamlit UI or the ingestion pipeline first.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Eval: {book_name}")
    print(f"Index size: {vs.count()} chunks | {len(queries)} queries")
    print(f"{'='*60}\n")

    max_k = max(k_values)
    hits_by_k: dict[int, int] = {k: 0 for k in k_values}
    misses: list[dict] = []

    for q in queries:
        qid = q["id"]
        query = q["query"]
        expected = q["expected_sections"]
        results = vs.search(query, k=max_k)

        hit_at: int | None = None
        for rank, r in enumerate(results, start=1):
            if _section_matches(r.section, expected):
                hit_at = rank
                break

        for k in k_values:
            if hit_at is not None and hit_at <= k:
                hits_by_k[k] += 1

        if verbose:
            status = f"HIT@{hit_at}" if hit_at else "MISS"
            top_section = results[0].section if results else "?"
            print(f"[{status:6s}] {qid}: {query[:55]}")
            print(f"         expected: {expected[0]!r}")
            print(f"         top-1:    {top_section!r} (score={results[0].score:.4f})")
            if hit_at is None and len(results) > 1:
                print(f"         top-3:    {[r.section for r in results[:3]]}")
            print()
        elif hit_at is None:
            misses.append({"id": qid, "query": query, "expected": expected,
                           "top1": results[0].section if results else None})

    n = len(queries)
    print("Recall@k:")
    for k in k_values:
        recall = hits_by_k[k] / n if n else 0
        bar = "#" * int(recall * 30)
        print(f"  @{k:2d}: {recall:.1%}  [{bar:<30s}] ({hits_by_k[k]}/{n})")

    if not verbose and misses:
        print(f"\nMisses at k={max_k} ({len(misses)}):")
        for m in misses:
            print(f"  [{m['id']}] {m['query'][:55]}")
            print(f"           expected: {m['expected'][0]!r} | top-1: {m['top1']!r}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval eval: recall@k for a book eval set")
    parser.add_argument("query_file", type=Path, help="Path to eval query JSON file")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10],
                        help="k values to evaluate (default: 1 5 10)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print result for every query")
    args = parser.parse_args()

    if not args.query_file.exists():
        print(f"ERROR: {args.query_file} not found")
        sys.exit(1)

    run_eval(args.query_file, sorted(args.k), args.verbose)


if __name__ == "__main__":
    main()
