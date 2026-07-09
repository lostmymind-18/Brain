"""Append-only JSONL interaction log.

Each record captures one completed Q&A turn. The format is designed to be
directly usable as DPO preference data in Week 10-11: question + retrieved
context + answer are the key fields, with eval scores added later.

Update semantics: the log is append-only. To update a record (e.g. add
eval_score after the fact), write a new record with the same record_id and a
later timestamp. load_all() returns the latest record per record_id.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class InteractionRecord:
    record_id: str
    timestamp: str
    book_id: str
    book_name: str
    question: str
    retrieved_chunks: list[str]
    answer: str
    strategy: str           # "react" | "plan_execute" | "reflexion"
    model: str              # e.g. "gpt-4o-mini"
    provider: str           # "openai" | "anthropic"
    num_llm_calls: int
    elapsed_seconds: float
    # Populated later by evaluator or human feedback:
    eval_score: float | None = None
    eval_breakdown: dict | None = None
    human_rating: int | None = None     # 1-5 stars (future)


class InteractionLogger:
    """Write and read the JSONL interaction log."""

    def __init__(self, path: Path | str = Path("data/interactions.jsonl")) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: InteractionRecord) -> None:
        """Append one record. Creates the file if it doesn't exist."""
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")
            f.flush()

    def load_all(self) -> list[InteractionRecord]:
        """
        Read all records from the log.

        Latest record per record_id wins (supports append-only updates).
        Malformed lines are silently skipped.
        """
        if not self._path.exists():
            return []

        latest: dict[str, dict] = {}
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    latest[row["record_id"]] = row
                except (json.JSONDecodeError, KeyError):
                    continue

        records = []
        for row in latest.values():
            try:
                records.append(InteractionRecord(**row))
            except TypeError:
                continue
        return records
