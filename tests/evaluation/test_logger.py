"""Tests for InteractionLogger."""
import json
from pathlib import Path

import pytest

from bookmind_tutor.evaluation.logger import InteractionLogger, InteractionRecord


def _record(record_id: str = "r1", question: str = "What is X?") -> InteractionRecord:
    return InteractionRecord(
        record_id=record_id,
        timestamp="2026-07-09T10:00:00+00:00",
        book_id="mybook",
        book_name="My Book",
        question=question,
        retrieved_chunks=["Chunk A text.", "Chunk B text."],
        answer="X is Y.",
        strategy="react",
        model="gpt-4o-mini",
        provider="openai",
        num_llm_calls=2,
        elapsed_seconds=1.23,
    )


class TestInteractionLogger:
    def test_log_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        logger = InteractionLogger(path=path)
        logger.log(_record())
        assert path.exists()

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        logger = InteractionLogger(path=path)
        r = _record()
        logger.log(r)
        loaded = logger.load_all()
        assert len(loaded) == 1
        assert loaded[0].record_id == "r1"
        assert loaded[0].question == "What is X?"
        assert loaded[0].retrieved_chunks == ["Chunk A text.", "Chunk B text."]
        assert loaded[0].eval_score is None

    def test_multiple_records(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        logger = InteractionLogger(path=path)
        logger.log(_record("r1", "Q1"))
        logger.log(_record("r2", "Q2"))
        records = logger.load_all()
        assert len(records) == 2
        ids = {r.record_id for r in records}
        assert ids == {"r1", "r2"}

    def test_latest_record_wins(self, tmp_path: Path) -> None:
        """Writing a second record with same id overwrites the first in load_all."""
        path = tmp_path / "test.jsonl"
        logger = InteractionLogger(path=path)
        r1 = _record("r1")
        r1.eval_score = None
        logger.log(r1)

        r1_updated = _record("r1")
        r1_updated.eval_score = 0.85
        r1_updated.timestamp = "2026-07-09T11:00:00+00:00"
        logger.log(r1_updated)

        records = logger.load_all()
        assert len(records) == 1
        assert records[0].eval_score == 0.85

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        path.write_text('{"record_id": "r1"}\nnot json at all\n{"record_id": "r2"}\n')
        logger = InteractionLogger(path=path)
        # r1 and r2 are missing required fields, so they fail InteractionRecord(**row)
        # and are skipped; the malformed line is also skipped
        records = logger.load_all()
        assert isinstance(records, list)

    def test_skips_json_without_record_id(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        path.write_text('{"question": "hello"}\n')
        logger = InteractionLogger(path=path)
        records = logger.load_all()
        assert records == []

    def test_load_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        logger = InteractionLogger(path=path)
        assert logger.load_all() == []

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.jsonl"
        logger = InteractionLogger(path=path)
        assert logger.load_all() == []

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        path = tmp_path / "subdir" / "nested" / "log.jsonl"
        logger = InteractionLogger(path=path)
        logger.log(_record())
        assert path.exists()

    def test_eval_fields_persisted(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        logger = InteractionLogger(path=path)
        r = _record()
        r.eval_score = 0.75
        r.eval_breakdown = {"factual_accuracy": 0.8, "citation_quality": 0.7, "depth_and_relevance": 0.75}
        r.human_rating = 4
        logger.log(r)
        loaded = logger.load_all()[0]
        assert loaded.eval_score == 0.75
        assert loaded.eval_breakdown == {"factual_accuracy": 0.8, "citation_quality": 0.7, "depth_and_relevance": 0.75}
        assert loaded.human_rating == 4
