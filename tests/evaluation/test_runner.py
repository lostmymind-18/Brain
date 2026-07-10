"""Tests for EvalRunner and EvalReport."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bookmind_tutor.agents.models import ReasoningTrace
from bookmind_tutor.evaluation.evaluator import AnswerEvaluator, EvalResult
from bookmind_tutor.evaluation.logger import InteractionLogger, InteractionRecord
from bookmind_tutor.evaluation.runner import EvalReport, EvalRow, EvalRunner, RunConfig


def _eval_result(fa: float = 0.8, cq: float = 0.7, dr: float = 0.9, hs: float = 0.1) -> EvalResult:
    return EvalResult(
        factual_accuracy=fa,
        citation_quality=cq,
        depth_and_relevance=dr,
        hallucination_score=hs,
        overall=(fa + cq + dr) / 3.0,
        reasoning="Good answer.",
        model_used="gpt-4o-mini",
    )


def _mock_agent(answer: str = "The answer is 42.", chunks: list[str] | None = None) -> MagicMock:
    agent = MagicMock()
    agent.chat.return_value = answer
    trace = MagicMock(spec=ReasoningTrace)
    trace.retrieved_chunks = chunks or ["chunk text A", "chunk text B"]
    trace.num_llm_calls = 2
    agent.last_trace = trace
    return agent


def _mock_evaluator(result: EvalResult | None = None) -> MagicMock:
    evaluator = MagicMock(spec=AnswerEvaluator)
    evaluator.evaluate.return_value = result or _eval_result()
    return evaluator


def _runner(
    tmp_path: Path,
    evaluator: AnswerEvaluator | None = None,
    logger: InteractionLogger | None = None,
) -> tuple[EvalRunner, MagicMock]:
    """Build a runner with a fake _build_agent override."""
    agent = _mock_agent()
    runner = EvalRunner(
        book_id="mybook",
        vector_store=MagicMock(),
        graph_store=MagicMock(),
        evaluator=evaluator or _mock_evaluator(),
        interaction_logger=logger or InteractionLogger(path=tmp_path / "log.jsonl"),
        report_dir=tmp_path / "reports",
    )
    # Override _build_agent so we don't need real providers
    runner._build_agent = lambda config: agent
    return runner, agent


class TestEvalReport:
    def test_summary_by_config_single(self) -> None:
        rows = [
            EvalRow("q1", "Q1?", "cfg-a", "A1", 0.8, 0.7, 0.9, 0.1, 0.8, 1.0),
            EvalRow("q2", "Q2?", "cfg-a", "A2", 0.6, 0.5, 0.7, 0.2, 0.6, 1.5),
        ]
        report = EvalReport(timestamp="t", book_id="b", configs=[], rows=rows)
        summary = report.summary_by_config()
        assert "cfg-a" in summary
        assert summary["cfg-a"]["factual_accuracy"] == pytest.approx(0.7)
        assert summary["cfg-a"]["n_questions"] == 2.0

    def test_summary_by_config_multiple(self) -> None:
        rows = [
            EvalRow("q1", "Q1?", "a", "A", 0.9, 0.9, 0.9, 0.0, 0.9, 1.0),
            EvalRow("q1", "Q1?", "b", "B", 0.5, 0.5, 0.5, 0.5, 0.5, 2.0),
        ]
        report = EvalReport(timestamp="t", book_id="b", configs=[], rows=rows)
        summary = report.summary_by_config()
        assert summary["a"]["overall"] == pytest.approx(0.9)
        assert summary["b"]["overall"] == pytest.approx(0.5)

    def test_to_markdown_table_contains_headers(self) -> None:
        rows = [EvalRow("q1", "What is X?", "cfg", "X is Y.", 0.9, 0.8, 0.7, 0.1, 0.8, 1.0)]
        report = EvalReport(timestamp="t", book_id="b", configs=[], rows=rows)
        table = report.to_markdown_table()
        assert "Question" in table
        assert "Overall" in table
        assert "cfg" in table

    def test_to_csv_contains_rows(self) -> None:
        rows = [EvalRow("q1", "What is X?", "cfg", "X is Y.", 0.9, 0.8, 0.7, 0.1, 0.8, 1.0)]
        report = EvalReport(timestamp="t", book_id="b", configs=[], rows=rows)
        csv_text = report.to_csv()
        assert "question_id" in csv_text
        assert "q1" in csv_text
        assert "What is X?" in csv_text


class TestEvalRunner:
    def test_rows_created_per_question_config(self, tmp_path: Path) -> None:
        runner, _ = _runner(tmp_path)
        questions = [{"id": "q1", "question": "Q1?"}, {"id": "q2", "question": "Q2?"}]
        config = RunConfig(provider="openai", model="gpt-4o-mini", strategy="react", label="test")
        report = runner.run(questions=questions, configs=[config])
        assert len(report.rows) == 2
        assert {r.question_id for r in report.rows} == {"q1", "q2"}

    def test_multiple_configs_cross_product(self, tmp_path: Path) -> None:
        agent = _mock_agent()
        runner = EvalRunner(
            book_id="b",
            vector_store=MagicMock(),
            graph_store=MagicMock(),
            evaluator=_mock_evaluator(),
            interaction_logger=InteractionLogger(path=tmp_path / "log.jsonl"),
            report_dir=tmp_path / "reports",
        )
        runner._build_agent = lambda config: agent
        questions = [{"id": "q1", "question": "Q1?"}]
        configs = [
            RunConfig("openai", "gpt-4o-mini", "react", "react"),
            RunConfig("openai", "gpt-4o-mini", "plan-execute", "plan"),
        ]
        report = runner.run(questions=questions, configs=configs)
        # 1 question x 2 configs = 2 rows
        assert len(report.rows) == 2
        labels = {r.config_label for r in report.rows}
        assert labels == {"react", "plan"}

    def test_interaction_logger_called(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        logger = InteractionLogger(path=log_path)
        runner, _ = _runner(tmp_path, logger=logger)
        questions = [{"id": "q1", "question": "Q1?"}]
        config = RunConfig("openai", "gpt-4o-mini", "react", "test")
        runner.run(questions=questions, configs=[config])
        records = logger.load_all()
        assert len(records) == 1
        assert records[0].question == "Q1?"

    def test_eval_scores_in_log_record(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        logger = InteractionLogger(path=log_path)
        evaluator = _mock_evaluator(_eval_result(fa=0.9, cq=0.8, dr=0.7))
        runner, _ = _runner(tmp_path, evaluator=evaluator, logger=logger)
        questions = [{"id": "q1", "question": "Q1?"}]
        config = RunConfig("openai", "gpt-4o-mini", "react", "test")
        runner.run(questions=questions, configs=[config])
        records = logger.load_all()
        assert records[0].eval_score == pytest.approx((0.9 + 0.8 + 0.7) / 3.0)
        assert records[0].eval_breakdown["factual_accuracy"] == pytest.approx(0.9)

    def test_report_json_saved_to_disk(self, tmp_path: Path) -> None:
        runner, _ = _runner(tmp_path)
        questions = [{"id": "q1", "question": "Q1?"}]
        config = RunConfig("openai", "gpt-4o-mini", "react", "test")
        runner.run(questions=questions, configs=[config])
        report_files = list((tmp_path / "reports").glob("*.json"))
        assert len(report_files) == 1
        import json
        data = json.loads(report_files[0].read_text())
        assert data["book_id"] == "mybook"
        assert len(data["rows"]) == 1

    def test_retrieved_chunks_passed_to_evaluator(self, tmp_path: Path) -> None:
        evaluator = _mock_evaluator()
        runner, agent = _runner(tmp_path, evaluator=evaluator)
        agent.last_trace.retrieved_chunks = ["specific chunk text"]
        questions = [{"id": "q1", "question": "Q1?"}]
        config = RunConfig("openai", "gpt-4o-mini", "react", "test")
        runner.run(questions=questions, configs=[config])
        call_kwargs = evaluator.evaluate.call_args
        assert "specific chunk text" in call_kwargs.args[1]  # chunks is positional arg 2
