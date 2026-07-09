"""Eval runner: benchmark multiple (provider, model, strategy) configs.

For each (question, config) pair the runner:
1. Builds a fresh QAAgent with the config's provider/model
2. Calls QAAgent.chat() and records the answer + trace
3. Logs the interaction via InteractionLogger
4. Scores the answer with AnswerEvaluator
5. Writes an updated log record with the eval score
6. Collects all results into an EvalReport

The report is also persisted as JSON in data/eval_reports/.

Run via CLI:
    python -m bookmind_tutor.evaluation --book-id <id> --questions data/eval_questions.json
"""
from __future__ import annotations

import csv
import io
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bookmind_tutor.evaluation.evaluator import AnswerEvaluator, EvalResult
from bookmind_tutor.evaluation.logger import InteractionLogger, InteractionRecord


@dataclass
class RunConfig:
    """One evaluation configuration to benchmark."""
    provider: str
    model: str
    strategy: str           # "react" | "plan-execute" | "reflexion" | "auto"
    label: str              # human-readable column header in the report
    base_url: str | None = None  # custom API base for Ollama/LM Studio


@dataclass
class EvalRow:
    """One (question, config) result."""
    question_id: str
    question: str
    config_label: str
    answer: str
    factual_accuracy: float
    citation_quality: float
    depth_and_relevance: float
    overall: float
    elapsed_seconds: float


@dataclass
class EvalReport:
    timestamp: str
    book_id: str
    configs: list[RunConfig]
    rows: list[EvalRow] = field(default_factory=list)

    def summary_by_config(self) -> dict[str, dict[str, float]]:
        """Mean scores per config label, plus question count."""
        buckets: dict[str, list[EvalRow]] = defaultdict(list)
        for row in self.rows:
            buckets[row.config_label].append(row)
        summary: dict[str, dict[str, float]] = {}
        for label, rows in buckets.items():
            n = len(rows)
            summary[label] = {
                "factual_accuracy": sum(r.factual_accuracy for r in rows) / n,
                "citation_quality": sum(r.citation_quality for r in rows) / n,
                "depth_and_relevance": sum(r.depth_and_relevance for r in rows) / n,
                "overall": sum(r.overall for r in rows) / n,
                "n_questions": float(n),
            }
        return summary

    def to_markdown_table(self) -> str:
        header = "| Question | Config | Accuracy | Citation | Depth | Overall | Time |"
        sep    = "|---------|--------|----------|---------|-------|---------|------|"
        lines = [header, sep]
        for row in self.rows:
            q = (row.question[:40] + "...") if len(row.question) > 40 else row.question
            lines.append(
                f"| {q} | {row.config_label} "
                f"| {row.factual_accuracy:.2f} | {row.citation_quality:.2f} "
                f"| {row.depth_and_relevance:.2f} | {row.overall:.2f} "
                f"| {row.elapsed_seconds:.1f}s |"
            )
        return "\n".join(lines)

    def to_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=[
            "question_id", "question", "config_label", "answer",
            "factual_accuracy", "citation_quality", "depth_and_relevance",
            "overall", "elapsed_seconds",
        ])
        writer.writeheader()
        for row in self.rows:
            writer.writerow({
                "question_id": row.question_id,
                "question": row.question,
                "config_label": row.config_label,
                "answer": row.answer,
                "factual_accuracy": row.factual_accuracy,
                "citation_quality": row.citation_quality,
                "depth_and_relevance": row.depth_and_relevance,
                "overall": row.overall,
                "elapsed_seconds": row.elapsed_seconds,
            })
        return buf.getvalue()


class EvalRunner:
    """Run benchmark questions against one or more configs and produce a report."""

    def __init__(
        self,
        book_id: str,
        vector_store,
        graph_store,
        evaluator: AnswerEvaluator,
        interaction_logger: InteractionLogger | None = None,
        report_dir: Path | str = Path("data/eval_reports"),
    ) -> None:
        self._book_id = book_id
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._evaluator = evaluator
        self._logger = interaction_logger or InteractionLogger()
        self._report_dir = Path(report_dir)
        self._report_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        questions: list[dict],
        configs: list[RunConfig],
        book_name: str = "",
    ) -> EvalReport:
        """
        Run all (question, config) pairs and return an EvalReport.

        The report JSON is also written to data/eval_reports/<timestamp>.json.
        """
        ts = datetime.now(timezone.utc).isoformat()
        report = EvalReport(timestamp=ts, book_id=self._book_id, configs=configs)

        for config in configs:
            agent = self._build_agent(config)
            for q in questions:
                row = self._run_one(q, config, agent, book_name)
                report.rows.append(row)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = self._report_dir / f"{stamp}.json"
        report_path.write_text(
            json.dumps({
                "timestamp": report.timestamp,
                "book_id": report.book_id,
                "summary": report.summary_by_config(),
                "rows": [
                    {
                        "question_id": r.question_id,
                        "question": r.question,
                        "config_label": r.config_label,
                        "answer": r.answer,
                        "factual_accuracy": r.factual_accuracy,
                        "citation_quality": r.citation_quality,
                        "depth_and_relevance": r.depth_and_relevance,
                        "overall": r.overall,
                        "elapsed_seconds": r.elapsed_seconds,
                    }
                    for r in report.rows
                ],
            }, indent=2),
            encoding="utf-8",
        )
        return report

    def _run_one(
        self, q: dict, config: RunConfig, agent, book_name: str
    ) -> EvalRow:
        question = q["question"]
        qid = q.get("id", str(uuid.uuid4())[:8])
        force = config.strategy if config.strategy != "auto" else None

        t0 = time.monotonic()
        answer = agent.chat(question, force_strategy=force)
        elapsed = time.monotonic() - t0

        trace = agent.last_trace
        retrieved_chunks = trace.retrieved_chunks if trace else []
        num_llm_calls = trace.num_llm_calls if trace else 0

        eval_result: EvalResult = self._evaluator.evaluate(question, retrieved_chunks, answer)

        # Log the interaction with eval scores already filled in
        self._logger.log(InteractionRecord(
            record_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            book_id=self._book_id,
            book_name=book_name,
            question=question,
            retrieved_chunks=retrieved_chunks,
            answer=answer,
            strategy=config.strategy,
            model=config.model,
            provider=config.provider,
            num_llm_calls=num_llm_calls,
            elapsed_seconds=elapsed,
            eval_score=eval_result.overall,
            eval_breakdown={
                "factual_accuracy": eval_result.factual_accuracy,
                "citation_quality": eval_result.citation_quality,
                "depth_and_relevance": eval_result.depth_and_relevance,
            },
        ))

        return EvalRow(
            question_id=qid,
            question=question,
            config_label=config.label,
            answer=answer,
            factual_accuracy=eval_result.factual_accuracy,
            citation_quality=eval_result.citation_quality,
            depth_and_relevance=eval_result.depth_and_relevance,
            overall=eval_result.overall,
            elapsed_seconds=elapsed,
        )

    def _build_agent(self, config: RunConfig):
        """Build a fresh QAAgent wired to the config's provider and model."""
        from bookmind_tutor.agents.harness import AgentHarness
        from bookmind_tutor.agents.strategies import (
            PlanExecuteStrategy,
            ReActStrategy,
            ReflexionStrategy,
            StrategyRouter,
        )
        from bookmind_tutor.agents.tools.graph_rag_search import GraphRAGSearchTool
        from bookmind_tutor.knowledge_graph.graph_rag import GraphRAGRetriever
        from bookmind_tutor.llm import create_client
        from bookmind_tutor.tutor.qa_agent import QAAgent

        kwargs = {}
        if config.base_url and config.provider == "openai":
            kwargs["base_url"] = config.base_url
        llm_client = create_client(provider=config.provider, model=config.model, **kwargs)
        retriever = GraphRAGRetriever(
            vector_store=self._vector_store,
            graph_store=self._graph_store,
            k=5,
        )
        tool = GraphRAGSearchTool(retriever=retriever)
        harness = AgentHarness(llm_client=llm_client, tools=[tool])
        react = ReActStrategy(harness=harness)
        plan = PlanExecuteStrategy(harness=harness, llm_client=llm_client)
        reflexion = ReflexionStrategy(harness=harness, llm_client=llm_client)
        router = StrategyRouter(react=react, plan_execute=plan, reflexion=reflexion)
        return QAAgent(harness=harness, router=router)
