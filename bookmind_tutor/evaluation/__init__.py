"""Evaluation framework for BookMind Tutor (Week 7).

Components:
  InteractionLogger  — append-only JSONL log of Q&A turns (doubles as DPO seed data)
  AnswerEvaluator   — LLM-as-judge scoring on a 3-dimension rubric
  EvalRunner        — benchmark multiple (provider, model, strategy) configs
"""
from bookmind_tutor.evaluation.evaluator import AnswerEvaluator, EvalResult
from bookmind_tutor.evaluation.logger import InteractionLogger, InteractionRecord
from bookmind_tutor.evaluation.runner import EvalReport, EvalRow, EvalRunner, RunConfig

__all__ = [
    "AnswerEvaluator",
    "EvalResult",
    "EvalReport",
    "EvalRow",
    "EvalRunner",
    "InteractionLogger",
    "InteractionRecord",
    "RunConfig",
]
