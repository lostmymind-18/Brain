"""LLM-as-judge answer quality evaluator.

Scores one answer on three dimensions using a single LLM call.
Retries once with a stricter prompt if JSON parsing fails.
Returns a sentinel EvalResult (all scores = -1.0) on double failure.

Design note: the evaluator model is intentionally separate from the tutor
model. A cheaper, faster model (e.g. gpt-4o-mini) is fine for judging.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from bookmind_tutor.llm.base import LLMClient


_EVAL_PROMPT = """\
You are evaluating the quality of an answer given by an AI book tutor.

QUESTION: {question}

RETRIEVED CONTEXT (from the book):
---
{context}
---

ANSWER:
{answer}

Rate the answer on three dimensions from 0.0 to 1.0.
Return ONLY valid JSON - no explanation outside the JSON:
{{
  "factual_accuracy": <float 0.0-1.0>,
  "citation_quality": <float 0.0-1.0>,
  "depth_and_relevance": <float 0.0-1.0>,
  "reasoning": "<one sentence explanation>"
}}\
"""

_STRICT_SUFFIX = (
    "\n\nIMPORTANT: Reply with ONLY the JSON object. "
    "No markdown fences, no text before or after."
)


@dataclass
class EvalResult:
    factual_accuracy: float
    citation_quality: float
    depth_and_relevance: float
    overall: float          # simple mean of the three dimensions
    reasoning: str
    model_used: str         # which model acted as judge


class AnswerEvaluator:
    """Score an answer against its retrieved context using an LLM as judge."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def evaluate(self, question: str, chunks: list[str], answer: str) -> EvalResult:
        """
        Score one answer.

        Args:
            question: The user's original question.
            chunks: Texts returned by book_search during the answer turn.
            answer: The final answer text produced by the tutor.

        Returns:
            EvalResult with scores in [0.0, 1.0]. If evaluation fails twice,
            returns a sentinel with all scores = -1.0 and reasoning noting the failure.
        """
        context = "\n---\n".join(chunks) if chunks else "(no retrieved context)"
        prompt = _EVAL_PROMPT.format(
            question=question,
            context=context,
            answer=answer,
        )

        result = self._try_parse([{"role": "user", "content": prompt}])
        if result is not None:
            return result

        result = self._try_parse([{"role": "user", "content": prompt + _STRICT_SUFFIX}])
        if result is not None:
            return result

        return EvalResult(
            factual_accuracy=-1.0,
            citation_quality=-1.0,
            depth_and_relevance=-1.0,
            overall=-1.0,
            reasoning="Evaluation failed: could not parse LLM response after retry.",
            model_used=self._client.model,
        )

    def _try_parse(self, messages: list[dict]) -> EvalResult | None:
        try:
            response = self._client.complete(messages=messages, max_tokens=256)
        except Exception:
            return None

        raw = response.text.strip()
        # Strip markdown code fences if the model wrapped the JSON
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        try:
            fa = float(data["factual_accuracy"])
            cq = float(data["citation_quality"])
            dr = float(data["depth_and_relevance"])
            reasoning = str(data.get("reasoning", ""))
        except (KeyError, TypeError, ValueError):
            return None

        return EvalResult(
            factual_accuracy=fa,
            citation_quality=cq,
            depth_and_relevance=dr,
            overall=(fa + cq + dr) / 3.0,
            reasoning=reasoning,
            model_used=self._client.model,
        )
