"""
Reflexion strategy (Shinn et al. 2023).

After getting an initial answer from AgentHarness, a separate LLM call
evaluates the answer quality on three criteria:
  1. Does it cite specific sources (chapter, page) from the book?
  2. Is it deep enough for the question asked?
  3. Are important aspects missing?

If the score is below the threshold, the strategy retries with the original
question augmented by the reflection feedback. Max one retry to keep API
cost predictable.

Why this helps: ReAct does not self-evaluate. If the first search returns
vague results, Claude may still produce a confident but shallow answer.
Reflexion adds a quality gate that catches this.

Limitation: only works when Claude can accurately judge its own answer.
If the initial answer is confidently wrong, the reflection score may still
be high and the retry never triggers.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

from bookmind_tutor.agents.harness import AgentHarness
from bookmind_tutor.agents.memory import ConversationMemory
from bookmind_tutor.agents.models import ReasoningTrace
from bookmind_tutor.llm.base import LLMClient

logger = logging.getLogger(__name__)

# Retry if reflection score is strictly below this value (scale 1–5)
SCORE_THRESHOLD = 4

_REFLECT_SYSTEM = (
    "You are a critical evaluator of AI-generated answers about book content. "
    "Evaluate the answer strictly and return only JSON."
)

_REFLECT_PROMPT = """\
Original question: {question}

Answer to evaluate:
{answer}

Evaluate on these three criteria:
1. Does it cite specific sources (chapter name and page number) from the book?
2. Is the depth appropriate for the complexity of the question?
3. Are there important aspects of the question left unanswered?

Respond with exactly this JSON (no markdown fences, no extra text):
{{"score": <integer 1-5>, "missing": "<what is missing, or empty string>", "should_retry": <true|false>}}

Set should_retry to true only when score < {threshold}.\
"""


class ReflexionStrategy:
    """
    Adds a self-evaluation loop on top of AgentHarness.

    Args:
        harness: AgentHarness used for initial and retry answers.
        llm_client: LLMClient for the reflection call.
    """

    name = "reflexion"

    def __init__(
        self,
        harness: AgentHarness,
        llm_client: LLMClient,
        model: str = "",  # ignored — model is part of llm_client
    ) -> None:
        self._harness = harness
        self._client = llm_client
        self.last_trace: ReasoningTrace | None = None

    def chat(
        self,
        question: str,
        memory: ConversationMemory | None = None,
    ) -> str:
        t0 = time.monotonic()
        own_memory = memory is None
        if own_memory:
            memory = ConversationMemory()

        memory.add_user(question)
        steps: list[str] = []
        num_llm_calls = 0

        # --- Round 1: initial answer ---
        logger.info("Reflexion: getting initial answer")
        before = self._harness.total_llm_calls
        answer = self._harness.chat(question)  # stateless
        num_llm_calls += self._harness.total_llm_calls - before
        steps.append("Round 1: initial answer from harness")

        # --- Reflect ---
        reflection = self._reflect(question, answer)
        num_llm_calls += 1  # reflection call
        score = reflection.get("score", 5)
        missing = reflection.get("missing", "")
        should_retry = reflection.get("should_retry", False)
        steps.append(f"Reflect: score={score}/5, should_retry={should_retry}, missing={missing[:60]!r}")
        logger.info("Reflexion: score=%d/5, should_retry=%s", score, should_retry)

        # --- Round 2: retry with feedback (at most once) ---
        if should_retry and missing:
            enhanced = (
                f"{question}\n\n"
                f"[Note: a previous attempt was rated {score}/5. "
                f"Please specifically address: {missing}]"
            )
            logger.info("Reflexion: retrying with feedback")
            before = self._harness.total_llm_calls
            answer = self._harness.chat(enhanced)  # stateless
            num_llm_calls += self._harness.total_llm_calls - before
            steps.append("Round 2: retry with reflection feedback")

        if not own_memory:
            memory.add_assistant({"role": "assistant", "content": answer})

        self.last_trace = ReasoningTrace(
            strategy_name=self.name,
            question=question,
            answer=answer,
            steps=steps,
            num_llm_calls=num_llm_calls,
            elapsed_seconds=time.monotonic() - t0,
        )
        return answer

    def stream_chat(
        self,
        question: str,
        memory: ConversationMemory | None = None,
    ) -> Iterator[str]:
        """
        Run initial answer + reflection (blocking), then stream the final answer.

        If no retry is needed: stream the initial answer.
        If retry is needed: stream the retry via harness.stream_chat().
        """
        t0 = time.monotonic()
        own_memory = memory is None
        if own_memory:
            memory = ConversationMemory()

        memory.add_user(question)
        steps: list[str] = []
        num_llm_calls = 0

        before = self._harness.total_llm_calls
        initial_answer = self._harness.chat(question)
        num_llm_calls += self._harness.total_llm_calls - before
        steps.append("Round 1: initial answer (blocking)")

        reflection = self._reflect(question, initial_answer)
        num_llm_calls += 1
        score = reflection.get("score", 5)
        missing = reflection.get("missing", "")
        should_retry = reflection.get("should_retry", False)
        steps.append(f"Reflect: score={score}/5, retry={should_retry}")

        parts: list[str] = []
        if should_retry and missing:
            enhanced = (
                f"{question}\n\n"
                f"[Note: a previous attempt was rated {score}/5. "
                f"Please specifically address: {missing}]"
            )
            steps.append("Round 2: streaming retry with feedback")
            before = self._harness.total_llm_calls
            for token in self._harness.stream_chat(enhanced):
                parts.append(token)
                yield token
            num_llm_calls += self._harness.total_llm_calls - before
        else:
            # No retry needed — stream the initial answer
            steps.append("No retry — streaming initial answer")
            yield initial_answer
            parts = [initial_answer]

        answer = "".join(parts).strip()
        if not own_memory:
            memory.add_assistant({"role": "assistant", "content": answer})

        self.last_trace = ReasoningTrace(
            strategy_name=self.name,
            question=question,
            answer=answer,
            steps=steps,
            num_llm_calls=num_llm_calls,
            elapsed_seconds=time.monotonic() - t0,
        )

    def _reflect(self, question: str, answer: str) -> dict:
        """
        Evaluate answer quality and return reflection dict.

        Returns {"score": int, "missing": str, "should_retry": bool}.
        Falls back to {"score": 5, "should_retry": False} on any parse error
        so that a broken reflection never blocks the caller from getting an answer.
        """
        prompt = _REFLECT_PROMPT.format(
            question=question,
            answer=answer,
            threshold=SCORE_THRESHOLD,
        )
        try:
            response = self._client.complete(
                messages=[{"role": "user", "content": prompt}],
                system=_REFLECT_SYSTEM,
                max_tokens=256,
            )
            raw = response.text
            logger.debug("Reflection raw: %s", raw[:200])

            # Strip markdown fences if present
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("No JSON object in reflection response")
            result = json.loads(raw[start : end + 1])
            return result
        except Exception as exc:
            logger.warning("Reflection parsing failed (%s), skipping retry", exc)
            return {"score": 5, "missing": "", "should_retry": False}
