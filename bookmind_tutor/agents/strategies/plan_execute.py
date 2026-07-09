"""
Plan-and-Execute strategy (Wang et al. 2023).

Unlike ReAct which decides the next action after each observation,
Plan-and-Execute separates planning from execution:

  Phase 1 — Plan:   ask Claude to decompose the question into sub-tasks
                     (no tools, structured JSON output)
  Phase 2 — Execute: run each sub-task through AgentHarness (has tool access)
  Phase 3 — Synthesize: ask Claude to merge all results into one final answer

Why this helps: ReAct can "forget" parts of a complex question as the
tool-call loop stretches on. Enumerating sub-tasks up front ensures every
requirement is addressed before synthesis.

Fallback: if the planning step returns unparseable JSON, the strategy falls
back to treating the original question as a single sub-task (equivalent to
one ReAct pass).
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from bookmind_tutor.agents.harness import AgentHarness
from bookmind_tutor.agents.memory import ConversationMemory
from bookmind_tutor.agents.models import ReasoningTrace
from bookmind_tutor.llm.base import LLMClient

logger = logging.getLogger(__name__)

MAX_SUBTASKS = 5

_PLAN_SYSTEM = (
    "You are a planning assistant. "
    "Your only job is to decompose a user question into a list of concrete sub-tasks "
    "that can each be answered independently by searching a book. "
    "Return ONLY a JSON array of strings, nothing else. "
    f"Use at most {MAX_SUBTASKS} sub-tasks."
)

_PLAN_PROMPT = (
    "Decompose this question into sub-tasks:\n\n{question}"
)

_SYNTHESIZE_SYSTEM = (
    "You are a synthesis assistant. "
    "Given a user's original question and the results of several sub-tasks, "
    "write a single coherent, well-structured answer. "
    "Preserve all citations (chapter names, page numbers) from the sub-task results."
)


class PlanExecuteStrategy:
    """
    Implements Plan-and-Execute reasoning over AgentHarness.

    Args:
        harness: AgentHarness instance used for sub-task execution.
        llm_client: LLMClient for planning and synthesis calls
            (these don't use tools, so they bypass the harness).
    """

    name = "plan_execute"

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

        # --- Phase 1: Plan ---
        sub_tasks = self._plan(question)
        num_llm_calls += 1  # plan call
        steps.append(f"Plan: {len(sub_tasks)} sub-tasks: {sub_tasks}")
        logger.info("Plan-and-Execute: %d sub-tasks for question: %s", len(sub_tasks), question[:80])

        # --- Phase 2: Execute sub-tasks in parallel (each is stateless) ---
        # Sub-tasks are independent — no shared memory, no ordering dependency.
        # ThreadPoolExecutor handles concurrent Anthropic API calls safely because
        # AgentHarness.chat() with memory=None creates fresh state per call.
        # harness._call_lock ensures the counter is thread-safe across parallel calls.
        before_harness = self._harness.total_llm_calls
        results: list[tuple[str, str]] = [("", "")] * len(sub_tasks)
        with ThreadPoolExecutor(max_workers=len(sub_tasks)) as executor:
            future_to_index = {
                executor.submit(self._harness.chat, task): (i, task)
                for i, task in enumerate(sub_tasks)
            }
            for future in as_completed(future_to_index):
                idx, task = future_to_index[future]
                result = future.result()
                results[idx] = (task, result)
                logger.info("Completed sub-task %d/%d: %s", idx + 1, len(sub_tasks), task[:80])

        num_llm_calls += self._harness.total_llm_calls - before_harness
        for i, (task, _) in enumerate(results, 1):
            steps.append(f"Execute sub-task {i} (parallel): {task[:60]}")

        # --- Phase 3: Synthesize ---
        answer = self._synthesize(question, results)
        num_llm_calls += 1  # synthesis call
        steps.append("Synthesize: merged all sub-task results")
        logger.info("Plan-and-Execute: synthesis complete (%d chars)", len(answer))

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

    def _plan(self, question: str) -> list[str]:
        """
        Ask the LLM to decompose question into sub-tasks.

        Returns a list of sub-task strings. Falls back to [question] if the
        LLM returns invalid JSON — treating the whole question as one task
        is equivalent to a single ReAct pass and always safe.
        """
        response = self._client.complete(
            messages=[{"role": "user", "content": _PLAN_PROMPT.format(question=question)}],
            system=_PLAN_SYSTEM,
            max_tokens=512,
        )
        raw = response.text
        logger.debug("Plan raw response: %s", raw[:200])

        try:
            # LLMs sometimes wrap JSON in markdown fences — strip to bare array
            start = raw.find("[")
            end = raw.rfind("]")
            if start == -1 or end == -1:
                raise ValueError("No JSON array found in plan response")
            tasks = json.loads(raw[start : end + 1])
            if not isinstance(tasks, list) or not tasks:
                raise ValueError("Plan response is not a non-empty list")
            # Enforce type and cap
            tasks = [str(t) for t in tasks if t][:MAX_SUBTASKS]
            return tasks
        except Exception as exc:
            logger.warning("Plan parsing failed (%s), falling back to single task", exc)
            return [question]

    def stream_chat(
        self,
        question: str,
        memory: ConversationMemory | None = None,
    ) -> Iterator[str]:
        """
        Run plan + parallel execute (blocking), then stream the synthesis step.

        The first two phases are unchanged. Only the final synthesis LLM call
        streams tokens to the caller.
        """
        t0 = time.monotonic()
        own_memory = memory is None
        if own_memory:
            memory = ConversationMemory()

        memory.add_user(question)
        steps: list[str] = []
        num_llm_calls = 0

        sub_tasks = self._plan(question)
        num_llm_calls += 1
        steps.append(f"Plan: {len(sub_tasks)} sub-tasks")

        before_harness = self._harness.total_llm_calls
        results: list[tuple[str, str]] = [("", "")] * len(sub_tasks)
        with ThreadPoolExecutor(max_workers=len(sub_tasks)) as executor:
            future_to_index = {
                executor.submit(self._harness.chat, task): (i, task)
                for i, task in enumerate(sub_tasks)
            }
            for future in as_completed(future_to_index):
                idx, task = future_to_index[future]
                results[idx] = (task, future.result())
        num_llm_calls += self._harness.total_llm_calls - before_harness
        steps.append(f"Execute {len(sub_tasks)} sub-tasks (parallel)")

        # Stream synthesis
        parts: list[str] = []
        context = self._build_synthesis_context(question, results)
        for token in self._client.stream(
            messages=[{"role": "user", "content": context}],
            system=_SYNTHESIZE_SYSTEM,
            max_tokens=2048,
        ):
            parts.append(token)
            yield token
        num_llm_calls += 1
        steps.append("Synthesize: streamed")

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

    def _build_synthesis_context(self, question: str, results: list[tuple[str, str]]) -> str:
        parts = [f"Original question: {question}\n"]
        for i, (task, result) in enumerate(results, 1):
            parts.append(f"--- Sub-task {i}: {task} ---\n{result}")
        return "\n\n".join(parts)

    def _synthesize(self, question: str, results: list[tuple[str, str]]) -> str:
        """Merge all sub-task results into one coherent answer."""
        context = self._build_synthesis_context(question, results)
        response = self._client.complete(
            messages=[{"role": "user", "content": context}],
            system=_SYNTHESIZE_SYSTEM,
            max_tokens=2048,
        )
        return response.text
