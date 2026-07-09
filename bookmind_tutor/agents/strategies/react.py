"""ReActStrategy: thin wrapper around AgentHarness for uniform interface."""
from __future__ import annotations

import time
from collections.abc import Iterator

from bookmind_tutor.agents.harness import AgentHarness
from bookmind_tutor.agents.memory import ConversationMemory
from bookmind_tutor.agents.models import ReasoningTrace


class ReActStrategy:
    """
    Delegates directly to AgentHarness — the ReAct loop implemented in Week 2.

    Exists so ReAct can be compared with Plan-and-Execute and Reflexion
    using the same interface via StrategyRouter.
    """

    name = "react"

    def __init__(self, harness: AgentHarness) -> None:
        self._harness = harness
        self.last_trace: ReasoningTrace | None = None

    def chat(
        self,
        question: str,
        memory: ConversationMemory | None = None,
    ) -> str:
        t0 = time.monotonic()
        before = self._harness.total_llm_calls
        answer = self._harness.chat(question, memory=memory)
        self.last_trace = ReasoningTrace(
            strategy_name=self.name,
            question=question,
            answer=answer,
            steps=self._harness.last_steps,
            num_llm_calls=self._harness.total_llm_calls - before,
            elapsed_seconds=time.monotonic() - t0,
            retrieved_chunks=self._harness.last_retrieved_chunks,
        )
        return answer

    def stream_chat(
        self,
        question: str,
        memory: ConversationMemory | None = None,
    ) -> Iterator[str]:
        """Stream the final answer token-by-token. Sets last_trace when exhausted."""
        t0 = time.monotonic()
        before = self._harness.total_llm_calls
        parts: list[str] = []

        for token in self._harness.stream_chat(question, memory=memory):
            parts.append(token)
            yield token

        answer = "".join(parts).strip()
        self.last_trace = ReasoningTrace(
            strategy_name=self.name,
            question=question,
            answer=answer,
            steps=self._harness.last_steps,
            num_llm_calls=self._harness.total_llm_calls - before,
            elapsed_seconds=time.monotonic() - t0,
            retrieved_chunks=self._harness.last_retrieved_chunks,
        )
