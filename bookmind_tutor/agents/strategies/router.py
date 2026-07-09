"""
StrategyRouter: selects a reasoning strategy based on question characteristics.

Current implementation uses keyword heuristics — fast, zero-cost, but limited.
The routing table is intentionally transparent so the logic is easy to inspect,
override, and extend.

Known limitations (documented here, not TODO'd away):
- "compare apples and oranges" → Plan-and-Execute even if it's a trivial question.
- "explain in depth" at the end of a long question may be missed if other keywords
  appear earlier (the check is order-independent: any keyword match wins).
- No LLM-based classification yet. If heuristics cause too many misroutes after
  live testing, replace select() with a small LLM call that classifies intent.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

from bookmind_tutor.agents.memory import ConversationMemory
from bookmind_tutor.agents.models import ReasoningTrace
from bookmind_tutor.agents.strategies.plan_execute import PlanExecuteStrategy
from bookmind_tutor.agents.strategies.react import ReActStrategy
from bookmind_tutor.agents.strategies.reflexion import ReflexionStrategy

logger = logging.getLogger(__name__)

# Multi-part / comparison questions → enumerate sub-tasks upfront
_PLAN_KEYWORDS: frozenset[str] = frozenset({
    "compare", "comparison", "difference", "differences",
    "vs", "versus", "contrast", "both", "how do they differ",
    "differ", "differs",
})

# Depth-sensitive questions → self-evaluate and possibly retry
_REFLEXION_KEYWORDS: frozenset[str] = frozenset({
    "in depth", "in detail", "detailed", "comprehensively",
    "thorough", "thoroughly", "elaborate", "explain fully",
})


class StrategyRouter:
    """
    Routes each question to the most appropriate reasoning strategy.

    Args:
        react: ReActStrategy instance.
        plan_execute: PlanExecuteStrategy instance.
        reflexion: ReflexionStrategy instance.
    """

    name = "router"

    def __init__(
        self,
        react: ReActStrategy,
        plan_execute: PlanExecuteStrategy,
        reflexion: ReflexionStrategy,
    ) -> None:
        self._react = react
        self._plan_execute = plan_execute
        self._reflexion = reflexion
        self.last_trace: ReasoningTrace | None = None

    def select(self, question: str) -> ReActStrategy | PlanExecuteStrategy | ReflexionStrategy:
        """Return the strategy that best fits this question."""
        lower = question.lower()
        if any(kw in lower for kw in _PLAN_KEYWORDS):
            return self._plan_execute
        if any(kw in lower for kw in _REFLEXION_KEYWORDS):
            return self._reflexion
        return self._react

    def chat(
        self,
        question: str,
        memory: ConversationMemory | None = None,
        force_strategy: str | None = None,
    ) -> str:
        """
        Route and answer a question.

        Args:
            force_strategy: If set, bypasses keyword routing. Accepted values:
                "react", "plan-execute", "reflexion". Useful for manual overrides
                in the UI without changing the auto-routing logic.
        """
        strategy = self._resolve_strategy(question, force_strategy)
        logger.info(
            "Router selected: %s%s for question: %s",
            strategy.name,
            " (forced)" if force_strategy else "",
            question[:80],
        )
        answer = strategy.chat(question, memory=memory)
        self.last_trace = strategy.last_trace
        return answer

    def stream_chat(
        self,
        question: str,
        memory: ConversationMemory | None = None,
        force_strategy: str | None = None,
    ) -> Iterator[str]:
        """Route and stream an answer. Sets last_trace when generator is exhausted."""
        strategy = self._resolve_strategy(question, force_strategy)
        logger.info(
            "Router selected (streaming): %s%s",
            strategy.name,
            " (forced)" if force_strategy else "",
        )
        yield from strategy.stream_chat(question, memory=memory)
        self.last_trace = strategy.last_trace

    def _resolve_strategy(
        self, question: str, force: str | None
    ) -> ReActStrategy | PlanExecuteStrategy | ReflexionStrategy:
        """Return the strategy indicated by `force`, falling back to auto-select."""
        if force == "react":
            return self._react
        if force == "plan-execute":
            return self._plan_execute
        if force == "reflexion":
            return self._reflexion
        return self.select(question)
