"""Base protocol for all reasoning strategies."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from bookmind_tutor.agents.memory import ConversationMemory
from bookmind_tutor.agents.models import ReasoningTrace


@runtime_checkable
class ReasoningStrategy(Protocol):
    """
    Common interface for ReAct, Plan-and-Execute, Reflexion, etc.

    All strategies expose the same chat() signature so StrategyRouter
    and the demo script can call them interchangeably.

    last_trace is populated after each chat() call and holds the
    reasoning trace for that turn (useful for comparison/debugging).
    """

    name: str
    last_trace: ReasoningTrace | None

    def chat(
        self,
        question: str,
        memory: ConversationMemory | None = None,
    ) -> str: ...
