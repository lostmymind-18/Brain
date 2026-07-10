"""Abstract base class for agent tools."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from bookmind_tutor.agents.models import ToolSpec


@dataclass
class SourceRef:
    """Lightweight pointer to book content a tool actually retrieved.

    Tools return plain strings to the LLM, which loses the structured
    location metadata the UI needs for an honest Sources panel. Retrieval
    tools record one SourceRef per retrieved scope in `Tool.last_sources`
    so the UI can show what the agent really read (instead of running a
    separate search that may not match the agent's retrieval at all).
    """
    chapter: str | None
    section: str | None
    page_range: tuple[int, int]
    subsection: str | None = None  # L2 heading; None for pre-subsection indexes


class Tool(ABC):
    """
    A callable tool that the agent can invoke.

    Subclasses must implement `spec` (describing the tool to Claude)
    and `execute` (the actual implementation).

    Raise any exception from execute() to signal failure.
    ToolExecutor will catch it and return a ToolCallResult(is_error=True).

    Retrieval tools append SourceRef entries to `last_sources` on each
    successful execute. Contract: the caller that runs a conversation turn
    clears `last_sources` before invoking the agent, then reads it after.
    Tools that do not retrieve chunk content (e.g. book_outline) leave it empty.
    """

    # Class-level default; retrieval tools shadow it with an instance list.
    last_sources: list[SourceRef] = []

    @property
    @abstractmethod
    def spec(self) -> ToolSpec: ...

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """
        Run the tool with the provided keyword arguments.

        Returns a string result that will be shown to Claude as the tool output.
        Raise any exception to signal an error.
        """
