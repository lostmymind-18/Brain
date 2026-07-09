"""Abstract base class for agent tools."""
from __future__ import annotations

from abc import ABC, abstractmethod

from bookmind_tutor.agents.models import ToolSpec


class Tool(ABC):
    """
    A callable tool that the agent can invoke.

    Subclasses must implement `spec` (describing the tool to Claude)
    and `execute` (the actual implementation).

    Raise any exception from execute() to signal failure.
    ToolExecutor will catch it and return a ToolCallResult(is_error=True).
    """

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
