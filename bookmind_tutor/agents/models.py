"""Data models for the agent harness."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolSpec:
    """
    Describes a callable tool.

    input_schema must be a valid JSON Schema object (type: "object").
    to_dict() returns OpenAI function-tool format, which all provider
    adapters convert internally to their wire format.
    """
    name: str
    description: str
    input_schema: dict

    def to_dict(self) -> dict:
        """Return the tool definition in OpenAI function-tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class ToolCallResult:
    """
    The result of executing a single tool call.

    is_error=True signals that the tool failed. The model will see content
    as an error message and may retry with different arguments or give up.
    """
    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class ReasoningTrace:
    """
    Records what happened during one chat() call — used to compare strategies.

    steps: high-level description of each phase (e.g. "Plan: 3 sub-tasks",
           "Execute sub-task 1", "Reflect: score=3, retrying").
    num_llm_calls: direct LLM calls made by the strategy itself (not counting
                   calls made inside AgentHarness, which are logged separately).
    elapsed_seconds: wall-clock time for the full chat() call.
    retrieved_chunks: raw text returned by book_search tool calls during this turn,
                      accumulated across all ReAct steps. Used by the eval framework
                      (Week 7) to assess factual grounding, and as context for DPO
                      data generation (Week 10-11).
    """
    strategy_name: str
    question: str
    answer: str
    steps: list[str] = field(default_factory=list)
    num_llm_calls: int = 0
    elapsed_seconds: float = 0.0
    retrieved_chunks: list[str] = field(default_factory=list)
