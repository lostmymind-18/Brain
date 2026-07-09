"""
Provider-agnostic LLM interface.

All messages flow through in OpenAI format (the de facto industry standard,
also used by Groq, Together, Mistral, ...). Provider-specific adapters
convert at the API call boundary.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class ToolCallRequest:
    """A single tool invocation requested by the model."""
    id: str
    name: str
    input: dict


@dataclass
class CompletionResponse:
    """Normalized response from any LLM provider."""
    text: str
    stop_reason: str          # "end_turn" | "tool_use" | "max_tokens"
    tool_calls: list[ToolCallRequest]
    usage: TokenUsage
    # Raw assistant message in OpenAI format — pass to ConversationMemory.add_assistant()
    raw_message: dict


class LLMAPIError(Exception):
    """
    Provider-agnostic API error.

    Adapters catch provider-specific exceptions and re-raise as LLMAPIError
    so callers never import provider packages for error handling.

    code values (best-effort, not exhaustive):
      "auth_error"         - invalid API key
      "credit_exhausted"   - account balance too low
      "rate_limit"         - too many requests
      "connection_error"   - network failure
      "api_error"          - other server-side error
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class LLMClient(ABC):
    """
    Abstract LLM client.

    All messages use OpenAI format:
      - user:      {"role": "user", "content": "..."}
      - assistant: {"role": "assistant", "content": "..."} or
                   {"role": "assistant", "content": None, "tool_calls": [...]}
      - tool:      {"role": "tool", "tool_call_id": "...", "content": "..."}

    Tool definitions use OpenAI function format:
      {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

    Adapters translate to/from the provider's native format internally.
    """

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResponse: ...

    @abstractmethod
    def stream(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> Iterator[str]:
        """
        Yield text tokens as they arrive.

        After the iterator is exhausted, call last_response to get the full
        CompletionResponse (stop_reason, tool_calls, usage, raw_message).
        """
        ...

    @property
    @abstractmethod
    def last_response(self) -> CompletionResponse | None:
        """CompletionResponse from the most recent stream() call, set after exhaustion."""
        ...
