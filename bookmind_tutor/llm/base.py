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

    Instrumentation — Template Method pattern:
      Subclasses implement _complete() and _stream(). The public complete()
      and stream() methods in this base class wrap them with OpenTelemetry
      spans following the GenAI Semantic Conventions. This way every provider
      gets traces for free without repeating instrumentation code.
    """

    @property
    @abstractmethod
    def model(self) -> str: ...

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider name: "anthropic" | "openai" (used in OTel spans)."""
        ...

    @abstractmethod
    def _complete(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResponse:
        """Provider-specific implementation of a blocking completion."""
        ...

    @abstractmethod
    def _stream(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> Iterator[str]:
        """
        Provider-specific streaming implementation.

        Must set self._last_response (or equivalent) before the generator
        is exhausted so that last_response returns the full CompletionResponse.
        """
        ...

    @property
    @abstractmethod
    def last_response(self) -> CompletionResponse | None:
        """CompletionResponse from the most recent stream() call, set after exhaustion."""
        ...

    # ------------------------------------------------------------------
    # Public API — wraps _complete/_stream with OTel instrumentation
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResponse:
        """Blocking completion with OTel span + Langfuse generation."""
        from bookmind_tutor.observability.tracing import GenAIAttrs, get_tracer, set_error
        from bookmind_tutor.observability.langfuse_tracing import langfuse_observation
        lf_input = ([{"role": "system", "content": system}] + list(messages)) if system else list(messages)
        tracer = get_tracer()
        with tracer.start_as_current_span("gen_ai.complete") as span:
            span.set_attribute(GenAIAttrs.SYSTEM, self.provider)
            span.set_attribute(GenAIAttrs.REQUEST_MODEL, self.model)
            span.set_attribute(GenAIAttrs.OPERATION, "complete")
            with langfuse_observation(
                "llm.complete", as_type="generation",
                model=self.model, input=lf_input,
            ) as lf_gen:
                try:
                    response = self._complete(messages, system, tools, max_tokens)
                    span.set_attribute(GenAIAttrs.INPUT_TOKENS, response.usage.input_tokens)
                    span.set_attribute(GenAIAttrs.OUTPUT_TOKENS, response.usage.output_tokens)
                    span.set_attribute(GenAIAttrs.STOP_REASON, response.stop_reason)
                    if lf_gen is not None:
                        lf_gen.update(
                            output=response.text or response.raw_message,
                            usage_details={
                                "input": response.usage.input_tokens,
                                "output": response.usage.output_tokens,
                            },
                        )
                    return response
                except Exception as exc:
                    set_error(span, exc)
                    if lf_gen is not None:
                        lf_gen.update(level="ERROR", status_message=str(exc))
                    raise

    def stream(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> Iterator[str]:
        """
        Yield text tokens as they arrive, wrapped in an OTel span + Langfuse generation.

        After the iterator is exhausted, last_response holds the full
        CompletionResponse (stop_reason, tool_calls, usage, raw_message).
        """
        from bookmind_tutor.observability.tracing import GenAIAttrs, get_tracer, set_error
        from bookmind_tutor.observability.langfuse_tracing import langfuse_observation
        lf_input = ([{"role": "system", "content": system}] + list(messages)) if system else list(messages)
        tracer = get_tracer()
        with tracer.start_as_current_span("gen_ai.stream") as span:
            span.set_attribute(GenAIAttrs.SYSTEM, self.provider)
            span.set_attribute(GenAIAttrs.REQUEST_MODEL, self.model)
            span.set_attribute(GenAIAttrs.OPERATION, "stream")
            with langfuse_observation(
                "llm.stream", as_type="generation",
                model=self.model, input=lf_input,
            ) as lf_gen:
                try:
                    yield from self._stream(messages, system, tools, max_tokens)
                    if self.last_response:
                        span.set_attribute(GenAIAttrs.INPUT_TOKENS, self.last_response.usage.input_tokens)
                        span.set_attribute(GenAIAttrs.OUTPUT_TOKENS, self.last_response.usage.output_tokens)
                        span.set_attribute(GenAIAttrs.STOP_REASON, self.last_response.stop_reason)
                        if lf_gen is not None:
                            lf_gen.update(
                                output=self.last_response.text or self.last_response.raw_message,
                                usage_details={
                                    "input": self.last_response.usage.input_tokens,
                                    "output": self.last_response.usage.output_tokens,
                                },
                            )
                except Exception as exc:
                    set_error(span, exc)
                    if lf_gen is not None:
                        lf_gen.update(level="ERROR", status_message=str(exc))
                    raise
