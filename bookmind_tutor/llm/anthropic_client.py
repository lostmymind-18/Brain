"""
Anthropic adapter for the LLMClient interface.

Converts between OpenAI format (internal canonical) and Anthropic format (wire).

Outbound (OpenAI → Anthropic):
  - user "content" string  → content block list with type="text"
  - role="tool" messages   → role="user" with type="tool_result" blocks (merged)
  - role="assistant" with tool_calls → role="assistant" with type="tool_use" blocks
  - tool defs: function.parameters → input_schema; unwrap "function" wrapper

Inbound (Anthropic → OpenAI):
  - TextBlock                → CompletionResponse.text
  - ToolUseBlock             → CompletionResponse.tool_calls + raw_message.tool_calls
  - stop_reason passthrough  → "end_turn" | "tool_use" (same strings)
  - usage                    → TokenUsage
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import anthropic

from bookmind_tutor.llm.base import (
    CompletionResponse,
    LLMAPIError,
    LLMClient,
    TokenUsage,
    ToolCallRequest,
)

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """
    Wraps anthropic.Anthropic and normalizes to/from OpenAI format.

    Args:
        model: Anthropic model ID.
        api_key: Anthropic API key. If None, reads ANTHROPIC_API_KEY from env.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._sdk = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._last_response: CompletionResponse | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def last_response(self) -> CompletionResponse | None:
        return self._last_response

    def _complete(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResponse:
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=self._to_anthropic_messages(messages),
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(tools)

        try:
            response = self._sdk.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            raise self._wrap_error(exc) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMAPIError(
                "Cannot connect to Anthropic API. Check your internet connection.",
                code="connection_error",
            ) from exc

        return self._normalize_response(response)

    def _stream(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> Iterator[str]:
        self._last_response = None
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=self._to_anthropic_messages(messages),
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(tools)

        try:
            with self._sdk.messages.stream(**kwargs) as stream:
                for token in stream.text_stream:
                    yield token
                final = stream.get_final_message()
        except anthropic.APIStatusError as exc:
            raise self._wrap_error(exc) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMAPIError(
                "Cannot connect to Anthropic API. Check your internet connection.",
                code="connection_error",
            ) from exc

        self._last_response = self._normalize_response(final)

    # ------------------------------------------------------------------
    # Outbound conversion: OpenAI → Anthropic
    # ------------------------------------------------------------------

    def _to_anthropic_messages(self, messages: list[dict]) -> list[dict]:
        """
        Convert OpenAI-format messages to Anthropic format.

        Key differences:
        - OpenAI role="tool" messages → merged into Anthropic role="user" with
          tool_result blocks (Anthropic requires tool results inside a user turn).
        - Multiple consecutive tool results → single user message with multiple blocks.
        - OpenAI assistant tool_calls → Anthropic tool_use content blocks.
        """
        result = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")

            if role == "user":
                content = msg["content"]
                if isinstance(content, str):
                    result.append({"role": "user", "content": [{"type": "text", "text": content}]})
                else:
                    result.append({"role": "user", "content": content})
                i += 1

            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    blocks = []
                    if msg.get("content"):
                        blocks.append({"type": "text", "text": msg["content"]})
                    for tc in tool_calls:
                        fn = tc["function"]
                        args = fn.get("arguments") or "{}"
                        try:
                            input_dict = json.loads(args)
                        except json.JSONDecodeError:
                            input_dict = {}
                        blocks.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": fn["name"],
                            "input": input_dict,
                        })
                    result.append({"role": "assistant", "content": blocks})
                else:
                    text = msg.get("content") or ""
                    result.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
                i += 1

            elif role == "tool":
                # Collect all consecutive tool results → single user message
                blocks = []
                while i < len(messages) and messages[i].get("role") == "tool":
                    t = messages[i]
                    blocks.append({
                        "type": "tool_result",
                        "tool_use_id": t["tool_call_id"],
                        "content": t["content"],
                    })
                    i += 1
                result.append({"role": "user", "content": blocks})

            else:
                i += 1  # skip system or unknown (system is passed separately)

        return result

    @staticmethod
    def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
        """Convert OpenAI function-tool defs to Anthropic tool defs."""
        result = []
        for t in tools:
            fn = t["function"]
            result.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    # ------------------------------------------------------------------
    # Inbound conversion: Anthropic → CompletionResponse
    # ------------------------------------------------------------------

    def _normalize_response(self, response: Any) -> CompletionResponse:
        """Convert an Anthropic Messages response to the canonical CompletionResponse."""
        stop_reason = response.stop_reason  # "end_turn" or "tool_use" — same strings

        text_parts = []
        tool_calls: list[ToolCallRequest] = []
        raw_tool_calls = []

        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "tool_use":
                tool_calls.append(ToolCallRequest(id=block.id, name=block.name, input=block.input))
                raw_tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

        text = "\n".join(text_parts).strip()

        raw_message: dict = {"role": "assistant", "content": text or None}
        if raw_tool_calls:
            raw_message["tool_calls"] = raw_tool_calls

        return CompletionResponse(
            text=text,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            raw_message=raw_message,
        )

    # ------------------------------------------------------------------
    # Error mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_error(exc: anthropic.APIStatusError) -> LLMAPIError:
        body = getattr(exc, "body", {}) or {}
        api_msg = body.get("error", {}).get("message", "")

        if isinstance(exc, anthropic.BadRequestError):
            if "credit balance" in api_msg.lower():
                return LLMAPIError(
                    "Your API credit balance is too low. "
                    "Top up at console.anthropic.com/settings/billing to continue.",
                    status_code=exc.status_code,
                    code="credit_exhausted",
                )
            return LLMAPIError(
                f"Bad request: {api_msg or exc}",
                status_code=exc.status_code,
                code="api_error",
            )

        if isinstance(exc, anthropic.AuthenticationError):
            return LLMAPIError(
                "Invalid Anthropic API key. Check the ANTHROPIC_API_KEY environment variable.",
                status_code=exc.status_code,
                code="auth_error",
            )

        if isinstance(exc, anthropic.RateLimitError):
            return LLMAPIError(
                "Anthropic rate limit reached. Please wait a moment and try again.",
                status_code=exc.status_code,
                code="rate_limit",
            )

        return LLMAPIError(
            f"Anthropic API error (status {exc.status_code}). Try again in a moment.",
            status_code=exc.status_code,
            code="api_error",
        )
