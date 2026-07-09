"""
OpenAI adapter for the LLMClient interface.

Near-passthrough since OpenAI format is the canonical internal format.

Differences handled:
  - system string → prepended as {"role": "system", ...} message
  - finish_reason "stop" → "end_turn", "tool_calls" → "tool_use"
  - streaming: accumulates tool_call argument fragments across chunks
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import openai

from bookmind_tutor.llm.base import (
    CompletionResponse,
    LLMAPIError,
    LLMClient,
    TokenUsage,
    ToolCallRequest,
)

logger = logging.getLogger(__name__)

_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


class OpenAIClient(LLMClient):
    """
    Wraps openai.OpenAI and normalizes to/from the canonical format.

    Args:
        model: OpenAI model ID (e.g. "gpt-4o-mini", "gpt-4o").
        api_key: OpenAI API key. If None, reads OPENAI_API_KEY from env.
        base_url: Optional API base URL for compatible providers (Groq, Together, etc.).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._sdk = openai.OpenAI(**kwargs)
        self._last_response: CompletionResponse | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "openai"

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
        oai_messages = self._with_system(messages, system)
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=oai_messages,
        )
        if tools:
            kwargs["tools"] = tools

        try:
            response = self._sdk.chat.completions.create(**kwargs)
        except openai.APIStatusError as exc:
            raise self._wrap_error(exc) from exc
        except openai.APIConnectionError as exc:
            raise LLMAPIError(
                "Cannot connect to OpenAI API. Check your internet connection.",
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
        oai_messages = self._with_system(messages, system)
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=oai_messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        if tools:
            kwargs["tools"] = tools

        accumulated_text: list[str] = []
        # tool_calls accumulator: index → partial call dict
        tool_acc: dict[int, dict] = {}
        finish_reason: str | None = None
        usage_data: dict = {"prompt_tokens": 0, "completion_tokens": 0}

        try:
            for chunk in self._sdk.chat.completions.create(**kwargs):
                if chunk.usage:
                    usage_data = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                    }
                    continue

                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta
                if delta.content:
                    accumulated_text.append(delta.content)
                    yield delta.content

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_acc:
                            tool_acc[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        entry = tool_acc[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        fn = tc_delta.function
                        if fn:
                            if fn.name:
                                entry["function"]["name"] += fn.name
                            if fn.arguments:
                                entry["function"]["arguments"] += fn.arguments

        except openai.APIStatusError as exc:
            raise self._wrap_error(exc) from exc
        except openai.APIConnectionError as exc:
            raise LLMAPIError(
                "Cannot connect to OpenAI API. Check your internet connection.",
                code="connection_error",
            ) from exc

        # Build CompletionResponse after streaming ends
        text = "".join(accumulated_text)
        stop_reason = _FINISH_REASON_MAP.get(finish_reason or "", "end_turn")

        raw_tool_calls = [tool_acc[i] for i in sorted(tool_acc)]
        tool_calls = []
        for tc in raw_tool_calls:
            args_str = tc["function"]["arguments"] or "{}"
            try:
                input_dict = json.loads(args_str)
            except json.JSONDecodeError:
                input_dict = {}
            tool_calls.append(ToolCallRequest(id=tc["id"], name=tc["function"]["name"], input=input_dict))

        raw_message: dict = {"role": "assistant", "content": text or None}
        if raw_tool_calls:
            raw_message["tool_calls"] = raw_tool_calls

        self._last_response = CompletionResponse(
            text=text,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            usage=TokenUsage(
                input_tokens=usage_data["prompt_tokens"],
                output_tokens=usage_data["completion_tokens"],
            ),
            raw_message=raw_message,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _with_system(messages: list[dict], system: str) -> list[dict]:
        """Prepend system message if provided."""
        if not system:
            return list(messages)
        return [{"role": "system", "content": system}] + list(messages)

    def _normalize_response(self, response: Any) -> CompletionResponse:
        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason or "stop"
        stop_reason = _FINISH_REASON_MAP.get(finish_reason, "end_turn")

        text = message.content or ""

        tool_calls: list[ToolCallRequest] = []
        raw_tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                args_str = tc.function.arguments or "{}"
                try:
                    input_dict = json.loads(args_str)
                except json.JSONDecodeError:
                    input_dict = {}
                tool_calls.append(ToolCallRequest(id=tc.id, name=tc.function.name, input=input_dict))
                raw_tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                })

        raw_message: dict = {"role": "assistant", "content": text or None}
        if raw_tool_calls:
            raw_message["tool_calls"] = raw_tool_calls

        usage = response.usage
        return CompletionResponse(
            text=text,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
            ),
            raw_message=raw_message,
        )

    @staticmethod
    def _wrap_error(exc: openai.APIStatusError) -> LLMAPIError:
        status = exc.status_code

        if status == 401:
            return LLMAPIError(
                "Invalid OpenAI API key. Check the OPENAI_API_KEY environment variable.",
                status_code=status,
                code="auth_error",
            )
        if status == 429:
            body = getattr(exc, "body", {}) or {}
            err_code = (body.get("error") or {}).get("code", "")
            if err_code == "insufficient_quota":
                return LLMAPIError(
                    "Your OpenAI quota is exhausted. Add credits at platform.openai.com/account/billing.",
                    status_code=status,
                    code="credit_exhausted",
                )
            return LLMAPIError(
                "OpenAI rate limit reached. Please wait a moment and try again.",
                status_code=status,
                code="rate_limit",
            )

        return LLMAPIError(
            f"OpenAI API error (status {status}). Try again in a moment.",
            status_code=status,
            code="api_error",
        )
