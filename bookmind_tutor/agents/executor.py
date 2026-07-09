"""
ToolExecutor: dispatches tool calls to registered Tool instances.

Provides retry logic for transient errors and returns structured
ToolCallResult objects so the harness never needs to handle raw exceptions.
"""
from __future__ import annotations

import json
import time

import structlog

from bookmind_tutor.agents.models import ToolCallResult
from bookmind_tutor.agents.tools.base import Tool
from bookmind_tutor.llm.base import ToolCallRequest
from bookmind_tutor.observability.tracing import ToolAttrs, get_tracer

logger = structlog.get_logger()

# Errors worth retrying (transient infrastructure issues)
_RETRYABLE_ERRORS = (ConnectionError, TimeoutError, OSError)


class ToolExecutor:
    """
    Executes tool calls from a CompletionResponse.

    Args:
        tools: List of Tool instances. Names must be unique.
        max_retries: How many times to retry a transient error before giving up.
        retry_delay: Seconds to wait between retries (doubles each attempt).
    """

    def __init__(
        self,
        tools: list[Tool],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self._tools: dict[str, Tool] = {t.spec.name: t for t in tools}
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def execute_all(self, tool_calls: list[ToolCallRequest]) -> list[ToolCallResult]:
        """Execute all tool calls and return one ToolCallResult per call, in order."""
        return [self._execute_one(tc) for tc in tool_calls]

    def _execute_one(self, tc: ToolCallRequest) -> ToolCallResult:
        args_summary = json.dumps(tc.input)[:120]
        logger.info("tool_call", tool=tc.name, args=args_summary)

        if tc.name not in self._tools:
            msg = f"Unknown tool '{tc.name}'. Available: {list(self._tools)}"
            logger.warning("unknown_tool", tool=tc.name)
            return ToolCallResult(tool_use_id=tc.id, content=msg, is_error=True)

        tool = self._tools[tc.name]
        delay = self._retry_delay
        tracer = get_tracer()

        for attempt in range(1, self._max_retries + 1):
            with tracer.start_as_current_span("tool.execute") as span:
                span.set_attribute(ToolAttrs.NAME, tc.name)
                span.set_attribute(ToolAttrs.INPUT_SUMMARY, args_summary[:100])
                try:
                    result_text = tool.execute(**tc.input)
                    span.set_attribute(ToolAttrs.RESULT_CHARS, len(result_text))
                    span.set_attribute(ToolAttrs.IS_ERROR, False)
                    logger.info("tool_ok", tool=tc.name, result_chars=len(result_text))
                    return ToolCallResult(tool_use_id=tc.id, content=result_text)

                except _RETRYABLE_ERRORS as exc:
                    span.set_attribute(ToolAttrs.IS_ERROR, True)
                    if attempt < self._max_retries:
                        logger.warning(
                            "tool_retry",
                            tool=tc.name,
                            attempt=attempt,
                            max_retries=self._max_retries,
                            error=str(exc),
                            retry_in=delay,
                        )
                        time.sleep(delay)
                        delay *= 2
                    else:
                        msg = f"Tool '{tc.name}' failed after {self._max_retries} attempts: {exc}"
                        logger.error("tool_failed", tool=tc.name, attempts=self._max_retries)
                        return ToolCallResult(tool_use_id=tc.id, content=msg, is_error=True)

                except Exception as exc:
                    span.set_attribute(ToolAttrs.IS_ERROR, True)
                    msg = f"Tool '{tc.name}' error: {type(exc).__name__}: {exc}"
                    logger.error("tool_error", tool=tc.name, exc_type=type(exc).__name__, error=str(exc))
                    return ToolCallResult(tool_use_id=tc.id, content=msg, is_error=True)

        return ToolCallResult(tool_use_id=tc.id, content="Unexpected executor state.", is_error=True)
