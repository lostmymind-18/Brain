"""
ToolExecutor: dispatches tool calls to registered Tool instances.

Provides retry logic for transient errors and returns structured
ToolCallResult objects so the harness never needs to handle raw exceptions.
"""
from __future__ import annotations

import json
import logging
import time

from bookmind_tutor.agents.models import ToolCallResult
from bookmind_tutor.agents.tools.base import Tool
from bookmind_tutor.llm.base import ToolCallRequest

logger = logging.getLogger(__name__)

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
        logger.info("Tool call: %s(%s)", tc.name, args_summary)

        if tc.name not in self._tools:
            msg = f"Unknown tool '{tc.name}'. Available: {list(self._tools)}"
            logger.warning(msg)
            return ToolCallResult(tool_use_id=tc.id, content=msg, is_error=True)

        tool = self._tools[tc.name]
        delay = self._retry_delay

        for attempt in range(1, self._max_retries + 1):
            try:
                result_text = tool.execute(**tc.input)
                logger.info("Tool %s → OK (%d chars)", tc.name, len(result_text))
                return ToolCallResult(tool_use_id=tc.id, content=result_text)

            except _RETRYABLE_ERRORS as exc:
                if attempt < self._max_retries:
                    logger.warning(
                        "Tool %s attempt %d/%d failed (%s), retrying in %.1fs",
                        tc.name, attempt, self._max_retries, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    msg = f"Tool '{tc.name}' failed after {self._max_retries} attempts: {exc}"
                    logger.error(msg)
                    return ToolCallResult(tool_use_id=tc.id, content=msg, is_error=True)

            except Exception as exc:
                msg = f"Tool '{tc.name}' error: {type(exc).__name__}: {exc}"
                logger.error(msg)
                return ToolCallResult(tool_use_id=tc.id, content=msg, is_error=True)

        return ToolCallResult(tool_use_id=tc.id, content="Unexpected executor state.", is_error=True)
