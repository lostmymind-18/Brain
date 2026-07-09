"""Output guards: check LLM responses before they reach the user."""
from __future__ import annotations

import structlog

from bookmind_tutor.safety.models import GuardContext, GuardResult, Severity, ViolationType

logger = structlog.get_logger()


class OutputLengthGuard:
    """
    Warn when a response is unusually long.

    Severity is WARN (not BLOCK) — a long but correct answer is better than
    truncating valid content. The log helps identify prompts that cause
    runaway generation, which is usually a system prompt issue to fix upstream.
    """

    def __init__(self, max_chars: int = 8000) -> None:
        self._max = max_chars

    def check(self, text: str, ctx: GuardContext) -> GuardResult:
        if len(text) <= self._max:
            return GuardResult(passed=True)
        logger.warning(
            "guardrail_warn",
            violation=ViolationType.OUTPUT_TOO_LONG,
            output_chars=len(text),
            max_chars=self._max,
            book_id=ctx.book_id,
        )
        return GuardResult(
            passed=False,
            violation_type=ViolationType.OUTPUT_TOO_LONG,
            severity=Severity.WARN,
            user_message="",
            internal_detail=f"output_chars={len(text)} max={self._max}",
        )
