"""
GuardrailsLayer: orchestrates input and output guards.

Execution order:
1. Guards run in the order they are listed.
2. First BLOCK result short-circuits — remaining guards are skipped.
3. WARN results are logged by the guard itself; the layer continues.
4. If all guards pass, returns GuardResult(passed=True).
"""
from __future__ import annotations

import structlog

from bookmind_tutor.safety.models import GuardContext, GuardResult, Severity

logger = structlog.get_logger()


class GuardrailsLayer:
    def __init__(
        self,
        input_guards: list,
        output_guards: list,
    ) -> None:
        self._input_guards = input_guards
        self._output_guards = output_guards

    def check_input(self, text: str, ctx: GuardContext) -> GuardResult:
        """
        Run all input guards in order.

        Returns the first BLOCK result immediately.
        WARN results are logged by the guard; execution continues.
        """
        for guard in self._input_guards:
            result = guard.check(text, ctx)
            if not result.passed and result.severity == Severity.BLOCK:
                logger.info(
                    "input_blocked",
                    guard=type(guard).__name__,
                    violation=result.violation_type,
                    book_id=ctx.book_id,
                )
                return result
        return GuardResult(passed=True)

    def check_output(self, text: str, ctx: GuardContext) -> GuardResult:
        """
        Run all output guards in order.

        Currently all output guards are WARN-only, so this never blocks.
        Returns GuardResult(passed=True) after running all guards.
        """
        for guard in self._output_guards:
            result = guard.check(text, ctx)
            if not result.passed and result.severity == Severity.BLOCK:
                return result
        return GuardResult(passed=True)
