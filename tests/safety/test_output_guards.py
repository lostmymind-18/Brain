"""Unit tests for output guards."""
from __future__ import annotations

from bookmind_tutor.safety.models import GuardContext, Severity, ViolationType
from bookmind_tutor.safety.output_guards import OutputLengthGuard


def _ctx() -> GuardContext:
    return GuardContext(book_name="Test Book", book_id="test")


class TestOutputLengthGuard:
    def test_short_output_passes(self):
        result = OutputLengthGuard().check("Short answer.", _ctx())
        assert result.passed

    def test_exact_limit_passes(self):
        text = "a" * 8000
        result = OutputLengthGuard(max_chars=8000).check(text, _ctx())
        assert result.passed

    def test_over_limit_warns(self):
        text = "a" * 8001
        result = OutputLengthGuard(max_chars=8000).check(text, _ctx())
        assert not result.passed
        assert result.severity == Severity.WARN
        assert result.violation_type == ViolationType.OUTPUT_TOO_LONG

    def test_custom_limit(self):
        result = OutputLengthGuard(max_chars=100).check("x" * 101, _ctx())
        assert not result.passed

    def test_internal_detail_contains_char_counts(self):
        text = "a" * 200
        result = OutputLengthGuard(max_chars=100).check(text, _ctx())
        assert "200" in result.internal_detail
