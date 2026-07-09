"""Unit tests for GuardrailsLayer orchestration."""
from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from bookmind_tutor.safety.layer import GuardrailsLayer
from bookmind_tutor.safety.models import GuardContext, GuardResult, Severity, ViolationType


def _ctx() -> GuardContext:
    return GuardContext(book_name="Test Book", book_id="test")


def _pass() -> GuardResult:
    return GuardResult(passed=True)


def _block(vtype=ViolationType.INPUT_TOO_LONG) -> GuardResult:
    return GuardResult(
        passed=False,
        violation_type=vtype,
        severity=Severity.BLOCK,
        user_message="blocked",
    )


def _warn() -> GuardResult:
    return GuardResult(
        passed=False,
        violation_type=ViolationType.OFF_TOPIC,
        severity=Severity.WARN,
        user_message="",
    )


def _mock_guard(result: GuardResult) -> MagicMock:
    g = MagicMock()
    g.check.return_value = result
    return g


class TestGuardrailsLayerInput:
    def test_all_pass_returns_passed(self):
        layer = GuardrailsLayer(
            input_guards=[_mock_guard(_pass()), _mock_guard(_pass())],
            output_guards=[],
        )
        result = layer.check_input("hello", _ctx())
        assert result.passed

    def test_first_block_short_circuits(self):
        block_guard = _mock_guard(_block())
        second_guard = _mock_guard(_pass())
        layer = GuardrailsLayer(input_guards=[block_guard, second_guard], output_guards=[])
        result = layer.check_input("hello", _ctx())
        assert not result.passed
        assert result.severity == Severity.BLOCK
        # Second guard should not have been called
        second_guard.check.assert_not_called()

    def test_warn_does_not_short_circuit(self):
        warn_guard = _mock_guard(_warn())
        pass_guard = _mock_guard(_pass())
        layer = GuardrailsLayer(input_guards=[warn_guard, pass_guard], output_guards=[])
        result = layer.check_input("hello", _ctx())
        # WARN alone -> overall passed=True (layer continues to next guard which passed)
        assert result.passed
        pass_guard.check.assert_called_once()

    def test_no_input_guards_passes(self):
        layer = GuardrailsLayer(input_guards=[], output_guards=[])
        result = layer.check_input("anything", _ctx())
        assert result.passed

    def test_context_forwarded_to_guards(self):
        ctx = _ctx()
        guard = _mock_guard(_pass())
        layer = GuardrailsLayer(input_guards=[guard], output_guards=[])
        layer.check_input("question", ctx)
        guard.check.assert_called_once_with("question", ctx)


class TestGuardrailsLayerOutput:
    def test_all_output_pass_returns_passed(self):
        layer = GuardrailsLayer(input_guards=[], output_guards=[_mock_guard(_pass())])
        result = layer.check_output("answer", _ctx())
        assert result.passed

    def test_warn_output_guard_continues_and_returns_passed(self):
        warn_guard = _mock_guard(_warn())
        layer = GuardrailsLayer(input_guards=[], output_guards=[warn_guard])
        result = layer.check_output("long answer...", _ctx())
        # WARN-only output guards should not block
        assert result.passed

    def test_no_output_guards_passes(self):
        layer = GuardrailsLayer(input_guards=[], output_guards=[])
        result = layer.check_output("anything", _ctx())
        assert result.passed
