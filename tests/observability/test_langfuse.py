"""
Unit tests for Langfuse integration.
All tests mock the Langfuse SDK — no real API calls, no account needed.
"""
from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from bookmind_tutor.observability import langfuse_tracing


@pytest.fixture(autouse=True)
def reset_langfuse():
    """Ensure each test starts with a clean Langfuse state."""
    langfuse_tracing.reset_for_testing()
    yield
    langfuse_tracing.reset_for_testing()


class TestConfigureLangfuse:
    def test_disabled_when_no_secret_key(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        langfuse_tracing.configure_langfuse()
        assert langfuse_tracing.get_langfuse() is None

    def test_enabled_when_keys_set(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        with patch("langfuse.Langfuse") as mock_cls:
            mock_cls.return_value = MagicMock()
            langfuse_tracing.configure_langfuse()
        assert langfuse_tracing.get_langfuse() is not None

    def test_idempotent(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        with patch("langfuse.Langfuse") as mock_cls:
            mock_cls.return_value = MagicMock()
            langfuse_tracing.configure_langfuse()
            langfuse_tracing.configure_langfuse()  # second call is no-op
            assert mock_cls.call_count == 1

    def test_custom_host(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
        with patch("langfuse.Langfuse") as mock_cls:
            mock_cls.return_value = MagicMock()
            langfuse_tracing.configure_langfuse()
            _, kwargs = mock_cls.call_args
            assert kwargs["host"] == "http://localhost:3000"


class TestLangfuseObservation:
    def test_returns_nullcontext_when_disabled(self):
        # Langfuse not configured
        cm = langfuse_tracing.langfuse_observation("test", as_type="span")
        assert isinstance(cm, type(nullcontext()))

    def test_context_var_is_none_when_disabled(self):
        with langfuse_tracing.langfuse_observation("qa.turn", as_type="agent") as obs:
            assert obs is None

    def test_calls_start_as_current_observation_when_enabled(self, monkeypatch):
        mock_lf = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=MagicMock())
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_lf.start_as_current_observation.return_value = mock_cm
        langfuse_tracing._langfuse = mock_lf

        with langfuse_tracing.langfuse_observation("llm.stream", as_type="generation", model="gpt-4o-mini"):
            pass

        mock_lf.start_as_current_observation.assert_called_once_with(
            name="llm.stream", as_type="generation", model="gpt-4o-mini"
        )


class TestLangfuseFlush:
    def test_flush_no_op_when_disabled(self):
        # Should not raise even when Langfuse is not configured
        langfuse_tracing.langfuse_flush()

    def test_flush_calls_langfuse_when_enabled(self):
        mock_lf = MagicMock()
        langfuse_tracing._langfuse = mock_lf
        langfuse_tracing.langfuse_flush()
        mock_lf.flush.assert_called_once()


class TestIntegrationWithLLMClient:
    """Verify that LLMClient.complete() calls Langfuse when enabled."""

    def test_complete_creates_langfuse_generation(self, monkeypatch):
        from bookmind_tutor.llm.base import CompletionResponse, TokenUsage

        # Mock Langfuse
        mock_lf = MagicMock()
        mock_gen = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_gen)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_lf.start_as_current_observation.return_value = mock_cm
        langfuse_tracing._langfuse = mock_lf

        # Minimal concrete LLMClient for testing
        from bookmind_tutor.llm.base import LLMClient
        from collections.abc import Iterator

        class FakeClient(LLMClient):
            @property
            def model(self): return "test-model"
            @property
            def provider(self): return "test"
            @property
            def last_response(self): return self._resp

            def _complete(self, messages, system="", tools=None, max_tokens=4096):
                self._resp = CompletionResponse(
                    text="hello",
                    stop_reason="end_turn",
                    tool_calls=[],
                    usage=TokenUsage(input_tokens=10, output_tokens=5),
                    raw_message={"role": "assistant", "content": "hello"},
                )
                return self._resp

            def _stream(self, messages, system="", tools=None, max_tokens=4096) -> Iterator[str]:
                yield "hello"

        client = FakeClient()
        client.complete([{"role": "user", "content": "hi"}])

        # Langfuse should have been called with "generation" type
        mock_lf.start_as_current_observation.assert_called_once()
        call_kwargs = mock_lf.start_as_current_observation.call_args[1]
        assert call_kwargs["as_type"] == "generation"
        assert call_kwargs["model"] == "test-model"

        # Generation should have been updated with output + usage
        mock_gen.update.assert_called_once()
        update_kwargs = mock_gen.update.call_args[1]
        assert update_kwargs["output"] == "hello"
        assert update_kwargs["usage_details"]["input"] == 10
        assert update_kwargs["usage_details"]["output"] == 5
