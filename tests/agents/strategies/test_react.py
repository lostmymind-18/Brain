"""Tests for ReActStrategy."""
from unittest.mock import MagicMock

from bookmind_tutor.agents.memory import ConversationMemory
from bookmind_tutor.agents.strategies.react import ReActStrategy


def _make_strategy():
    harness = MagicMock()
    harness.chat.return_value = "mocked answer"
    harness.total_llm_calls = 0
    return ReActStrategy(harness=harness), harness


def test_delegates_to_harness():
    strategy, harness = _make_strategy()
    result = strategy.chat("What is CAP theorem?")
    harness.chat.assert_called_once_with("What is CAP theorem?", memory=None)
    assert result == "mocked answer"


def test_passes_memory_to_harness():
    strategy, harness = _make_strategy()
    memory = ConversationMemory()
    strategy.chat("What is replication?", memory=memory)
    harness.chat.assert_called_once_with("What is replication?", memory=memory)


def test_last_trace_populated():
    strategy, _ = _make_strategy()
    strategy.chat("What is eventual consistency?")
    assert strategy.last_trace is not None
    assert strategy.last_trace.strategy_name == "react"
    assert strategy.last_trace.answer == "mocked answer"
    assert strategy.last_trace.elapsed_seconds >= 0
