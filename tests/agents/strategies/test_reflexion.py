"""Tests for ReflexionStrategy."""
import json
from unittest.mock import MagicMock

from bookmind_tutor.agents.memory import ConversationMemory
from bookmind_tutor.agents.strategies.reflexion import SCORE_THRESHOLD, ReflexionStrategy
from bookmind_tutor.llm.base import CompletionResponse, TokenUsage


def _completion(text: str) -> CompletionResponse:
    return CompletionResponse(
        text=text,
        stop_reason="end_turn",
        tool_calls=[],
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        raw_message={"role": "assistant", "content": text},
    )


def _make_strategy(reflection_score: int, missing: str = "citations"):
    harness = MagicMock()
    harness.chat.side_effect = ["answer v1", "answer v2"]
    harness.total_llm_calls = 0

    llm_client = MagicMock()
    should_retry = reflection_score < SCORE_THRESHOLD
    reflection_json = json.dumps({
        "score": reflection_score,
        "missing": missing,
        "should_retry": should_retry,
    })
    llm_client.complete.return_value = _completion(reflection_json)

    strategy = ReflexionStrategy(harness=harness, llm_client=llm_client)
    return strategy, harness, llm_client


def test_no_retry_when_score_at_threshold():
    strategy, harness, _ = _make_strategy(reflection_score=SCORE_THRESHOLD)
    result = strategy.chat("What is CAP theorem?")
    assert result == "answer v1"
    assert harness.chat.call_count == 1


def test_no_retry_when_score_above_threshold():
    strategy, harness, _ = _make_strategy(reflection_score=5)
    result = strategy.chat("question")
    assert result == "answer v1"
    assert harness.chat.call_count == 1


def test_retry_when_score_below_threshold():
    strategy, harness, _ = _make_strategy(reflection_score=SCORE_THRESHOLD - 1)
    result = strategy.chat("Explain replication in depth")
    assert result == "answer v2"
    assert harness.chat.call_count == 2


def test_retry_prompt_includes_feedback():
    strategy, harness, _ = _make_strategy(
        reflection_score=2,
        missing="no page citations and misses trade-offs",
    )
    strategy.chat("original question")
    second_call_question = harness.chat.call_args_list[1].args[0]
    assert "no page citations and misses trade-offs" in second_call_question
    assert "original question" in second_call_question


def test_no_retry_when_missing_is_empty_even_if_score_low():
    """should_retry=False when missing is empty — nothing specific to improve."""
    harness = MagicMock()
    harness.chat.return_value = "answer"
    harness.total_llm_calls = 0
    llm_client = MagicMock()
    reflection = json.dumps({"score": 2, "missing": "", "should_retry": False})
    llm_client.complete.return_value = _completion(reflection)
    strategy = ReflexionStrategy(harness=harness, llm_client=llm_client)
    strategy.chat("question")
    assert harness.chat.call_count == 1


def test_skip_retry_on_reflection_parse_failure():
    """A broken reflection response must never crash — fall back to v1 answer."""
    harness = MagicMock()
    harness.chat.return_value = "answer v1"
    harness.total_llm_calls = 0
    llm_client = MagicMock()
    llm_client.complete.return_value = _completion("not json")
    strategy = ReflexionStrategy(harness=harness, llm_client=llm_client)
    result = strategy.chat("question")
    assert result == "answer v1"
    assert harness.chat.call_count == 1


def test_last_trace_contains_reflection_info():
    strategy, _, _ = _make_strategy(reflection_score=2, missing="missing citations")
    strategy.chat("question")
    trace = strategy.last_trace
    assert trace is not None
    assert trace.strategy_name == "reflexion"
    assert any("score=2" in s for s in trace.steps)
    assert trace.num_llm_calls >= 1  # at minimum: reflection call


def test_memory_updated_with_final_answer():
    strategy, _, _ = _make_strategy(reflection_score=5)
    memory = ConversationMemory()
    strategy.chat("question", memory=memory)
    msgs = memory.messages()
    assert len(msgs) == 2
    assert msgs[-1]["role"] == "assistant"
