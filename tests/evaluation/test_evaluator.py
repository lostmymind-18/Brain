"""Tests for AnswerEvaluator."""
import json
from unittest.mock import MagicMock

import pytest

from bookmind_tutor.evaluation.evaluator import AnswerEvaluator, EvalResult
from bookmind_tutor.llm.base import CompletionResponse, TokenUsage


def _completion(text: str) -> CompletionResponse:
    return CompletionResponse(
        text=text,
        stop_reason="end_turn",
        tool_calls=[],
        usage=TokenUsage(input_tokens=50, output_tokens=30),
        raw_message={"role": "assistant", "content": text},
    )


def _mock_client(model: str = "gpt-4o-mini") -> MagicMock:
    client = MagicMock()
    client.model = model
    return client


def _valid_json(fa: float = 0.9, cq: float = 0.8, dr: float = 0.7) -> str:
    return json.dumps({
        "factual_accuracy": fa,
        "citation_quality": cq,
        "depth_and_relevance": dr,
        "reasoning": "The answer is well-grounded in the retrieved text.",
    })


class TestAnswerEvaluator:
    def test_valid_json_response(self) -> None:
        client = _mock_client()
        client.complete.return_value = _completion(_valid_json())
        evaluator = AnswerEvaluator(client=client)
        result = evaluator.evaluate("What is X?", ["Chunk A"], "X is Y.")
        assert result.factual_accuracy == pytest.approx(0.9)
        assert result.citation_quality == pytest.approx(0.8)
        assert result.depth_and_relevance == pytest.approx(0.7)
        assert result.overall == pytest.approx((0.9 + 0.8 + 0.7) / 3.0)
        assert result.reasoning == "The answer is well-grounded in the retrieved text."
        assert result.model_used == "gpt-4o-mini"

    def test_markdown_fenced_json_stripped(self) -> None:
        client = _mock_client()
        fenced = "```json\n" + _valid_json() + "\n```"
        client.complete.return_value = _completion(fenced)
        evaluator = AnswerEvaluator(client=client)
        result = evaluator.evaluate("Q?", [], "A.")
        assert result.overall != -1.0

    def test_overall_is_mean(self) -> None:
        client = _mock_client()
        client.complete.return_value = _completion(_valid_json(fa=1.0, cq=0.5, dr=0.0))
        evaluator = AnswerEvaluator(client=client)
        result = evaluator.evaluate("Q?", [], "A.")
        assert result.overall == pytest.approx(0.5)

    def test_first_call_fails_retries(self) -> None:
        """If the first response is invalid JSON, it retries and parses the second."""
        client = _mock_client()
        client.complete.side_effect = [
            _completion("not json at all"),   # first attempt fails
            _completion(_valid_json()),        # retry succeeds
        ]
        evaluator = AnswerEvaluator(client=client)
        result = evaluator.evaluate("Q?", ["ctx"], "A.")
        assert result.overall != -1.0
        assert client.complete.call_count == 2

    def test_both_attempts_fail_returns_sentinel(self) -> None:
        client = _mock_client()
        client.complete.side_effect = [
            _completion("garbage"),
            _completion("still garbage"),
        ]
        evaluator = AnswerEvaluator(client=client)
        result = evaluator.evaluate("Q?", [], "A.")
        assert result.factual_accuracy == pytest.approx(-1.0)
        assert result.overall == pytest.approx(-1.0)
        assert "failed" in result.reasoning.lower()

    def test_api_exception_triggers_retry(self) -> None:
        """If the first call raises, retry is attempted."""
        client = _mock_client()
        client.complete.side_effect = [
            ConnectionError("network down"),   # first attempt raises
            _completion(_valid_json()),        # retry succeeds
        ]
        evaluator = AnswerEvaluator(client=client)
        result = evaluator.evaluate("Q?", [], "A.")
        assert result.overall != -1.0

    def test_no_chunks_uses_no_context_placeholder(self) -> None:
        """Empty chunks list doesn't crash; placeholder inserted in prompt."""
        client = _mock_client()
        client.complete.return_value = _completion(_valid_json())
        evaluator = AnswerEvaluator(client=client)
        result = evaluator.evaluate("Q?", [], "A.")
        # The prompt was sent (we can inspect the call)
        call_args = client.complete.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "(no retrieved context)" in user_content
        assert result.overall != -1.0

    def test_model_used_field_set(self) -> None:
        client = _mock_client(model="claude-haiku-4-5-20251001")
        client.complete.return_value = _completion(_valid_json())
        evaluator = AnswerEvaluator(client=client)
        result = evaluator.evaluate("Q?", ["chunk"], "A.")
        assert result.model_used == "claude-haiku-4-5-20251001"

    def test_sentinel_model_used_on_failure(self) -> None:
        client = _mock_client(model="gpt-4o-mini")
        client.complete.side_effect = [_completion("bad"), _completion("bad")]
        evaluator = AnswerEvaluator(client=client)
        result = evaluator.evaluate("Q?", [], "A.")
        assert result.model_used == "gpt-4o-mini"
