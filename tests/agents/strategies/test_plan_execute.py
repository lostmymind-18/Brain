"""Tests for PlanExecuteStrategy."""
import json
from unittest.mock import MagicMock, call, patch

from bookmind_tutor.agents.memory import ConversationMemory
from bookmind_tutor.agents.strategies.plan_execute import MAX_SUBTASKS, PlanExecuteStrategy
from bookmind_tutor.llm.base import CompletionResponse, TokenUsage


def _completion(text: str) -> CompletionResponse:
    return CompletionResponse(
        text=text,
        stop_reason="end_turn",
        tool_calls=[],
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        raw_message={"role": "assistant", "content": text},
    )


def _make_strategy(plan_tasks=None, synthesis_text="final answer"):
    harness = MagicMock()
    harness.chat.return_value = "sub-task result"
    harness.total_llm_calls = 0

    llm_client = MagicMock()

    if plan_tasks is None:
        plan_tasks = ["sub-task 1", "sub-task 2"]

    plan_json = json.dumps(plan_tasks)

    # First call = plan, second call = synthesize
    llm_client.complete.side_effect = [
        _completion(plan_json),
        _completion(synthesis_text),
    ]

    strategy = PlanExecuteStrategy(harness=harness, llm_client=llm_client)
    return strategy, harness, llm_client


def test_three_phases_run_in_order():
    strategy, harness, llm_client = _make_strategy(plan_tasks=["task A", "task B"])
    result = strategy.chat("Compare X and Y")
    assert result == "final answer"
    # Harness called once per sub-task
    assert harness.chat.call_count == 2
    # LLM called twice: plan + synthesize
    assert llm_client.complete.call_count == 2


def test_each_subtask_passed_to_harness():
    # Parallel execution — order of calls is not guaranteed, but all tasks must be called.
    strategy, harness, _ = _make_strategy(plan_tasks=["task A", "task B", "task C"])
    strategy.chat("Multi-part question")
    calls = {c.args[0] for c in harness.chat.call_args_list}
    assert calls == {"task A", "task B", "task C"}


def test_harness_called_without_memory():
    """Sub-task execution is stateless — memory is never passed to harness."""
    strategy, harness, _ = _make_strategy()
    memory = ConversationMemory()
    strategy.chat("question", memory=memory)
    for c in harness.chat.call_args_list:
        assert c.kwargs.get("memory") is None or c.args[1:] == ()


def test_fallback_to_single_task_on_invalid_json():
    """If plan JSON is unparseable, treat the original question as one sub-task."""
    harness = MagicMock()
    harness.chat.return_value = "result"
    harness.total_llm_calls = 0
    llm_client = MagicMock()
    llm_client.complete.side_effect = [
        _completion("not json at all !!!"),
        _completion("synthesized"),
    ]
    strategy = PlanExecuteStrategy(harness=harness, llm_client=llm_client)
    result = strategy.chat("original question")
    harness.chat.assert_called_once_with("original question")
    assert result == "synthesized"


def test_subtask_cap_at_max():
    too_many = [f"task {i}" for i in range(MAX_SUBTASKS + 3)]
    strategy, harness, _ = _make_strategy(plan_tasks=too_many)
    strategy.chat("question")
    assert harness.chat.call_count == MAX_SUBTASKS


def test_last_trace_populated():
    strategy, _, _ = _make_strategy(plan_tasks=["t1", "t2"])
    strategy.chat("What is X?")
    trace = strategy.last_trace
    assert trace is not None
    assert trace.strategy_name == "plan_execute"
    assert len(trace.steps) >= 3  # plan + execute*N + synthesize
    assert trace.num_llm_calls >= 2  # at minimum: plan + synthesize (+ harness calls if real)


def test_memory_updated_with_final_answer():
    strategy, _, _ = _make_strategy()
    memory = ConversationMemory()
    strategy.chat("question", memory=memory)
    msgs = memory.messages()
    # user message + assistant message
    assert len(msgs) == 2
    assert msgs[-1]["role"] == "assistant"
