"""
AgentHarness: the core ReAct loop for BookMind Tutor.

Architecture:
  User message → LLMClient → tool_use? → ToolExecutor → LLMClient → ... → end_turn

This implements the ReAct pattern (Yao et al. 2022) using the model's native
tool_use stop_reason rather than prompt-engineering explicit Thought/Action/
Observation markers. The "reasoning" step is internal to the model; what we
observe externally is the sequence of tool calls and their results.

Intentional design choices:
- Provider-agnostic: uses LLMClient abstraction, not anthropic SDK directly.
- No LangChain/LangGraph: implementing the loop manually exposes every detail
  that frameworks abstract away, which is the learning goal for Week 2.
"""
from __future__ import annotations

import threading
from collections.abc import Iterator

import structlog

from bookmind_tutor.agents.executor import ToolExecutor
from bookmind_tutor.agents.memory import ConversationMemory
from bookmind_tutor.agents.models import ToolCallResult
from bookmind_tutor.agents.tools.base import Tool
from bookmind_tutor.llm.base import LLMAPIError, LLMClient
from bookmind_tutor.observability.tracing import AgentAttrs, get_tracer

logger = structlog.get_logger()

_DEFAULT_SYSTEM = (
    "You are BookMind Tutor, an AI assistant that helps users understand a specific book "
    "that has already been indexed and is available via the book_search tool. "
    "Always call book_search first before answering any question — do not ask the user "
    "which book they are referring to, because the book is already loaded. "
    "Always cite the chapter and page number when you quote or paraphrase from the book. "
    "Write the final answer directly. Never include meta-commentary like "
    "'Now I have enough information', 'Let me compile', 'Based on my search', "
    "'Perfect!', or any phrase that describes what you are about to do. "
    "Start immediately with the content of the answer. "
    "GROUNDING RULES: Every factual claim about the book — chapter names, page numbers, "
    "author arguments, specific examples — must appear in the retrieved excerpts or "
    "book_outline output. Never use your training knowledge to fill gaps. "
    "If the search results do not contain enough information to answer confidently, "
    "say 'I could not find this in the book' rather than guessing or fabricating details."
)

_MAX_ITERATIONS_FALLBACK = (
    "(Note: I reached the maximum number of reasoning steps. "
    "The answer above may be incomplete.)"
)


class AgentHarness:
    """
    Orchestrates multi-turn, tool-using conversations with an LLM.

    Usage - stateless single turn:
        harness = AgentHarness(tools=[BookSearchTool(store)], llm_client=client)
        answer = harness.chat("What is replication?")

    Usage - stateful multi-turn:
        memory = ConversationMemory()
        answer1 = harness.chat("What is replication?", memory=memory)
        answer2 = harness.chat("What are its tradeoffs?", memory=memory)

    Args:
        tools: List of Tool instances available to the model.
        system_prompt: System prompt shown at the start of every call.
        max_iterations: Max ReAct loop steps before forcing a stop.
        llm_client: LLMClient instance (AnthropicClient or OpenAIClient).
        max_tokens: Max tokens in the model's response per step.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: str = "claude-haiku-4-5-20251001",  # kept for backward compat, ignored when llm_client provided
        system_prompt: str = _DEFAULT_SYSTEM,
        max_iterations: int = 10,
        llm_client: LLMClient | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self._tools = tools
        self._system = system_prompt
        self._max_iterations = max_iterations
        self._max_tokens = max_tokens
        self._executor = ToolExecutor(tools)

        if llm_client is not None:
            self._client = llm_client
        else:
            # Fallback: create Anthropic client using the model param for backward compat
            from bookmind_tutor.llm.anthropic_client import AnthropicClient
            self._client = AnthropicClient(model=model)

        # Cumulative counter across all chat() calls. Protected by a lock so
        # PlanExecuteStrategy can call harness from multiple threads safely.
        self._total_llm_calls: int = 0
        self._call_lock = threading.Lock()
        # Per-turn step log — reset at the start of each chat()/stream_chat() call.
        self._last_steps: list[str] = []
        # book_search results from the most recent chat()/stream_chat() call.
        # Populated to support eval logging (Week 7) and DPO data (Week 10-11).
        self._last_retrieved_chunks: list[str] = []

    @property
    def total_llm_calls(self) -> int:
        """Total number of LLM API calls made by this harness since creation."""
        return self._total_llm_calls

    @property
    def last_steps(self) -> list[str]:
        """Step log from the most recent chat() or stream_chat() call."""
        return list(self._last_steps)

    @property
    def last_retrieved_chunks(self) -> list[str]:
        """Texts returned by book_search, graph_rag_search, or get_book_section during the most recent chat() call."""
        return list(self._last_retrieved_chunks)

    @property
    def system_prompt(self) -> str:
        return self._system

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system = value

    def chat(self, user_message: str, memory: ConversationMemory | None = None) -> str:
        """
        Process one user turn through the ReAct loop.

        If memory is None, the conversation is stateless (fresh each call).
        Pass a ConversationMemory instance to maintain multi-turn context.

        Returns the final text response.
        """
        own_memory = memory is None
        if own_memory:
            memory = ConversationMemory()

        self._last_steps = []
        self._last_retrieved_chunks = []
        memory.add_user(user_message)
        tool_dicts = [t.spec.to_dict() for t in self._tools]

        final_text = ""
        response = None
        tracer = get_tracer()

        with tracer.start_as_current_span("agent.react_turn") as turn_span:
            turn_span.set_attribute(AgentAttrs.STRATEGY, "react")

            for step in range(1, self._max_iterations + 1):
                logger.info("react_step", step=step, max_steps=self._max_iterations)

                with tracer.start_as_current_span("agent.react_step") as step_span:
                    step_span.set_attribute(AgentAttrs.STEP_NUM, step)

                    response = self._client.complete(
                        messages=memory.messages(),
                        system=self._system,
                        tools=tool_dicts if tool_dicts else None,
                        max_tokens=self._max_tokens,
                    )
                    with self._call_lock:
                        self._total_llm_calls += 1

                    in_tok = response.usage.input_tokens
                    out_tok = response.usage.output_tokens
                    logger.debug(
                        "llm_response",
                        stop_reason=response.stop_reason,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                    )

                    if response.stop_reason == "end_turn":
                        final_text = response.text
                        self._last_steps.append(
                            f"[{step}] end_turn — {out_tok} out / {in_tok} in tokens"
                        )
                        logger.info("react_done", step=step, answer_chars=len(final_text))
                        break

                    if response.stop_reason == "tool_use":
                        for tc in response.tool_calls:
                            query = _tool_input_summary(tc.input)
                            self._last_steps.append(f"[{step}] {tc.name}({query})")
                        memory.add_assistant(response.raw_message)
                        results: list[ToolCallResult] = self._executor.execute_all(response.tool_calls)
                        for tc, r in zip(response.tool_calls, results):
                            if tc.name in ("book_search", "graph_rag_search", "get_book_section") and not r.is_error:
                                self._last_retrieved_chunks.append(r.content)
                            status = "error" if r.is_error else f"{len(r.content)} chars"
                            self._last_steps.append(f"[{step}]  → {status}")
                        memory.add_tool_results(results)
                        _log_tool_results(results)
                        continue

                    logger.warning("unexpected_stop_reason", stop_reason=response.stop_reason)
                    self._last_steps.append(f"[{step}] unexpected stop: {response.stop_reason}")
                    final_text = response.text
                    break

            else:
                logger.warning("max_iterations_reached", max_iterations=self._max_iterations)
                self._last_steps.append(f"[{self._max_iterations}] max iterations reached")
                final_text = (response.text if response else "") + "\n\n" + _MAX_ITERATIONS_FALLBACK

            turn_span.set_attribute(AgentAttrs.TOTAL_LLM_CALLS, self._total_llm_calls)

        if not own_memory:
            memory.add_assistant({"role": "assistant", "content": final_text})

        return final_text

    def stream_chat(
        self, user_message: str, memory: ConversationMemory | None = None
    ) -> Iterator[str]:
        """
        Process one user turn through the ReAct loop, streaming the final answer.

        Tool-call steps yield nothing (no text tokens during tool_use). The
        end_turn step yields tokens as they arrive.

        Caller receives a generator — consume it fully before reading last_steps
        or total_llm_calls, as those are updated while the generator runs.
        """
        own_memory = memory is None
        if own_memory:
            memory = ConversationMemory()

        self._last_steps = []
        self._last_retrieved_chunks = []
        memory.add_user(user_message)
        tool_dicts = [t.spec.to_dict() for t in self._tools]
        final_text_parts: list[str] = []
        tracer = get_tracer()

        with tracer.start_as_current_span("agent.react_turn") as turn_span:
            turn_span.set_attribute(AgentAttrs.STRATEGY, "react")

            for step in range(1, self._max_iterations + 1):
                logger.info("react_step_stream", step=step, max_steps=self._max_iterations)
                step_text: list[str] = []

                with tracer.start_as_current_span("agent.react_step") as step_span:
                    step_span.set_attribute(AgentAttrs.STEP_NUM, step)

                    for token in self._client.stream(
                        messages=memory.messages(),
                        system=self._system,
                        tools=tool_dicts if tool_dicts else None,
                        max_tokens=self._max_tokens,
                    ):
                        step_text.append(token)
                        yield token

                    response = self._client.last_response
                    if response is None:
                        logger.error("stream_no_last_response")
                        break

                    with self._call_lock:
                        self._total_llm_calls += 1

                    in_tok = response.usage.input_tokens
                    out_tok = response.usage.output_tokens
                    logger.debug(
                        "llm_response",
                        stop_reason=response.stop_reason,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                    )

                    if response.stop_reason == "end_turn":
                        final_text_parts = step_text
                        self._last_steps.append(
                            f"[{step}] end_turn — {out_tok} out / {in_tok} in tokens"
                        )
                        logger.info("react_done", step=step)
                        break

                    if response.stop_reason == "tool_use":
                        for tc in response.tool_calls:
                            query = _tool_input_summary(tc.input)
                            self._last_steps.append(f"[{step}] {tc.name}({query})")
                        memory.add_assistant(response.raw_message)
                        results: list[ToolCallResult] = self._executor.execute_all(response.tool_calls)
                        for tc, r in zip(response.tool_calls, results):
                            if tc.name in ("book_search", "graph_rag_search", "get_book_section") and not r.is_error:
                                self._last_retrieved_chunks.append(r.content)
                            status = "error" if r.is_error else f"{len(r.content)} chars"
                            self._last_steps.append(f"[{step}]  → {status}")
                        memory.add_tool_results(results)
                        _log_tool_results(results)
                        continue

                    logger.warning("unexpected_stop_reason", stop_reason=response.stop_reason)
                    self._last_steps.append(f"[{step}] unexpected stop: {response.stop_reason}")
                    final_text_parts = step_text
                    break
            else:
                logger.warning("max_iterations_reached", max_iterations=self._max_iterations)
                self._last_steps.append(f"[{self._max_iterations}] max iterations reached")

            turn_span.set_attribute(AgentAttrs.TOTAL_LLM_CALLS, self._total_llm_calls)

        if not own_memory:
            final_text = "".join(final_text_parts).strip()
            memory.add_assistant({"role": "assistant", "content": final_text})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_input_summary(input_dict: dict) -> str:
    """One-line summary of a tool's input for the step log."""
    if "query" in input_dict:
        q = str(input_dict["query"])[:60]
        return f"'{q}'" if len(str(input_dict["query"])) <= 60 else f"'{q}...'"
    for k, v in input_dict.items():
        return f"{k}={str(v)[:50]!r}"
    return ""


def _log_tool_results(results: list[ToolCallResult]) -> None:
    for r in results:
        logger.info(
            "tool_result",
            tool_use_id=r.tool_use_id[:8],
            status="ERROR" if r.is_error else "OK",
            preview=r.content[:80].replace("\n", " "),
        )
