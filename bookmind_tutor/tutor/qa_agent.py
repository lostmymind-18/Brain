"""
QAAgent: reactive Q&A agent with lightweight pedagogical enhancements.

Role: answer user questions accurately and with depth. Not a proactive tutor —
it does not design curricula or initiate learning paths. Socratic questioning
and Feynman technique suggestions are mixed in *when contextually appropriate*,
not on every turn.

Pedagogical techniques (plan: plans/qa_agent.md):
  - Student modeling: infers user expertise level from conversation history via
    keyword heuristics. Adjusts system prompt depth/language accordingly.
  - Socratic mixing: after every 3rd turn, if the question is not a simple
    factual lookup, the system prompt invites (but does not mandate) a
    thought-provoking follow-up question.
  - Feynman suggestion: after 5+ turns on the same topic domain, the system
    prompt optionally suggests asking the user to explain a concept in their
    own words.

All of these are *soft hints* to the LLM, not hard-coded post-processing.
The LLM decides whether to apply them based on the actual context.
"""
from __future__ import annotations

import re
from collections.abc import Iterator

from bookmind_tutor.agents.harness import AgentHarness, _DEFAULT_SYSTEM
from bookmind_tutor.agents.memory import ConversationMemory
from bookmind_tutor.agents.models import ReasoningTrace
from bookmind_tutor.agents.strategies.router import StrategyRouter
from bookmind_tutor.observability.tracing import QAAttrs, get_tracer

# ---------------------------------------------------------------------------
# Level detection heuristics
# ---------------------------------------------------------------------------

# Generic question-starters ("what is", "what are") are excluded — they don't
# reliably indicate beginner level; an expert also asks "what is X" when
# exploring unfamiliar territory. Beginner signals focus on framing/vocabulary.
_BEGINNER_PATTERNS = re.compile(
    r"\b(i don't understand|i do not understand|what do you mean"
    r"|in simple terms|simply explain|for beginners|can you explain"
    r"|how does .{1,30} work|what does .{1,20} mean)\b",
    re.IGNORECASE,
)

# Advanced signals: reasoning ("why does/would"), tradeoffs, plural forms
# included so "edge cases" and "failure modes" also match.
_ADVANCED_PATTERNS = re.compile(
    r"\b(why does|why would|tradeoffs?|trade-offs?|edge cases?"
    r"|failure modes?|internals?|nuance|implications?"
    r"|when should i|when to use|versus|differ|underlying|mechanism"
    r"|compare .{1,30} (?:vs|versus|with|against)|deep dive)\b",
    re.IGNORECASE,
)

# Simple factual lookups — skip Socratic on these
_FACTUAL_PATTERNS = re.compile(
    r"^(what is|what are|define|who is|when was|where is)\b",
    re.IGNORECASE,
)

_SHORT_QUESTION_WORDS = 5   # questions under this many words skip pedagogical mixing


# ---------------------------------------------------------------------------
# System prompt building blocks
# ---------------------------------------------------------------------------

_LEVEL_ADAPTATION = {
    "beginner": (
        "The user appears to be new to this topic. "
        "Use plain language and relatable analogies. "
        "Define technical terms before using them. "
        "Keep explanations concrete rather than abstract."
    ),
    "intermediate": (
        "The user has some familiarity with the subject. "
        "Assume basic concepts are understood. "
        "Highlight nuance, tradeoffs, and real-world implications."
    ),
    "advanced": (
        "The user is asking at a sophisticated level. "
        "Engage with depth: discuss edge cases, failure modes, "
        "and technical precision where relevant."
    ),
}

_SOCRATIC_HINT = (
    "If it feels genuinely natural and adds value, you may end your answer with "
    "one thought-provoking question that invites the user to think deeper. "
    "Do not force it — skip it if the question is purely factual."
)

_FEYNMAN_HINT = (
    "If the user seems to be building solid familiarity with a concept discussed "
    "in this answer, you may (optionally) briefly suggest they try explaining it "
    "in their own words as a quick self-check. Keep the suggestion short and light."
)


# ---------------------------------------------------------------------------
# QAAgent
# ---------------------------------------------------------------------------

class QAAgent:
    """
    Wraps StrategyRouter with conversation-aware pedagogical enhancements.

    Args:
        harness: The AgentHarness shared across all strategies. QAAgent updates
            its system_prompt before each turn.
        router: StrategyRouter that selects ReAct / Plan-Execute / Reflexion.
    """

    def __init__(
        self,
        harness: AgentHarness,
        router: StrategyRouter,
        book_name: str = "",
    ) -> None:
        self._harness = harness
        self._router = router
        self._memory = ConversationMemory()
        self._turn_count = 0
        self._book_name = book_name

    def chat(self, question: str, force_strategy: str | None = None) -> str:
        """
        Process one user turn.

        Updates the harness system prompt based on inferred user level and
        turn-based pedagogical triggers, then delegates to StrategyRouter.

        Args:
            force_strategy: Optional strategy override — "react", "plan-execute",
                or "reflexion". None means auto-route by keyword heuristic.
        """
        self._turn_count += 1
        self._harness.system_prompt = self._build_system_prompt(question)
        tracer = get_tracer()
        with tracer.start_as_current_span("qa.turn") as span:
            span.set_attribute(QAAttrs.TURN_COUNT, self._turn_count)
            answer = self._router.chat(question, memory=self._memory, force_strategy=force_strategy)
            if self._router.last_trace:
                span.set_attribute(QAAttrs.STRATEGY, self._router.last_trace.strategy_name)
        return answer

    def stream_chat(
        self, question: str, force_strategy: str | None = None
    ) -> Iterator[str]:
        """
        Stream one user turn. Updates system prompt same as chat(), then
        delegates to router.stream_chat(). Sets last_trace when exhausted.
        """
        self._turn_count += 1
        self._harness.system_prompt = self._build_system_prompt(question)
        tracer = get_tracer()
        with tracer.start_as_current_span("qa.turn") as span:
            span.set_attribute(QAAttrs.TURN_COUNT, self._turn_count)
            yield from self._router.stream_chat(
                question, memory=self._memory, force_strategy=force_strategy
            )
            if self._router.last_trace:
                span.set_attribute(QAAttrs.STRATEGY, self._router.last_trace.strategy_name)

    @property
    def last_trace(self) -> ReasoningTrace | None:
        return self._router.last_trace

    def seed_memory(self, messages: list[dict]) -> None:
        """
        Replay persisted UI messages into the agent's ConversationMemory.

        Called once after agent construction when restoring a saved session.
        Skips leading assistant messages (e.g. the book intro) because the
        Anthropic API requires the first message to be a user turn.
        Also restores _turn_count so Socratic/Feynman triggers resume at the
        right cadence after reload.

        Args:
            messages: The UI message dicts from book_histories (role, content, ...).
        """
        self._memory.clear()
        self._turn_count = 0
        found_first_user = False

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                found_first_user = True
                self._memory.add_user(content)
                self._turn_count += 1
            elif role == "assistant" and found_first_user:
                self._memory.add_assistant({"role": "assistant", "content": content})

    # ------------------------------------------------------------------
    # Student modeling
    # ------------------------------------------------------------------

    def _infer_level(self, question: str = "") -> str:
        """
        Infer user expertise level from recent conversation history + current question.

        Counts beginner and advanced signals in the last 6 user messages.
        Returns "beginner", "intermediate", or "advanced".

        The current question is included because _infer_level is called BEFORE
        the question is added to memory by the harness.
        """
        recent_user_msgs: list[str] = []
        for m in self._memory.messages():
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    recent_user_msgs.append(content)
                elif isinstance(content, list):
                    text = " ".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                    if text:
                        recent_user_msgs.append(text)
        recent_user_msgs = recent_user_msgs[-6:]

        text = " ".join(recent_user_msgs) + " " + question
        beginner_hits = len(_BEGINNER_PATTERNS.findall(text))
        advanced_hits = len(_ADVANCED_PATTERNS.findall(text))

        if advanced_hits > beginner_hits:
            return "advanced"
        if beginner_hits > 0:
            return "beginner"
        return "intermediate"

    # ------------------------------------------------------------------
    # Pedagogical trigger conditions
    # ------------------------------------------------------------------

    def _should_add_socratic(self, question: str) -> bool:
        """
        True after every 3rd turn, when the question is not a simple factual lookup.

        Factual lookups ("what is X?", very short questions) don't benefit from
        a Socratic follow-up — it would feel forced.
        """
        if self._turn_count < 2:
            return False
        if len(question.split()) < _SHORT_QUESTION_WORDS:
            return False
        if _FACTUAL_PATTERNS.match(question.strip()):
            return False
        return self._turn_count % 3 == 0

    def _should_add_feynman(self) -> bool:
        """
        True after 5+ turns — user has had enough interaction to benefit from
        a self-explanation check.
        """
        return self._turn_count >= 5

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self, question: str) -> str:
        """
        Compose a context-aware system prompt for this turn.

        Structure:
          1. Base BookMind instructions (search + cite)
          2. Level-adapted language guidance
          3. Socratic hint (conditional)
          4. Feynman hint (conditional)
        """
        level = self._infer_level(question)
        base = _DEFAULT_SYSTEM
        if self._book_name:
            base = f'The book currently loaded is: "{self._book_name}". {base}'
        parts = [base, _LEVEL_ADAPTATION[level]]

        if self._should_add_socratic(question):
            parts.append(_SOCRATIC_HINT)

        if self._should_add_feynman():
            parts.append(_FEYNMAN_HINT)

        return "\n\n".join(parts)
