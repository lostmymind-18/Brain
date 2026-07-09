"""
Input guards: check user input before it reaches the LLM.

Design principles:
- Rule-based guards (LengthGuard, PromptInjectionGuard) run in microseconds —
  no network calls, always safe to add.
- LLM-based guards (TopicalityGuard) fail OPEN: if the guard's own LLM call
  fails, the user's request goes through. Never punish the user for our infra failure.
- All violations are logged with structured fields for monitoring. The
  user_message shown on BLOCK is intentionally vague — no hint to attackers.
"""
from __future__ import annotations

import re

import structlog

from bookmind_tutor.safety.models import GuardContext, GuardResult, Severity, ViolationType

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Prompt injection patterns
# ---------------------------------------------------------------------------

# Each tuple: (compiled regex, short category name for logs)
_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"ignore\s+(all\s+|previous\s+|prior\s+)?(instructions?|prompts?|rules?|context)",
            re.IGNORECASE,
        ),
        "ROLE_OVERRIDE",
    ),
    (
        re.compile(
            r"\b(you\s+are\s+now|act\s+as|pretend\s+(you\s+are|to\s+be)|your\s+new\s+(role|persona|instructions?))\b",
            re.IGNORECASE,
        ),
        "ROLE_OVERRIDE",
    ),
    (
        re.compile(
            r"\b(repeat|print|show|reveal|output|tell\s+me)\s+(your\s+|the\s+)?(system\s+|initial\s+|original\s+)?prompt\b",
            re.IGNORECASE,
        ),
        "SYSTEM_LEAK",
    ),
    (
        re.compile(
            r"\b(jailbreak|dan\s+mode|developer\s+mode|no\s+restrictions|unrestricted\s+mode)\b",
            re.IGNORECASE,
        ),
        "JAILBREAK",
    ),
    (
        re.compile(
            r"(</?(system|human|assistant|user)>|<\|im_start\||<\|im_end\|>|\[/?INST\]|\[SYS\])",
            re.IGNORECASE,
        ),
        "DELIMITER_INJ",
    ),
]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

class LengthGuard:
    """Block inputs that exceed a character limit."""

    def __init__(self, max_chars: int = 2000) -> None:
        self._max = max_chars

    def check(self, text: str, ctx: GuardContext) -> GuardResult:
        if len(text) <= self._max:
            return GuardResult(passed=True)
        return GuardResult(
            passed=False,
            violation_type=ViolationType.INPUT_TOO_LONG,
            severity=Severity.BLOCK,
            user_message=(
                f"Your question is too long ({len(text)} characters, max {self._max}). "
                "Try splitting it into smaller questions."
            ),
            internal_detail=f"input_chars={len(text)} max={self._max}",
        )


class PromptInjectionGuard:
    """
    Detect common prompt injection patterns using rule-based regex.

    Rule-based is intentional here — it's deterministic, zero-latency,
    zero-cost, and covers the vast majority of real attack patterns.
    LLM-based injection detection exists but is overkill for this use case.
    """

    def check(self, text: str, ctx: GuardContext) -> GuardResult:
        for pattern, category in _INJECTION_PATTERNS:
            if pattern.search(text):
                logger.warning(
                    "guardrail_block",
                    violation=ViolationType.PROMPT_INJECTION,
                    category=category,
                    book_id=ctx.book_id,
                    preview=text[:50],
                )
                return GuardResult(
                    passed=False,
                    violation_type=ViolationType.PROMPT_INJECTION,
                    severity=Severity.BLOCK,
                    user_message="This question cannot be processed.",
                    internal_detail=f"matched_category={category}",
                )
        return GuardResult(passed=True)


class TopicalityGuard:
    """
    Detect off-topic questions using a single cheap LLM call (max_tokens=5).

    Severity is WARN, not BLOCK, because:
    - False positive rate is non-trivial (a book question that mentions
      "ignore the chapter structure" would hit rule-based guards instead).
    - A wrongly blocked question severely harms UX.
    - WARN lets us collect data and tune thresholds without breaking anything.

    Fails OPEN: if the LLM call itself fails, we return passed=True.
    """

    _PROMPT = (
        'Book: "{book_name}"\n'
        'Question: "{question}"\n'
        "Is this question related to the book above? "
        "Reply with exactly one word: yes or no."
    )

    def __init__(self, llm_client, enabled: bool = True) -> None:
        self._client = llm_client
        self._enabled = enabled

    def check(self, text: str, ctx: GuardContext) -> GuardResult:
        if not self._enabled or not ctx.book_name:
            return GuardResult(passed=True)

        try:
            resp = self._client.complete(
                messages=[{
                    "role": "user",
                    "content": self._PROMPT.format(
                        book_name=ctx.book_name,
                        question=text[:500],   # truncate for cost control
                    ),
                }],
                max_tokens=5,
            )
            is_related = resp.text.strip().lower().startswith("yes")
        except Exception:
            # Fail open — never block a user because our guard's infra failed
            return GuardResult(passed=True)

        if not is_related:
            logger.warning(
                "guardrail_warn",
                violation=ViolationType.OFF_TOPIC,
                book_id=ctx.book_id,
                book_name=ctx.book_name,
                preview=text[:50],
            )
            return GuardResult(
                passed=False,
                violation_type=ViolationType.OFF_TOPIC,
                severity=Severity.WARN,
                user_message="",   # WARN: user sees nothing, we just log it
                internal_detail=f"llm_answer={resp.text.strip()!r}",
            )

        return GuardResult(passed=True)
