"""
Post-answer citation verifier for hallucination detection.

Uses a single LLM call to:
1. Extract up to 5 factual claims from the tutor's answer
2. Check each claim against the retrieved book excerpts

The result is display-only — it annotates but never modifies or blocks the answer.
A "hard" check in the sense that it is programmatic (not just a prompt hint), but
the response to a failure is transparency rather than suppression.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from bookmind_tutor.llm.base import LLMClient

logger = logging.getLogger(__name__)

_VERIFY_PROMPT = """\
You are a fact-checker for an AI book tutor.

RETRIEVED EXCERPTS FROM THE BOOK:
---
{context}
---

AI TUTOR ANSWER:
{answer}

Extract up to 5 distinct factual claims from the tutor's answer (skip greetings, \
meta-commentary, and vague opinions — only concrete facts about the book's content).
For each claim, determine whether it is directly supported by the retrieved excerpts above.

Return ONLY valid JSON — an array of objects, no text before or after:
[
  {{
    "claim": "<short factual statement from the answer>",
    "grounded": <true if the excerpts clearly support it, false if not>,
    "source": "<brief quote or location hint from the excerpts, or empty string if not found>"
  }}
]

Be strict: if a claim cannot be verified from the excerpts alone, set grounded=false.
"""

_MIN_ANSWER_LENGTH = 80  # skip verification for very short answers


@dataclass
class ClaimCheck:
    claim: str
    grounded: bool
    source: str  # supporting quote/reference from excerpts, or "" if ungrounded


@dataclass
class VerificationResult:
    checks: list[ClaimCheck]

    @property
    def grounded_count(self) -> int:
        return sum(1 for c in self.checks if c.grounded)

    @property
    def ungrounded_count(self) -> int:
        return sum(1 for c in self.checks if not c.grounded)

    @property
    def is_fully_grounded(self) -> bool:
        return self.ungrounded_count == 0 and len(self.checks) > 0


class CitationVerifier:
    """
    Programmatic citation verifier — checks that claims in a tutor answer
    are grounded in the retrieved book excerpts.

    Single LLM call per answer: extract claims + check each against context.
    Recommended to use a cheap/fast model (e.g. gpt-4o-mini, claude-haiku-4-5)
    since this runs after every response.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._client = llm_client

    def verify(self, answer: str, chunks: list[str]) -> VerificationResult | None:
        """
        Verify that factual claims in answer appear in the retrieved chunks.

        Args:
            answer: Full answer text produced by the tutor.
            chunks: Retrieved chunk texts (SearchResult.text or raw tool output).

        Returns:
            VerificationResult with per-claim grounded/ungrounded status,
            or None if the call fails or the answer is too short to check.
        """
        if not answer.strip() or len(answer) < _MIN_ANSWER_LENGTH:
            return None
        if not chunks:
            return VerificationResult(checks=[])

        # Cap context and answer to stay within token limits
        context = "\n---\n".join(chunks[:5])[:3000]
        answer_excerpt = answer[:1500]

        prompt = _VERIFY_PROMPT.format(context=context, answer=answer_excerpt)

        try:
            response = self._client.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
        except Exception as exc:
            logger.debug("CitationVerifier: LLM call failed: %s", exc)
            return None

        raw = response.text.strip()
        # Strip markdown fences if the model wrapped the JSON
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("CitationVerifier: JSON parse failed, raw=%s", raw[:200])
            return None

        if not isinstance(data, list):
            return None

        checks: list[ClaimCheck] = []
        for item in data[:5]:  # enforce max 5 claims
            if not isinstance(item, dict):
                continue
            try:
                checks.append(ClaimCheck(
                    claim=str(item.get("claim", "")).strip(),
                    grounded=bool(item.get("grounded", False)),
                    source=str(item.get("source", "")).strip(),
                ))
            except (TypeError, ValueError):
                continue

        if not checks:
            return None

        return VerificationResult(checks=checks)
