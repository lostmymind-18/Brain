"""Data models for the guardrails layer."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    BLOCK = "block"  # hard stop — return error to user, do not call LLM
    WARN  = "warn"   # suspicious — log, let through
    LOG   = "log"    # silent track only


class ViolationType(str, Enum):
    INPUT_TOO_LONG   = "input_too_long"
    PROMPT_INJECTION = "prompt_injection"
    OFF_TOPIC        = "off_topic"
    OUTPUT_TOO_LONG  = "output_too_long"


@dataclass
class GuardContext:
    """Request-scoped context passed to every guard."""
    book_name: str = ""
    book_id: str = ""
    session_id: str = ""


@dataclass
class GuardResult:
    """
    Output of a single guard check.

    passed=True  → guard cleared (no violation).
    passed=False + BLOCK → caller must stop and show user_message.
    passed=False + WARN  → log internally, let through.
    """
    passed: bool
    violation_type: ViolationType | None = None
    severity: Severity | None = None
    user_message: str = ""        # shown to user on BLOCK; empty on WARN
    internal_detail: str = ""     # logged internally, never shown to user
