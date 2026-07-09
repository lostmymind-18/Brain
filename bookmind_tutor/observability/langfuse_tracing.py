"""
Langfuse integration - opt-in via LANGFUSE_SECRET_KEY env var.

All public functions are no-ops when Langfuse is not configured, so callers
never need to check whether it is enabled before calling.

Usage
-----
    with langfuse_observation("llm.stream", as_type="generation", model="gpt-4o-mini",
                              input=messages) as obs:
        result = call_llm()
        if obs is not None:
            obs.update(output=result, usage_details={"input": 100, "output": 50})

Setup (env vars)
----------------
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com   # default, or self-hosted URL
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

_langfuse: Any = None
_configured = False


def configure_langfuse() -> None:
    """
    Initialize Langfuse from env vars. Idempotent — safe to call multiple times.
    Silently skips if LANGFUSE_SECRET_KEY is not set.
    """
    global _langfuse, _configured
    if _configured:
        return
    _configured = True

    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not secret_key:
        return

    from langfuse import Langfuse
    _langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=secret_key,
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )


def get_langfuse() -> Any:
    """Return the Langfuse singleton, or None if not configured."""
    return _langfuse


def langfuse_observation(name: str, as_type: str = "span", **kwargs):
    """
    Context manager wrapping a Langfuse observation.
    Returns nullcontext() when Langfuse is disabled so callers never need
    a guard. The 'as' variable is None (from nullcontext) or a Langfuse
    span/generation/tool object.

    as_type options: "span", "generation", "agent", "tool", "chain",
                     "retriever", "evaluator", "guardrail", "embedding"
    """
    if _langfuse is None:
        return nullcontext()
    return _langfuse.start_as_current_observation(name=name, as_type=as_type, **kwargs)


def langfuse_flush() -> None:
    """Flush pending Langfuse events. Call after each streamed turn."""
    if _langfuse is not None:
        _langfuse.flush()


def reset_for_testing() -> None:
    """Reset state so unit tests can reconfigure from scratch."""
    global _langfuse, _configured
    _langfuse = None
    _configured = False
