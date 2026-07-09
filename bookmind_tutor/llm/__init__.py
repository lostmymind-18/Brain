"""LLM provider abstraction — public API."""
from __future__ import annotations

from bookmind_tutor.llm.base import (
    CompletionResponse,
    LLMAPIError,
    LLMClient,
    TokenUsage,
    ToolCallRequest,
)


def create_client(
    provider: str,
    model: str,
    api_key: str | None = None,
    **kwargs,
) -> LLMClient:
    """
    Factory function. Returns an LLMClient for the given provider.

    Args:
        provider: "anthropic" or "openai"
        model: Provider-specific model ID.
        api_key: API key. If None, reads from environment (ANTHROPIC_API_KEY or OPENAI_API_KEY).
        **kwargs: Passed to the adapter constructor (e.g. base_url for OpenAI-compatible endpoints).
    """
    if provider == "anthropic":
        from bookmind_tutor.llm.anthropic_client import AnthropicClient
        return AnthropicClient(model=model, api_key=api_key, **kwargs)

    if provider == "openai":
        from bookmind_tutor.llm.openai_client import OpenAIClient
        return OpenAIClient(model=model, api_key=api_key, **kwargs)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        "Supported providers: 'anthropic', 'openai'."
    )


__all__ = [
    "create_client",
    "LLMClient",
    "LLMAPIError",
    "CompletionResponse",
    "TokenUsage",
    "ToolCallRequest",
]
