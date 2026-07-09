# Plan: LLM Provider Abstraction

## Problem

The entire codebase is tightly coupled to `anthropic.Anthropic()`:
- 7 files import and call `anthropic` directly
- Provider-specific types leak everywhere (`stop_reason == "end_turn"`, `type == "tool_use"`, Anthropic content blocks in memory)
- Switching to OpenAI (or any other provider) requires editing every caller

## Goal

A thin abstraction layer that lets us swap providers (Anthropic, OpenAI, Groq, ...) by changing one config value, with zero changes to business logic.

## Design

### Canonical format: OpenAI

OpenAI's message format is the de facto industry standard — adopted by Groq, Together, Mistral, LiteLLM, and many others. Using it as the canonical internal format means the OpenAI adapter is near-zero-overhead, and adding new providers is easy.

This requires changing `ConversationMemory` to store OpenAI-format messages (significant but mechanical).

### New package: `bookmind_tutor/llm/`

```
bookmind_tutor/llm/
  __init__.py          # exports create_client(), LLMClient, LLMAPIError
  base.py              # abstract types and interface
  anthropic_client.py  # Anthropic adapter
  openai_client.py     # OpenAI adapter
```

### Core types (`base.py`)

```python
@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int

@dataclass
class ToolCallRequest:
    """A tool call the model wants to make."""
    id: str        # unique call ID (used in tool result)
    name: str
    input: dict

@dataclass
class CompletionResponse:
    text: str                          # final text (empty when stop_reason is "tool_use")
    stop_reason: str                   # "end_turn" | "tool_use" | "max_tokens"
    tool_calls: list[ToolCallRequest]  # populated when stop_reason == "tool_use"
    usage: TokenUsage
    # Raw assistant message in OpenAI format — pass directly to ConversationMemory.add_assistant()
    raw_message: dict

class LLMAPIError(Exception):
    """Provider-agnostic API error. Adapters catch provider exceptions and re-raise this."""
    def __init__(self, message: str, status_code: int | None = None) -> None: ...

class LLMClient(ABC):
    @property
    def model(self) -> str: ...

    def complete(
        self,
        messages: list[dict],      # OpenAI format
        system: str = "",
        tools: list[dict] | None = None,  # OpenAI function-tool format
        max_tokens: int = 4096,
    ) -> CompletionResponse: ...

    def stream(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> Iterator[str]:
        """Yield text tokens. After generator exhaustion, `last_response` is set."""
        ...

    @property
    def last_response(self) -> CompletionResponse | None:
        """The CompletionResponse from the most recent stream() call, available after exhaustion."""
        ...
```

### Tool spec format

`ToolSpec.to_dict()` returns OpenAI format (replaces `to_anthropic_dict()`):
```python
{
    "type": "function",
    "function": {
        "name": "book_search",
        "description": "...",
        "parameters": { ... }   # same JSON Schema as current input_schema
    }
}
```

Each adapter converts this to its provider's expected format internally.

### Memory format change (`ConversationMemory`)

Switch internal storage to OpenAI format:

```python
# add_user: unchanged (already stores plain string → list of text blocks)
# add_assistant: takes a dict (the raw_message from CompletionResponse)
def add_assistant(self, raw_message: dict) -> None:
    self._messages.append(raw_message)

# add_tool_results: each result becomes a separate role="tool" message
def add_tool_results(self, results: list[ToolCallResult]) -> None:
    for r in results:
        self._messages.append({
            "role": "tool",
            "tool_call_id": r.tool_use_id,
            "content": r.content,
        })
```

The `ToolCallResult.tool_use_id` field name stays unchanged (it holds the call ID regardless of provider).

### Anthropic adapter (`anthropic_client.py`)

Converts between OpenAI format (internal) and Anthropic format (wire):

**Outbound (OpenAI → Anthropic):**
- `system` string → `system` param (already separate, no change)
- `role: "tool"` messages → `role: "user"` with `content: [{type: "tool_result", ...}]`
- Tool definitions: `function.parameters` → `input_schema`
- Tool definitions: wrap as `{"name": ..., "description": ..., "input_schema": ...}`

**Inbound (Anthropic → CompletionResponse):**
- `stop_reason: "end_turn"` → `"end_turn"` (pass through)
- `stop_reason: "tool_use"` → `"tool_use"` (pass through)
- Content blocks of `type: "tool_use"` → `ToolCallRequest(id, name, input)`
- `raw_message`: synthesized as OpenAI-format assistant message with `tool_calls` list

**Error handling:** Catch `anthropic.APIStatusError`, `anthropic.APIConnectionError` → re-raise as `LLMAPIError`.

### OpenAI adapter (`openai_client.py`)

Near-passthrough — OpenAI format is already canonical:
- `system` string → prepend as `{"role": "system", "content": system}` to messages
- `stop_reason`: map `"stop"` → `"end_turn"`, `"tool_calls"` → `"tool_use"`
- Tool calls: map `choice.message.tool_calls[i]` → `ToolCallRequest`
- `raw_message`: `choice.message` as dict

**Error handling:** Catch `openai.APIStatusError`, `openai.APIConnectionError` → `LLMAPIError`.

### Factory function

```python
# bookmind_tutor/llm/__init__.py
def create_client(
    provider: str,          # "anthropic" | "openai"
    model: str,
    api_key: str | None = None,
    **kwargs,
) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(model=model, api_key=api_key, **kwargs)
    if provider == "openai":
        return OpenAIClient(model=model, api_key=api_key, **kwargs)
    raise ValueError(f"Unknown provider: {provider!r}")
```

### Configuration

Add to `.env`:
```
LLM_PROVIDER=openai          # or "anthropic"
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=...      # keep for when switching back
```

`app.py` reads these at startup via `create_client()`.

## Callers that change

| File | Change |
|---|---|
| `agents/harness.py` | Use `LLMClient.complete()` / `.stream()`; normalize stop_reason; use `raw_message` for memory |
| `agents/strategies/plan_execute.py` | Replace `client.messages.create()` / `.stream()` with `LLMClient.complete()` / `.stream()` |
| `agents/strategies/reflexion.py` | Replace `client.messages.create()` with `LLMClient.complete()` |
| `knowledge_graph/extractor.py` | Replace with `LLMClient.complete()` |
| `ingestion/structure_detector.py` | Replace with `LLMClient.complete()` |
| `tutor/app.py` | `create_client()` at startup; catch `LLMAPIError` instead of `anthropic.APIStatusError` |
| `agents/models.py` | `ToolSpec.to_dict()` returns OpenAI format; `ToolCallResult` unchanged |
| `agents/memory.py` | `add_assistant(raw_message: dict)`; `add_tool_results` stores role="tool" messages |

## What does NOT change

- `ToolExecutor` — dispatches tool calls by name, receives `ToolCallRequest` (same fields)
- `Tool` base class and all tool implementations
- All strategies (`ReActStrategy`, `StrategyRouter`, `ReflexionStrategy` logic)
- `QAAgent`, `KGUpdateSuggester`, `GraphStore`, all ingestion modules
- `VectorStore`, `ChunkIndexer` (embeddings use sentence-transformers, no LLM client)

## Tests

**Unit tests that need updating** (mechanical format changes):
- `tests/agents/test_harness.py` — mock returns now use OpenAI format
- `tests/agents/test_memory.py` — add_assistant / add_tool_results assertions updated

**New tests:**
- `tests/llm/test_anthropic_client.py` — format conversion round-trips (no real API calls)
- `tests/llm/test_openai_client.py` — format conversion round-trips
- `tests/llm/test_factory.py` — create_client() returns correct type

All other tests are unaffected (they mock at the harness level, not the memory level).

## Scope

- New: `bookmind_tutor/llm/` (4 files)
- Modified: `harness.py`, `memory.py`, `models.py`, `plan_execute.py`, `reflexion.py`, `extractor.py`, `structure_detector.py`, `app.py`
- Not touched: ingestion models/chunker, all strategies logic, retrieval, KG store, frontend pages
