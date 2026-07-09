# Plan: Book Introduction Message on Upload

## Problem

After uploading a book, the chat page is empty.
User has no context about what the book contains or where to start.

## Solution

After indexing completes, make one LLM call to generate a welcome intro message.
Store it as the first assistant message in the book's history.
When chat page renders, the intro appears immediately - no user action needed.

## Design Decisions

**LLM-generated, not static:**
Chapter lists alone are dry. LLM synthesizes a natural intro: key themes, scope, reading
suggestions. One extra call at upload time is acceptable (user is already waiting).

**Direct LLM call, not QAAgent:**
QAAgent is set up for the book after `_switch_book()`. The intro is generated before
that switch, so we use `llm_client.messages.create()` directly with a simple prompt.

**Not stored in conversation memory:**
The intro message is in `book_histories[bid]` (displayed in chat), but NOT fed into
`QAAgent.conversation_memory`. It's a one-way orientation, not a conversational turn.
This avoids polluting the QA context with a synthetic user turn.

**Consistent with reactive-agent philosophy:**
The intro ends with an open invitation ("What would you like to explore?") - not a
forced curriculum. The user can ignore it entirely or ask anything else.

## Implementation

### 1. `_generate_book_intro(vs, book_name, llm_client) -> str`

New function in `app.py`. Steps:
1. Query VectorStore with 3 broad probes: `"introduction overview"`, `"key concepts"`,
   `"conclusion summary"` - each returns k=3 chunks. Deduplicate by chunk_id.
2. Build a prompt with book name + excerpts (truncated to ~2000 chars total).
3. Call `llm_client.messages.create()` with `claude-haiku-4-5-20251001`, max_tokens=400.
4. System prompt instructs: introduce the book (themes, structure, scope), end with
   a light open-ended invitation to explore.
5. Return the text content.

### 2. Upload flow change (`_upload_page`)

After `indexer.index_pdf(tmp_path)` and before `_switch_book(bid)`:

```python
with st.spinner("Generating book introduction..."):
    intro = _generate_book_intro(vs, book_name, st.session_state.llm_client)

# Only add intro for freshly indexed books (not re-index).
if not already_indexed:
    st.session_state.book_histories[bid] = [{
        "role": "assistant",
        "content": intro,
        "trace": None,
        "sources": [],
        "suggestions": [],
        "msg_id": 0,
    }]
    st.session_state.next_msg_id = 1
```

For re-index: keep existing history (don't overwrite conversation).

### 3. No changes needed elsewhere

`_chat_page()` already renders `_current_messages()` in a loop - the intro will
appear automatically as the first assistant message.
`_render_trace(None)` and `_render_sources([])` already handle empty values gracefully.

## Scope

- Only `app.py` changes.
- No new modules, no new tests (LLM-dependent generation, difficult to unit test
  meaningfully; covered by E2E verification).
