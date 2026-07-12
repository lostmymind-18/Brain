"""
BookMind Tutor — Streamlit UI.

Flow:
  1. User uploads a PDF → indexed into a per-book VectorStore (isolated collection)
  2. User chats with the agent (StrategyRouter selects ReAct / Plan-and-Execute / Reflexion)
  3. After each answer, KGUpdateSuggester proposes new entities for the user's Knowledge Graph
  4. User approves suggestions → entities saved to Neo4j as UserEntity nodes

Multiple books are supported: each book gets its own ChromaDB collection so queries
never mix content across books. The user KG (UserEntity nodes in Neo4j) is shared
across all books — it is the user's personal brain, not tied to any single book.

Run:
    PYTHONPATH=. streamlit run bookmind_tutor/tutor/app.py
"""
from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Configure observability before any other imports so all modules pick up the
# tracer provider and structlog config set here.
from bookmind_tutor.observability.setup import configure_langfuse, configure_observability, configure_structlog
configure_structlog()
configure_observability()
configure_langfuse()

import streamlit as st

from bookmind_tutor.evaluation.logger import InteractionLogger, InteractionRecord
from bookmind_tutor.llm import LLMAPIError, create_client
from bookmind_tutor.safety import (
    GuardContext,
    GuardrailsLayer,
    LengthGuard,
    OutputLengthGuard,
    PromptInjectionGuard,
    Severity,
    TopicalityGuard,
)

from bookmind_tutor.agents.harness import AgentHarness
from bookmind_tutor.agents.strategies import (
    PlanExecuteStrategy,
    ReActStrategy,
    ReflexionStrategy,
    StrategyRouter,
)
from bookmind_tutor.agents.tools.book_outline import BookOutlineTool
from bookmind_tutor.agents.tools.get_book_section import GetBookSectionTool
from bookmind_tutor.agents.tools.graph_rag_search import GraphRAGSearchTool
from bookmind_tutor.agents.verification import CitationVerifier, VerificationResult
from bookmind_tutor.knowledge_graph.extractor import EntityExtractor
from bookmind_tutor.knowledge_graph.graph_rag import GraphRAGRetriever
from bookmind_tutor.tutor.persistence import BookRecord, LibraryStore
from bookmind_tutor.tutor.qa_agent import QAAgent
from bookmind_tutor.knowledge_graph.graph_store import GraphStore
from bookmind_tutor.knowledge_graph.suggester import KGUpdateSuggester
from bookmind_tutor.retrieval.indexer import ChunkIndexer
from bookmind_tutor.retrieval.reranker import CrossEncoderReranker
from bookmind_tutor.retrieval.vector_store import VectorStore

# Shared reranker instance — model is lazy-loaded on the first search.
# Set ENABLE_RERANKER=false to disable (falls back to RRF-only retrieval).
_RERANKER: CrossEncoderReranker | None = (
    CrossEncoderReranker()
    if os.getenv("ENABLE_RERANKER", "true").lower() not in ("false", "0", "no")
    else None
)

_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "bookmind123")

_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
_LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
_LLM_BASE_URL = os.getenv("LLM_BASE_URL")  # e.g. http://localhost:11434/v1 for Ollama

_DATA_ROOT = Path("data")
_CHROMA_DIR = str(_DATA_ROOT / "chroma")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_error_message(exc: LLMAPIError) -> str:
    """Convert an LLMAPIError into a short, actionable UI message."""
    code = exc.code
    if code == "credit_exhausted":
        return str(exc)
    if code == "auth_error":
        return str(exc)
    if code == "rate_limit":
        return str(exc)
    if code == "connection_error":
        return str(exc)
    return f"API error: {exc}"


_INTRO_SYSTEM_PROMPT = """\
You are a knowledgeable reading assistant introducing a book to a new reader.
Based on the excerpts provided, write a concise, engaging introduction that covers:
1. What the book is fundamentally about (1-2 sentences)
2. The key themes or areas it explores (a short list or brief paragraph)
3. Who would benefit most from reading it

End with a single, light open-ended question inviting the reader to share what they'd
like to explore — do not suggest a specific topic, just open the door.

Keep the total response under 250 words. Be warm but not overly enthusiastic.
"""


def _generate_book_intro(vs: "VectorStore", book_name: str, llm_client) -> str:
    """Generate a welcome intro for a newly indexed book using a few representative chunks."""
    probes = ["introduction overview", "key concepts principles", "conclusion summary"]
    seen: set[str] = set()
    excerpts: list[str] = []
    for probe in probes:
        for result in vs.search(probe, k=3):
            if result.chunk_id not in seen:
                seen.add(result.chunk_id)
                header = f"[{result.chapter or 'Unknown chapter'}, p.{result.page_range[0]}]"
                excerpts.append(f"{header}\n{result.text[:400]}")

    excerpts_text = "\n\n---\n\n".join(excerpts[:8])
    user_message = (
        f'Book title: "{book_name}"\n\n'
        f"Excerpts:\n\n{excerpts_text}"
    )
    response = llm_client.complete(
        messages=[{"role": "user", "content": user_message}],
        system=_INTRO_SYSTEM_PROMPT,
        max_tokens=400,
    )
    return response.text


def _book_id(filename: str) -> str:
    """Derive a stable, readable collection key from a PDF filename.

    ChromaDB requires names matching [a-zA-Z0-9._-], starting and ending
    with [a-zA-Z0-9], length 3-512. Strip leading/trailing underscores
    after substitution to satisfy the boundary constraint.
    """
    name = filename.removesuffix(".pdf").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", name)[:40].strip("_")
    return slug or "book"


def _current_vs() -> VectorStore:
    """Return the VectorStore for the currently active book."""
    return st.session_state.book_stores[st.session_state.current_book_id]


def _current_messages() -> list:
    """Return the conversation history for the currently active book."""
    return st.session_state.book_histories[st.session_state.current_book_id]


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        # Per-book state — keyed by book_id
        "book_stores": {},      # dict[book_id, VectorStore]
        "book_names": {},       # dict[book_id, str]  display name (no .pdf)
        "book_records": {},     # dict[book_id, BookRecord]  for persistence
        "book_histories": {},   # dict[book_id, list[msg]]

        # Active book
        "current_book_id": None,

        # Global UI state
        "approved_msg_ids": set(),  # msg_ids where user already approved KG suggestions
        "next_msg_id": 0,
        "strategy_mode": "Auto",    # sidebar strategy selector
        "confirm_delete": None,     # book_id pending delete confirmation
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if "llm_client" not in st.session_state:
        kwargs = {}
        if _LLM_BASE_URL and _LLM_PROVIDER == "openai":
            kwargs["base_url"] = _LLM_BASE_URL
        st.session_state.llm_client = create_client(
            provider=_LLM_PROVIDER,
            model=_LLM_MODEL,
            **kwargs,
        )
        st.session_state.llm_provider = _LLM_PROVIDER

    if "graph_store" not in st.session_state:
        # GraphStore is shared across all books — it holds the user's personal KG.
        st.session_state.graph_store = GraphStore(
            uri=_NEO4J_URI,
            user=_NEO4J_USER,
            password=_NEO4J_PASSWORD,
        )

    if "library_store" not in st.session_state:
        st.session_state.library_store = LibraryStore(_DATA_ROOT)
        _restore_from_disk()

    if "interaction_logger" not in st.session_state:
        st.session_state.interaction_logger = InteractionLogger(_DATA_ROOT / "interactions.jsonl")

    if "citation_verifier" not in st.session_state:
        st.session_state.citation_verifier = CitationVerifier(
            llm_client=st.session_state.llm_client
        )

    if "guardrails" not in st.session_state:
        # TopicalityGuard makes a cheap LLM call (max_tokens=5) to check relevance.
        # It fails open, so a guard LLM failure never blocks a legitimate question.
        st.session_state.guardrails = GuardrailsLayer(
            input_guards=[
                LengthGuard(),
                PromptInjectionGuard(),
                TopicalityGuard(llm_client=st.session_state.llm_client),
            ],
            output_guards=[OutputLengthGuard()],
        )


def _restore_from_disk() -> None:
    """
    Load persisted books and histories into session state.

    Called once per process at startup. Books whose ChromaDB collection is
    empty (e.g. the chroma directory was deleted) are skipped with a warning.
    """
    saved = st.session_state.library_store.load_library()
    stale: list[str] = []

    for bid, record in saved.items():
        vs = VectorStore(persist_dir=_CHROMA_DIR, collection_name=f"book_{bid}", reranker=_RERANKER)
        if vs.count() == 0:
            stale.append(record.name)
            continue
        st.session_state.book_stores[bid] = vs
        st.session_state.book_names[bid] = record.name
        st.session_state.book_records[bid] = record
        st.session_state.book_histories[bid] = (
            st.session_state.library_store.load_history(bid)
        )

    if stale:
        # Store names for main() to render — can't call st.warning here
        # because set_page_config hasn't run yet when _init_state is invoked.
        st.session_state["_restore_warnings"] = stale

    # Restore next_msg_id to avoid Streamlit form-key collisions across sessions.
    all_ids = [
        msg["msg_id"]
        for history in st.session_state.book_histories.values()
        for msg in history
        if msg.get("msg_id") is not None
    ]
    if all_ids:
        st.session_state.next_msg_id = max(all_ids) + 1


def _save_library() -> None:
    """Persist the current book registry to disk."""
    st.session_state.library_store.save_library(st.session_state.book_records)


def _delete_book(bid: str) -> None:
    """
    Permanently remove a book: drop ChromaDB collection, delete history file,
    remove from session state and library registry.
    """
    vs = st.session_state.book_stores.get(bid)
    if vs:
        vs.drop()
    st.session_state.library_store.delete_history(bid)

    st.session_state.book_stores.pop(bid, None)
    st.session_state.book_names.pop(bid, None)
    st.session_state.book_records.pop(bid, None)
    st.session_state.book_histories.pop(bid, None)
    _save_library()

    if st.session_state.current_book_id == bid:
        st.session_state.current_book_id = None
    st.session_state.confirm_delete = None


# ---------------------------------------------------------------------------
# Upload page
# ---------------------------------------------------------------------------

def _upload_page() -> None:
    st.title("📚 BookMind Tutor")

    # If books are already indexed in this session, show them first.
    if st.session_state.book_stores:
        st.subheader("Already indexed")
        for bid, name in st.session_state.book_names.items():
            col1, col2 = st.columns([4, 1])
            col1.write(f"📖 {name}")
            if col2.button("Chat →", key=f"switch_{bid}"):
                _switch_book(bid)
        st.divider()

    st.subheader("Upload a book")
    st.write("Choose a PDF to index. The tutor will be ready to answer questions about it.")

    uploaded = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded is not None:
        bid = _book_id(uploaded.name)
        already_indexed = bid in st.session_state.book_stores

        label = "Re-index Book" if already_indexed else "Index Book"
        if st.button(label, type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded.read())
                tmp_path = f.name

            try:
                with st.spinner(f"Indexing {uploaded.name}... (this may take ~30s)"):
                    # Each book gets its own persistent ChromaDB collection.
                    vs = VectorStore(
                        persist_dir=_CHROMA_DIR,
                        collection_name=f"book_{bid}",
                        reranker=_RERANKER,
                    )
                    indexer = ChunkIndexer(
                        vector_store=vs,
                        llm_client=st.session_state.llm_client,
                    )
                    n_chunks = indexer.index_pdf(tmp_path)

                # Register book in session state.
                book_name = uploaded.name.removesuffix(".pdf")
                st.session_state.book_stores[bid] = vs
                st.session_state.book_names[bid] = book_name
                st.session_state.book_records[bid] = BookRecord(
                    name=book_name,
                    n_chunks=n_chunks,
                    indexed_at=datetime.now(timezone.utc).isoformat(),
                )
                _save_library()

                is_new_book = not already_indexed
                if is_new_book:
                    st.session_state.book_histories[bid] = []

                # Generate intro only for freshly indexed books (not re-index).
                if is_new_book:
                    try:
                        with st.spinner("Generating book introduction..."):
                            intro = _generate_book_intro(
                                vs, book_name, st.session_state.llm_client
                            )
                        st.session_state.book_histories[bid] = [{
                            "role": "assistant",
                            "content": intro,
                            "trace": None,
                            "sources": [],
                            "suggestions": [],
                            "msg_id": 0,
                        }]
                        st.session_state.next_msg_id = 1
                    except LLMAPIError as exc:
                        st.warning(
                            f"Book indexed but intro could not be generated. "
                            f"{_api_error_message(exc)}"
                        )
                    # Persist the intro (or empty history) immediately.
                    st.session_state.library_store.save_history(
                        bid, st.session_state.book_histories[bid]
                    )

                st.success(f"Ready! Indexed {n_chunks} chunks from {uploaded.name}.")
                _switch_book(bid)
            except LLMAPIError as exc:
                st.error(f"Could not index book: {_api_error_message(exc)}")
            finally:
                os.unlink(tmp_path)


def _switch_book(book_id: str) -> None:
    """Set the active book and re-initialise the agent for it."""
    st.session_state.current_book_id = book_id
    _setup_agent()
    st.rerun()


def _setup_agent() -> None:
    """(Re-)create router + suggester for the currently active book."""
    llm = st.session_state.llm_client
    retriever = GraphRAGRetriever(
        vector_store=_current_vs(),
        graph_store=st.session_state.graph_store,  # shared user KG
        k=20,
    )
    search_tool = GraphRAGSearchTool(retriever=retriever)
    outline_tool = BookOutlineTool(vector_store=_current_vs())
    section_tool = GetBookSectionTool(vector_store=_current_vs())
    harness = AgentHarness(llm_client=llm, tools=[outline_tool, section_tool, search_tool])
    # Kept so the chat handler can read tool.last_sources after each turn
    # (the Sources panel shows what the agent actually retrieved).
    st.session_state.retrieval_tools = [search_tool, section_tool]
    react = ReActStrategy(harness=harness)
    plan = PlanExecuteStrategy(harness=harness, llm_client=llm)
    reflexion = ReflexionStrategy(harness=harness, llm_client=llm)
    router = StrategyRouter(react=react, plan_execute=plan, reflexion=reflexion)
    book_name = st.session_state.book_names.get(st.session_state.current_book_id, "")
    qa_agent = QAAgent(harness=harness, router=router, book_name=book_name)
    qa_agent.seed_memory(_current_messages())
    st.session_state.qa_agent = qa_agent
    st.session_state.suggester = KGUpdateSuggester(extractor=EntityExtractor(llm_client=llm))


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------

def _chat_page() -> None:
    _render_sidebar()

    book_name = st.session_state.book_names[st.session_state.current_book_id]
    st.title(f"📚 {book_name}")

    # Render conversation history for the active book.
    for msg in _current_messages():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_trace(msg.get("trace"))
                _render_sources(msg.get("sources", []))
                _render_verification_dict(msg.get("verification"))
                _render_suggestions(msg)

    if question := st.chat_input("Ask about the book..."):
        _handle_question(question)


def _handle_question(question: str) -> None:
    # Run input guardrails before touching history or calling the LLM.
    # BLOCK: show error to user and abort. WARN: logged internally, proceed normally.
    ctx = GuardContext(
        book_name=st.session_state.book_names.get(st.session_state.current_book_id, ""),
        book_id=st.session_state.current_book_id or "",
    )
    guard_result = st.session_state.guardrails.check_input(question, ctx)
    if not guard_result.passed and guard_result.severity == Severity.BLOCK:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.error(guard_result.user_message)
        return

    messages = _current_messages()
    messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        force = _strategy_to_force(st.session_state.strategy_mode)
        # Per-turn contract with retrieval tools: clear last_sources before the
        # turn, read them after. See Tool.last_sources in agents/tools/base.py.
        for tool in st.session_state.retrieval_tools:
            tool.last_sources.clear()
        try:
            # st.write_stream() renders tokens as they arrive and returns the full string.
            answer = st.write_stream(
                st.session_state.qa_agent.stream_chat(question, force_strategy=force)
            )
        except LLMAPIError as exc:
            error_msg = _api_error_message(exc)
            st.error(error_msg)
            messages.append({
                "role": "assistant",
                "content": f"_(Error: {error_msg})_",
                "trace": None,
                "sources": [],
                "suggestions": [],
                "msg_id": st.session_state.next_msg_id,
            })
            st.session_state.next_msg_id += 1
            st.session_state.library_store.save_history(
                st.session_state.current_book_id, messages
            )
            st.rerun()
            return

        trace = st.session_state.qa_agent.last_trace

        # Output guardrail (WARN-only: logs oversized responses, never blocks).
        st.session_state.guardrails.check_output(answer, ctx)

        # KG extraction temporarily disabled — skip suggester.suggest() to reduce latency.
        from bookmind_tutor.knowledge_graph.models import KGUpdateResult
        kg_result = KGUpdateResult(suggestions=[], already_known=[])
        # Sources = what the agent's tools actually retrieved this turn.
        # (Previously this ran an independent semantic search on the question,
        # which showed chunks the agent never read.)
        search_results = [
            ref
            for tool in st.session_state.retrieval_tools
            for ref in tool.last_sources
        ]

        # Citation verification — runs after answer is complete.
        with st.spinner("Verifying citations..."):
            verification: VerificationResult | None = None
            try:
                retrieved_texts = trace.retrieved_chunks if trace else []
                verification = st.session_state.citation_verifier.verify(
                    answer, retrieved_texts
                )
            except Exception:
                pass  # Non-critical: never let verifier crash the chat.

        _render_trace(trace)
        _render_sources(search_results)
        _render_verification(verification)

    msg_id = st.session_state.next_msg_id
    st.session_state.next_msg_id += 1
    # Serialize verification result as a plain dict so it survives JSON persistence.
    verification_dict = None
    if verification and verification.checks:
        verification_dict = {
            "checks": [
                {"claim": c.claim, "grounded": c.grounded, "source": c.source}
                for c in verification.checks
            ]
        }
    messages.append({
        "role": "assistant",
        "content": answer,
        "trace": trace,
        "sources": search_results,
        "suggestions": kg_result.suggestions,
        "verification": verification_dict,
        "msg_id": msg_id,
    })
    st.session_state.library_store.save_history(
        st.session_state.current_book_id, messages
    )

    # Log the interaction for evaluation and DPO data collection (Week 7).
    if answer and trace:
        import uuid
        try:
            st.session_state.interaction_logger.log(InteractionRecord(
                record_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                book_id=st.session_state.current_book_id,
                book_name=st.session_state.book_names.get(
                    st.session_state.current_book_id, ""
                ),
                question=question,
                retrieved_chunks=trace.retrieved_chunks,
                answer=answer,
                strategy=trace.strategy_name,
                model=st.session_state.llm_client.model,
                provider=st.session_state.llm_provider,
                num_llm_calls=trace.num_llm_calls,
                elapsed_seconds=trace.elapsed_seconds,
                eval_score=None,
                eval_breakdown=None,
                human_rating=None,
            ))
        except Exception:
            # Logging is non-critical; never let it crash the chat UI.
            pass

    st.rerun()


_STRATEGY_OPTIONS = ("Auto", "ReAct", "Plan-Execute", "Reflexion")
_STRATEGY_FORCE_MAP = {
    "Auto": None,
    "ReAct": "react",
    "Plan-Execute": "plan-execute",
    "Reflexion": "reflexion",
}


def _strategy_to_force(mode: str) -> str | None:
    return _STRATEGY_FORCE_MAP.get(mode)


def _render_trace(trace) -> None:
    if not trace:
        return
    summary = (
        f"Strategy: {trace.strategy_name} · "
        f"{trace.num_llm_calls} LLM calls · "
        f"{trace.elapsed_seconds:.1f}s"
    )
    if trace.steps:
        with st.expander(summary, expanded=False):
            for s in trace.steps:
                st.caption(s)
    else:
        st.caption(summary)


def _render_sources(sources: list) -> None:
    if not sources:
        return
    # Deduplicate by (chapter, subsection, page_range) — same chunk can appear
    # twice at different similarity scores if graph expansion doubled it.
    seen: set[tuple] = set()
    unique = []
    for s in sources:
        key = (s.chapter, getattr(s, "subsection", None), s.page_range)
        if key not in seen:
            seen.add(key)
            unique.append(s)

    with st.expander(f"Sources ({len(unique)})", expanded=False):
        for s in unique:
            p_start, p_end = s.page_range
            page_str = f"p. {p_start}" if p_start == p_end else f"pp. {p_start}-{p_end}"
            label = s.chapter or "Unknown section"
            # Subsection gives the user a page-precise heading to look up in
            # the physical book; chapter alone can span 40+ pages.
            subsection = getattr(s, "subsection", None)
            if subsection:
                label = f"{label} › {subsection}"
            st.caption(f"**{label}** · {page_str}")


def _render_verification(result: VerificationResult | None) -> None:
    """Render verification from a live VerificationResult object."""
    if result is None or not result.checks:
        return
    _render_verification_checks(result.checks)


def _render_verification_dict(data: dict | None) -> None:
    """Render verification from a serialized dict (stored in message history)."""
    if not data or not data.get("checks"):
        return
    from bookmind_tutor.agents.verification import ClaimCheck
    checks = [
        ClaimCheck(claim=c["claim"], grounded=c["grounded"], source=c.get("source", ""))
        for c in data["checks"]
    ]
    _render_verification_checks(checks)


def _render_verification_checks(checks) -> None:
    g = sum(1 for c in checks if c.grounded)
    u = sum(1 for c in checks if not c.grounded)
    label_parts = []
    if g:
        label_parts.append(f"{g} verified")
    if u:
        label_parts.append(f"{u} unverified")
    label = "Fact-check: " + ", ".join(label_parts)

    # Expand automatically if there are unverified claims to draw attention.
    with st.expander(label, expanded=u > 0):
        for check in checks:
            if check.grounded:
                hint = f" _({check.source})_" if check.source else ""
                st.caption(f"✓ {check.claim}{hint}")
            else:
                st.caption(f"⚠ {check.claim} — _not found in retrieved excerpts_")


def _render_suggestions(msg: dict) -> None:
    suggestions = msg.get("suggestions")
    if not suggestions:
        return

    msg_id = msg.get("msg_id")

    if msg_id in st.session_state.approved_msg_ids:
        st.success("Added to your knowledge graph.")
        return

    with st.expander(f"💡 {len(suggestions)} new concept(s) — add to your knowledge graph?"):
        with st.form(key=f"kg_form_{msg_id}"):
            selected_flags = []
            for i, s in enumerate(suggestions):
                checked = st.checkbox(
                    f"**{s.entity.name}** ({s.entity.type})  \n_{s.entity.description}_",
                    value=True,
                    key=f"chk_{msg_id}_{i}",
                )
                selected_flags.append((checked, s.entity))

            submitted = st.form_submit_button("Add selected to Knowledge Graph", type="primary")
            if submitted:
                to_add = [e for checked, e in selected_flags if checked]
                for entity in to_add:
                    st.session_state.graph_store.upsert_user_entity(entity)
                st.session_state.approved_msg_ids.add(msg_id)
                if to_add:
                    st.success(f"Added {len(to_add)} concept(s)!")
                st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    with st.sidebar:
        # --- Book library ---
        st.header("Library")
        current_bid = st.session_state.current_book_id
        for bid, name in st.session_state.book_names.items():
            col_name, col_del = st.columns([5, 1])
            with col_name:
                if bid == current_bid:
                    st.write(f"● **{name}**")
                else:
                    if st.button(name, key=f"lib_{bid}"):
                        _switch_book(bid)
            with col_del:
                if st.button("🗑", key=f"del_{bid}", help=f"Delete '{name}'"):
                    st.session_state.confirm_delete = bid
                    st.rerun()

        # Inline delete confirmation (replaces the book row)
        if st.session_state.confirm_delete:
            confirm_bid = st.session_state.confirm_delete
            confirm_name = st.session_state.book_names.get(confirm_bid, confirm_bid)
            st.warning(f"Delete **{confirm_name}**?\nThis removes all embeddings and chat history.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Delete", type="primary", key="del_confirm"):
                    _delete_book(confirm_bid)
                    st.rerun()
            with col_no:
                if st.button("Cancel", key="del_cancel"):
                    st.session_state.confirm_delete = None
                    st.rerun()

        if st.button("+ Upload another book", type="secondary"):
            # Return to upload page without losing existing book state.
            st.session_state.current_book_id = None
            st.rerun()

        st.divider()

        # --- Strategy selector ---
        st.header("Reasoning Strategy")
        st.session_state.strategy_mode = st.selectbox(
            "Mode",
            options=_STRATEGY_OPTIONS,
            index=_STRATEGY_OPTIONS.index(st.session_state.strategy_mode),
            help=(
                "Auto: keyword-based routing\n"
                "ReAct: single-pass tool + reason loop\n"
                "Plan-Execute: break into sub-tasks first\n"
                "Reflexion: self-evaluate and optionally retry"
            ),
            label_visibility="collapsed",
        )
        st.caption(
            "Auto selects based on your question. "
            "Override if you want a specific strategy."
        )

        st.divider()

        # --- User Knowledge Graph ---
        st.header("My Knowledge Graph")
        entity_names = st.session_state.graph_store.get_user_entity_names()

        if entity_names:
            st.write(f"**{len(entity_names)} concept(s) known:**")
            for name in sorted(entity_names):
                st.write(f"- {name}")

            if st.button("Clear my knowledge graph", type="secondary"):
                st.session_state.graph_store.clear_user_graph()
                st.rerun()
        else:
            st.write("Empty — start chatting to build it!")
            st.caption("After each answer, you can add new concepts to your personal knowledge graph.")

        st.divider()
        st.caption("BookMind Tutor · Week 5-6")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="BookMind Tutor",
        page_icon="📚",
        layout="wide",
    )
    _init_state()

    # Show one-shot warning for books that could not be restored from disk.
    if "_restore_warnings" in st.session_state:
        stale = st.session_state["_restore_warnings"]
        del st.session_state["_restore_warnings"]
        names = ", ".join(f"**{n}**" for n in stale)
        st.warning(
            f"Some books could not be restored (embeddings missing - please re-upload): {names}"
        )

    if st.session_state.current_book_id is None:
        _upload_page()
    else:
        _chat_page()


if __name__ == "__main__":
    main()
