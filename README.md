# BookMind Tutor

An interactive AI tutor that lets you have deep conversations with any book.
Upload a PDF, ask questions, and get answers grounded in the actual text — with citations, source pages, and a personal knowledge graph that grows as you learn.

Built as a 13-week learning project to master Agent Engineering, Knowledge Graphs, and RL fine-tuning (DPO proof-of-concept).

---

## What it does

- **Book Q&A** — Ask anything about an indexed book. The agent retrieves relevant passages and answers with chapter/page citations.
- **Multi-strategy reasoning** — Three agent strategies (ReAct, Plan-and-Execute, Reflexion) with automatic routing based on question complexity.
- **Knowledge Graph** — Concepts from the book are extracted into a Neo4j graph. Your personal knowledge graph grows as you approve suggestions after each answer.
- **GraphRAG retrieval** — Vector search enriched with knowledge graph traversal to surface passages that keyword search misses.
- **Streaming UI** — Token-by-token streaming in Streamlit with visible reasoning trace, sources, and KG suggestions per turn.
- **Evaluation framework** — LLM-as-judge scoring (factual accuracy, citation quality, depth) with JSONL interaction log that doubles as DPO training data.
- **Multi-provider** — Switch between Anthropic and OpenAI via a single env var. Canonical OpenAI message format internally; adapters convert at the wire.

---

## Architecture

```
bookmind_tutor/
├── ingestion/          # PDF → hierarchical chunks (PyMuPDF + NLTK)
├── retrieval/          # ChromaDB vector store + sentence-transformers embeddings
├── agents/
│   ├── harness.py      # ReAct loop (manual, no framework)
│   ├── memory.py       # Conversation memory (OpenAI message format)
│   ├── executor.py     # Tool execution
│   └── strategies/     # ReAct, Plan-and-Execute, Reflexion, StrategyRouter
├── knowledge_graph/    # Entity extraction, Neo4j store, GraphRAG retriever
├── llm/                # Provider abstraction: AnthropicClient, OpenAIClient
├── tutor/              # QAAgent (student modeling + pedagogical hints), Streamlit UI
└── evaluation/         # InteractionLogger, AnswerEvaluator (LLM-as-judge), EvalRunner
```

**Key design decisions:**
- No LangChain/LlamaIndex — the ReAct loop is implemented manually to expose every detail (learning goal).
- OpenAI message format as canonical internal format. Anthropic adapter converts at call boundary.
- `@dataclass` by default; pydantic only where validation complexity justifies it.
- Append-only JSONL interaction log — supports DPO preference pair construction in Week 10-11.

---

## Stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.11+ |
| PDF parsing | PyMuPDF (AGPL, non-commercial) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (embedded, persistent) |
| Knowledge graph | Neo4j 5 |
| LLM providers | Anthropic / OpenAI (configurable) |
| Frontend | Streamlit |
| Tests | pytest — 278 tests |

---

## Quickstart

### 1. Prerequisites

- Python 3.11+
- Neo4j 5 running locally (or Docker)

```bash
# Neo4j via Docker
docker run -d --name bookmind-neo4j \
  -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/bookmind123 \
  neo4j:5
```

### 2. Install

```bash
git clone git@github.com:lostmymind-18/Brain.git
cd Brain
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — set your API key and provider
```

```env
# Use Anthropic (default)
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=sk-ant-...

# Or switch to OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

### 4. Run

```bash
PYTHONPATH=. streamlit run bookmind_tutor/tutor/app.py
```

Open [http://localhost:8501](http://localhost:8501), upload a PDF, and start chatting.

---

## Running tests

```bash
pytest                    # all 278 tests
pytest tests/agents/      # agent harness + strategies
pytest tests/evaluation/  # eval framework
```

---

## Evaluation

Run benchmark questions against one or more configs and get a scored report:

```bash
# Copy and edit the example questions file for your book
cp data/eval_questions.example.json data/eval_questions.json

python -m bookmind_tutor.evaluation \
  --book-id <id> \
  --questions data/eval_questions.json \
  --provider openai \
  --model gpt-4o-mini \
  --strategy react
```

Output: markdown table with per-question scores + JSON report saved to `data/eval_reports/`.

---

## Roadmap

| Week | Focus | Status |
|------|-------|--------|
| 1 | PDF ingestion pipeline | Done |
| 2 | Agent Harness (ReAct loop) | Done |
| 3 | Reasoning strategies (Plan-Execute, Reflexion) | Done |
| 4 | Knowledge Graph + GraphRAG | Done |
| 5-6 | Q&A Agent + Streamlit UI | Done |
| 7 | Evaluation framework | Done |
| 8-9 | Safety/guardrails + Observability (OpenTelemetry) | In progress |
| 10-11 | DPO fine-tuning (proof-of-concept) | Planned |
| 12-13 | Integration, polish, portfolio write-up | Planned |

---

## Project context

This is a personal learning project — the goal is to deeply understand each component (agent harness, knowledge graph, evaluation, RL) by building it from scratch, not to ship a product. Code is commented to explain *why*, not just *what*.

The RL component (Week 10-11) is an **offline DPO proof-of-concept** on a small model — not online RL or full RLHF. The interaction logs collected from Week 7 onward provide the seed data for preference pair construction.
