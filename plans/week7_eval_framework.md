# Plan: Week 7 - Evaluation Framework

## Scope (what this is)

Answer quality evaluation: does the tutor give good answers?
- **LLM-as-judge evaluator** - score individual answers on defined rubric
- **Interaction logger** - JSONL log of every Q&A turn (doubles as DPO data for Week 10-11)
- **Eval runner** - run benchmark questions against one or more configs, produce a report

Not in scope (removed from original roadmap): quiz generation, engagement metrics, any measure of whether the user "learned deeply."

## Why this order

The provider abstraction (see `plans/llm_provider_abstraction.md`) should be implemented first because:
- The eval runner will want to test multiple providers/models
- The evaluator itself needs an LLM client - better to use the abstraction from day one

## Design

### 1. Interaction Logger (`bookmind_tutor/evaluation/logger.py`)

Appends one JSONL record per completed Q&A turn to `data/interactions.jsonl`.
The format is designed to be directly usable as DPO preference data (Week 10-11).

```python
@dataclass
class InteractionRecord:
    record_id: str                  # uuid4
    timestamp: str                  # ISO-8601 UTC
    book_id: str
    book_name: str
    question: str
    retrieved_chunks: list[str]     # chunk texts passed to context (for DPO)
    answer: str
    strategy: str                   # "react" | "plan_execute" | "reflexion"
    model: str                      # e.g. "gpt-4o-mini"
    provider: str                   # "openai" | "anthropic"
    num_llm_calls: int
    elapsed_seconds: float
    # Populated later by evaluator or human feedback:
    eval_score: float | None        # 0.0 - 1.0
    eval_breakdown: dict | None     # {"factual_accuracy": 0.9, ...}
    human_rating: int | None        # 1-5 stars from user (future)
```

**Writing:** `InteractionLogger.log(record: InteractionRecord)` - opens file in append mode, writes one JSON line, flushes.
**Reading:** `InteractionLogger.load_all()` - reads all lines, skips malformed (fault-tolerant).

The log file is append-only. To update a record (e.g. add eval_score later), write a new record with the same `record_id` and a later timestamp. The eval runner uses the latest record per `record_id`.

### 2. Answer Quality Evaluator (`bookmind_tutor/evaluation/evaluator.py`)

A single LLM call that scores one answer against the source chunks that were retrieved.

**Rubric (3 dimensions, each 0.0 - 1.0):**

| Dimension | Description |
|---|---|
| `factual_accuracy` | Are claims in the answer supported by the retrieved chunks? |
| `citation_quality` | Does the answer cite or reference specific parts of the book? |
| `depth_and_relevance` | Does the answer directly address the question with appropriate depth? |

**Prompt design:**

```
You are evaluating the quality of an answer given by an AI book tutor.

QUESTION: {question}

RETRIEVED CONTEXT (from the book):
---
{chunks joined by \n---\n}
---

ANSWER:
{answer}

Rate the answer on three dimensions from 0.0 to 1.0.
Return ONLY valid JSON:
{
  "factual_accuracy": <float>,
  "citation_quality": <float>,
  "depth_and_relevance": <float>,
  "reasoning": "<one sentence explanation>"
}
```

**Implementation:**
```python
@dataclass
class EvalResult:
    factual_accuracy: float
    citation_quality: float
    depth_and_relevance: float
    overall: float       # simple average of the three
    reasoning: str
    model_used: str      # which model did the judging

class AnswerEvaluator:
    def __init__(self, client: LLMClient) -> None: ...
    def evaluate(self, question: str, chunks: list[str], answer: str) -> EvalResult: ...
```

The evaluator uses `client.complete()` with `max_tokens=256` and parses JSON from the response. If parsing fails, retries once with a stricter prompt. If it fails again, returns a sentinel `EvalResult` with all scores = -1.0 and a note in `reasoning`.

The evaluator client can be a different (cheaper) model than the tutor client. The runner configures this separately.

### 3. Eval Runner (`bookmind_tutor/evaluation/runner.py`)

Runs a set of benchmark questions against one or more `(provider, model, strategy)` configs and writes a summary report.

**Benchmark questions (`data/eval_questions.json`):**
```json
{
  "version": 1,
  "book_id": "...",
  "questions": [
    {
      "id": "q1",
      "question": "What does the author say about X?",
      "expected_keywords": ["X", "Y"],    # optional, for sanity check
      "difficulty": "factual"             # "factual" | "conceptual" | "synthesis"
    }
  ]
}
```

**Runner config:**
```python
@dataclass
class RunConfig:
    provider: str
    model: str
    strategy: str
    label: str    # human-readable name for the report column
```

**Runner API:**
```python
class EvalRunner:
    def run(
        self,
        book_id: str,
        questions: list[dict],
        configs: list[RunConfig],
        evaluator: AnswerEvaluator,
    ) -> EvalReport: ...
```

For each (question, config) pair:
1. Call QAAgent with the config's provider/model/strategy
2. Get back answer + trace
3. Log the interaction via `InteractionLogger`
4. Call `AnswerEvaluator.evaluate()` with question + retrieved chunks + answer
5. Update the log record with the eval score
6. Collect into `EvalReport`

**EvalReport:**
```python
@dataclass
class EvalReport:
    timestamp: str
    book_id: str
    configs: list[RunConfig]
    rows: list[EvalRow]    # one per (question, config)

@dataclass
class EvalRow:
    question_id: str
    question: str
    config_label: str
    answer: str
    factual_accuracy: float
    citation_quality: float
    depth_and_relevance: float
    overall: float
    elapsed_seconds: float

def to_markdown_table(self) -> str: ...
def to_csv(self) -> str: ...
def summary_by_config(self) -> dict[str, dict[str, float]]: ...
    # {"gpt-4o-mini / react": {"factual_accuracy": 0.82, ...}, ...}
```

Report is also written to `data/eval_reports/YYYYMMDD_HHMMSS.json`.

### 4. Integration with app.py

**Interaction logging (always-on):**
After every successful Q&A turn in the chat loop, `app.py` logs the interaction:
```python
if answer and trace:
    logger.log(InteractionRecord(
        record_id=str(uuid4()),
        book_id=active_book_id,
        book_name=...,
        question=user_input,
        retrieved_chunks=trace.retrieved_chunks,   # new field on AgentTrace
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
```

`AgentTrace` needs a new field `retrieved_chunks: list[str]` - populated by `ReActStrategy` when it executes the `book_search` tool.

**Eval runner is CLI-only (no Streamlit page):**
Run via: `python -m bookmind_tutor.evaluation.runner --book-id <id> --questions data/eval_questions.json`
This keeps the app focused on the tutor experience, not the eval harness.

### 5. Directory layout

```
bookmind_tutor/evaluation/
  __init__.py
  logger.py         # InteractionRecord, InteractionLogger
  evaluator.py      # AnswerEvaluator, EvalResult
  runner.py         # EvalRunner, RunConfig, EvalReport, EvalRow
  __main__.py       # CLI entry point for the runner

data/
  interactions.jsonl          # append-only interaction log
  eval_questions.json         # benchmark questions (hand-curated per book)
  eval_reports/               # one JSON per runner invocation

tests/evaluation/
  test_logger.py              # round-trip, fault-tolerance, dedup by record_id
  test_evaluator.py           # JSON parsing, retry logic (mock LLM client)
  test_runner.py              # end-to-end with mock agent + mock evaluator
```

`.gitignore` additions: `data/interactions.jsonl`, `data/eval_reports/`

## Implementation order

1. `AgentTrace.retrieved_chunks` field - small change to `agents/models.py`, populated in `react.py`
2. `InteractionLogger` + tests - no LLM dependency, implement right away
3. `AnswerEvaluator` + tests - depends on `LLMClient` (needs provider abstraction done first)
4. `EvalReport` / `EvalRunner` + CLI - depends on evaluator
5. Integration in `app.py` - logging only (evaluator runs separately via CLI)

## Dependencies

- Provider abstraction must be done first (evaluator uses `LLMClient`)
- `openai` package must be installed (for the default eval judge model, e.g. `gpt-4o-mini`)
- No new Streamlit pages needed
