# Local Observability Setup

This guide shows how to run Jaeger locally and view distributed traces from BookMind Tutor.

## Prerequisites

- Docker Desktop installed and running
- BookMind Tutor running locally (`streamlit run bookmind_tutor/tutor/app.py`)

## Start Jaeger

Jaeger is the easiest all-in-one backend for local trace collection.
It listens on port 4317 for OTLP/gRPC (what we send) and port 16686 for the UI.

```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest
```

Verify it started:
```bash
docker logs jaeger
# Should end with: "Starting HTTP server on 0.0.0.0:16686"
```

## Configure BookMind to Export Traces

Set these environment variables before starting the app:

```bash
export OTEL_EXPORT=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

PYTHONPATH=. streamlit run bookmind_tutor/tutor/app.py
```

Or add them to your `.env` file:

```
OTEL_EXPORT=otlp
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

## View Traces

Open the Jaeger UI at http://localhost:16686.

1. Select **Service: bookmind-tutor** from the dropdown.
2. Click **Find Traces**.
3. Click on any trace to expand the span tree.

### What you will see

A complete question-answer trace looks like this:

```
qa.turn  (top-level span)
  agent.react_turn
    agent.react_step [step=1]
      gen_ai.complete  [gen_ai.system=anthropic, input_tokens=..., output_tokens=...]
      tool.execute     [tool.name=book_search, result_chars=...]
    agent.react_step [step=2]
      gen_ai.complete  [input_tokens=..., output_tokens=...]
```

### Useful span attributes

| Span | Attribute | Meaning |
|------|-----------|---------|
| `gen_ai.complete` | `gen_ai.usage.input_tokens` | Token cost per LLM call |
| `gen_ai.complete` | `gen_ai.request.model` | Which model was called |
| `tool.execute` | `tool.result_chars` | How much text was retrieved |
| `agent.react_turn` | `agent.total_llm_calls` | Total calls in this turn |
| `qa.turn` | `qa.strategy_selected` | ReAct / Plan-Execute / Reflexion |

## Stop Jaeger

```bash
docker stop jaeger && docker rm jaeger
```

## Alternative: Console exporter (default)

If `OTEL_EXPORT` is not set, spans are printed to stdout in the terminal where
you started Streamlit.
Each span appears as a JSON block after the context manager exits.
This is the default - no Docker needed.

## Production: Grafana Tempo + Loki

For a production-grade setup, replace Jaeger with Grafana Tempo (traces) + Loki (logs).
Use the official Grafana LGTM stack Docker Compose:

```bash
git clone https://github.com/grafana/docker-otel-lgtm.git
cd docker-otel-lgtm
docker compose up
```

Then set:
```
OTEL_EXPORT=otlp
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
LOG_FORMAT=json
```

Open Grafana at http://localhost:3000 (admin/admin).
Traces appear in the "Tempo" data source; logs in "Loki".
Trace IDs in structured logs link automatically to spans in Tempo.
