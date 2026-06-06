# RAG App

![Tests](https://github.com/selizondo/rag-pipeline-app/actions/workflows/test.yml/badge.svg)

Teams ship a RAG prototype that works on whiteboard questions. The real test is production: acronym queries that vector search misses, LLM costs that compound at 10k queries/day, and a wrong answer with no way to know whether retrieval or generation was the cause.

This is the step between "the RAG script works" and "it is observable, testable, and improvable." Every decision was measured before it was made.

**Stack:** Python · FastAPI · Streamlit · Chroma · BM25 · sentence-transformers · Claude/Ollama · SQLite

## Results

Hybrid BM25 + vector search versus the vector-only baseline, measured with the [llm-eval-harness](https://github.com/selizondo/llm-eval-harness):

| Retrieval approach | Accuracy@4 | Hallucination rate |
|--------------------|------------|-------------------|
| Vector only (baseline) | 79% | 4% |
| Hybrid alpha=0.7 (default) | **83%** | **3%** |
| Hybrid alpha=0.5 | 80% | 4% |
| BM25 only | 61% | 8% |

4 percentage points at alpha=0.7. The gain is concentrated on exact-keyword queries: acronyms (LoRA, BM25, RLHF), model names, and specific method names that embed poorly but score high on term frequency.

## How It Works

### Hybrid retrieval because pure vector has a known failure mode

Exact keyword queries score poorly on vector similarity: the acronym "LoRA" does not embed near its expansion "Low-Rank Adaptation." BM25 catches what vector misses. The combination with configurable alpha weighting provides 83% Accuracy@4 versus 79% vector-only. Alpha is exposed as a live slider in the UI so the tradeoff is observable at runtime, not just in docs.

### FastAPI + Streamlit split because Streamlit re-runs on every interaction

A monolithic Streamlit app re-runs the entire script on every user interaction: embedding model loads, BM25 index rebuilds, Chroma client reconnects. Startup time in the script: 4 to 6 seconds per query. The FastAPI backend starts once and holds all expensive state in memory for the lifetime of the process. The Streamlit frontend makes HTTP requests and stays thin. Secondary benefit: the eval harness, the CLI, and any future agent call the same `/api/v1/query` endpoint with no code duplication.

### Observability is the mechanism for validating changes

Every query writes to SQLite: latency split into retrieval and generation stages, which chunks were retrieved, per-chunk BM25 score, vector score, and combined score. When a query returns a wrong answer, you can distinguish a retrieval failure (right chunks not returned) from a generation failure (right chunks, still wrong answer). Those require different fixes. Without this distinction, changes are guesses.

```sql
-- Where did latency go?
SELECT question, latency_ms, retrieval_ms, generation_ms
FROM queries ORDER BY latency_ms DESC LIMIT 10;
```

**Companion post:** "From Prototype to Production: Hybrid RAG" (AI Systems in Production series, coming soon)
**Related projects:** [rag-pipeline-from-scratch](https://github.com/selizondo/rag-pipeline-from-scratch) (72% Accuracy@4 vector-only baseline this system extends) · [llm-eval-harness](https://github.com/selizondo/llm-eval-harness) (harness that produced the alpha=0.7 measurement) · [llm-drift-monitor](https://github.com/selizondo/llm-drift-monitor) (drift monitor for catching production degradation between releases)

---

## Go Deeper

| Audience | Doc |
|----------|-----|
| Business and product context | [Product and Cost](docs/product.md) |
| Running the code | [Setup and Usage](docs/setup.md) |
| Engineering decisions | [Design and Tradeoffs](docs/engineering.md) |
| What breaks and why | [Failure Modes](docs/failures.md) |
