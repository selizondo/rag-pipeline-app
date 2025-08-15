# RAG App

A full-stack AI knowledge assistant built to answer a specific question: **what does it actually take to go from a working RAG script to something you could ship?**

The baseline ([rag-pipeline-from-scratch](../rag-pipeline-from-scratch)) proved the core idea works. This project is about the gap between "it works on my laptop" and "it's observable, testable, and extendable." Every decision here was made to close a specific failure mode — and measured before committing to it.

**Stack:** Python · FastAPI · Streamlit · Chroma · BM25 · sentence-transformers · Claude/Ollama · SQLite

---

## The Problem With the Baseline

The baseline RAG pipeline was a single Python script. It worked. It also had four problems that would block any real use:

1. **Pure vector search fails on exact keywords.** Query "What is LoRA?" — vector search retrieved chunks about "low-rank matrix decomposition" but missed the definition because the acronym doesn't embed close to its expansion. This affects ~15% of realistic queries.

2. **Blocking generation.** The script waited for the full LLM response before printing anything. At 3–5 seconds per query, users stop waiting. Streaming isn't a nice-to-have; it's a UX threshold.

3. **Nothing to observe.** After a query, there was no record of what was retrieved, how long it took, or which chunks scored highest. No way to know if a change to chunk size or retrieval depth helped or hurt.

4. **Logic and display were tangled.** Retrieval and generation lived in the same script as the CLI output. Nothing else could call the pipeline — not a UI, not the eval harness, not an agent. Testing meant running the whole thing.

This project solves all four.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit UI  (ui/app.py · port 8501)                  │
│  ── SSE token stream  ──────────────────────────────►   │
│  ◄── sources + latency metadata ──────────────────────  │
└──────────────────┬──────────────────────────────────────┘
                   │ HTTP / Server-Sent Events
┌──────────────────▼──────────────────────────────────────┐
│  FastAPI  (api/ · port 8000)                            │
│  POST /api/v1/query/stream   ← SSE streaming            │
│  POST /api/v1/query          ← blocking (for evals)     │
│  POST /api/v1/ingest                                    │
│  GET  /api/v1/health                                    │
│  GET  /api/v1/queries        ← observability            │
└─────────────────┬───────────────────────────────────────┘
                  │
       ┌──────────┴───────────┐
       ▼                      ▼
┌─────────────┐        ┌──────────────────────┐
│  Retrieve   │        │  Generate            │
│             │        │                      │
│  BM25 index │        │  Ollama  (local)     │
│  + Chroma   │        │  or Claude API       │
│  vector DB  │        │  (streaming tokens)  │
└──────┬──────┘        └──────────────────────┘
       │
       ▼
┌─────────────────────┐
│  SQLite  (obs.db)   │
│  queries + chunks   │
│  latency per stage  │
└─────────────────────┘
```

The HTTP boundary between the API and UI is the central design decision. `core/` and `api/` are testable without a browser. The eval harness calls the same `/api/v1/query` endpoint a user would — no special instrumentation. Any future client (CLI, agent, mobile) calls the API; the UI doesn't change.

---

## What Changed and Why

| Feature | [rag-pipeline-from-scratch](../rag-pipeline-from-scratch) | This project |
|---------|----------------------------------------------------------|--------------|
| Interface | CLI script | Streamlit UI + FastAPI REST API |
| Search | Vector only | **Hybrid BM25 + vector** — measured improvement |
| Generation | Blocking, full response | **Token-by-token streaming** via SSE |
| Observability | None | SQLite: latency per stage, chunk scores, query log |
| Eval integration | Manual | Wired to [llm-eval-harness](../llm-eval-harness) — one command |
| Testability | Run the whole script | `core/` and `api/` independently testable |
| Deployable | No | Dockerfile + docker-compose |

---

## Decision 1: Hybrid BM25 + Vector Search

Pure vector search has a well-known failure mode on exact keyword queries. The fix — combining BM25 with vector similarity — is standard in production RAG systems. The question is whether it's worth the added complexity for this corpus.

We measured it using the [llm-eval-harness](../llm-eval-harness) before choosing the default:

| α (vector weight) | Accuracy@4 | Hallucination rate | Notes |
|-------------------|------------|-------------------|-------|
| 1.0 (vector only) | 79% | 4% | Good on semantic, misses keywords |
| **0.7 (default)** | **83%** | **3%** | Best overall |
| 0.5 | 80% | 4% | No gain over 0.7 |
| 0.0 (BM25 only) | 61% | 8% | Brittle on paraphrase |

The 4-point Accuracy@4 improvement at α=0.7 justified adding `rank_bm25`. The tradeoff is the BM25 index lives in RAM — all chunk texts, O(n) scoring on every query. At 50k chunks that's ~400 MB and ~2ms overhead. At 1M chunks, you'd replace this with Elasticsearch. See [docs/adr-01-hybrid-search.md](docs/adr-01-hybrid-search.md).

The α slider is exposed in the UI so you can see the difference live.

If metadata or version filters are used, they should be applied before the combined relevance ranking rather than as a post-hoc filter; otherwise stale or irrelevant chunks can still surface at the top. The hybrid search path also serves as the operational fallback when vector-only retrieval misses exact keyword matches.

---

## Decision 2: FastAPI + Streamlit Split

A monolithic Streamlit app was simpler to deploy. It was rejected for one concrete reason: Streamlit re-runs the entire script on every user interaction. That means the embedding model loads, the BM25 index rebuilds, and the Chroma client reconnects on every query. Startup time in the script: 4–6 seconds. Unacceptable.

The FastAPI backend starts once and holds all expensive state in memory for the lifetime of the process. The Streamlit frontend makes HTTP requests — it stays thin and fast.

Secondary benefits: the eval harness, the CLI, and any future agent call the same `/api/v1/query` endpoint without any code duplication. The OpenAPI schema is auto-generated and available at `/docs`. See [docs/adr-02-api-ui-split.md](docs/adr-02-api-ui-split.md).

---

## Decision 3: Observability From Day One

The baseline had no logging. The first sign of a retrieval problem would be a bad answer with no way to investigate.

Every query writes to `obs.db`:
- Total latency, split into retrieval time and generation time
- Which chunks were retrieved, ranked by combined score
- Per-chunk BM25 score, vector score, and combined score

This makes the retrieval decision auditable. When a query returns a wrong answer, you can check whether retrieval found the right chunks (retrieval failure) or whether the LLM had the right context and still got it wrong (generation failure). Those require different fixes.

```sql
-- Where did latency go?
SELECT question, latency_ms, retrieval_ms, generation_ms
FROM queries ORDER BY latency_ms DESC LIMIT 10;

-- What did retrieval actually return for a given query?
SELECT r.rank, r.source, r.combined_score, r.bm25_score, r.vector_score
FROM retrievals r JOIN queries q ON r.query_id = q.id
WHERE q.id = <query_id>;
```

REST endpoint: `GET /api/v1/queries?limit=20`

---

## Decision 4: Chunking at 256 Words

Chunk size is the most consequential parameter in a RAG pipeline and the one most often set by intuition. We ran the experiment using the eval harness before building this:

| Chunk size | Accuracy@4 | Hallucination rate | Why |
|------------|------------|-------------------|-----|
| 128 words | 74% | 5% | Too granular — fragments context |
| **256 words** | **83%** | **3%** | Best balance |
| 512 words | 58% | 17% | Dilutes embedding signal; retrieval becomes imprecise |

The 512-word result is counterintuitive — bigger context in each chunk, but worse answers. The reason: with larger chunks, embeddings average over more content and stop representing any specific concept well. Retrieval fetches chunks that are broadly related rather than specifically relevant, and the LLM fills the remaining gaps with hallucinations.

Chunk size 256, overlap 32 is the configuration used here.

---

## Quick Start

**Runs locally — no GPU required. Ollama is the default LLM.**

```bash
# 1. Set up env (API keys live in workspace master .env)
cp ../career/.env.example ../career/.env
# Default LLM_PROVIDER=ollama — no key needed.
# For Anthropic: uncomment ANTHROPIC_API_KEY and set LLM_PROVIDER=anthropic

# 2. Activate shared venv
source ~/.venvs/newline/bin/activate
# or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 3. Pull Ollama model (if not already pulled)
ollama pull qwen2.5-coder:7b

# 4. Start API
uvicorn api.main:app --reload --port 8000

# 5. Ingest a corpus (new terminal)
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"corpus_dir": "./demo/corpus"}'

# 6. Start UI
streamlit run ui/app.py
```

Open [http://localhost:8501](http://localhost:8501). Try:
- "What is the attention mechanism in transformers?" *(semantic — vector wins)*
- "What is LoRA?" *(acronym — BM25 catches what vector misses)*
- "Explain the bias-variance tradeoff" *(concept — hybrid wins)*

One-command shortcut: `make demo`

---

## Docker

```bash
export ANTHROPIC_API_KEY=sk-ant-...
docker compose up          # builds API + UI containers
make docker-ingest         # loads demo corpus into the running stack
```

API docs at [http://localhost:8000/docs](http://localhost:8000/docs) — full OpenAPI schema, try-it-out UI.

---

## Eval Integration

Requires the [llm-eval-harness](../llm-eval-harness) sibling project and an Anthropic API key for the LLM judge. The harness scores each answer on correctness, groundedness, and conciseness (1–5 each); reports Accuracy@4 and hallucination rate; detects regressions case-by-case.

```bash
# Baseline: record current performance
python evals/run_evals.py \
  --cases ../llm-eval-harness/evals/cases/rag_qa.jsonl \
  --tag rag_app_v1

# After a change (chunk size, α, model swap): compare
python evals/run_evals.py \
  --cases ../llm-eval-harness/evals/cases/rag_qa.jsonl \
  --tag rag_app_v2 \
  --compare <run_id>

make eval-smoke   # 3-case sanity check, ~30 seconds
```

The eval loop is what turned the decisions above from opinions into data.

---

## Where This Breaks (and What to Do)

**BM25 index in RAM:** holds all chunk texts in memory. At 50k chunks (~400 MB) — fine. At 1M chunks (~8 GB) — replace with Elasticsearch. The `HybridRetriever` interface stays stable; only the BM25 backend changes.

**Single Chroma instance:** file-backed, no replication. Fine for single-user demo; at team scale use Chroma's HTTP server mode or migrate to Qdrant.

**Synchronous ingest:** the `/ingest` endpoint blocks until all chunks are embedded. At 50k documents that's minutes. Fix: return a `job_id` immediately, process via Celery in the background.

**Cost at 10k queries/day with Claude Haiku:** ~$375/month uncached, ~$175/month with prompt caching on the context prefix. At 100k queries/day, evaluate self-hosted open models (Llama 3.x via vLLM).

Full analysis in [docs/scale-design.md](docs/scale-design.md) — covers multi-tenancy, async ingest queues, cost projections at 10k and 100k queries/day.

---

## Files

```
rag-pipeline-app/
├── api/
│   ├── main.py          # FastAPI app, lifespan startup (model + BM25 index load)
│   ├── routes.py        # /query, /query/stream, /ingest, /health, /queries
│   └── models.py        # Pydantic request/response schemas
├── core/
│   ├── ingest.py        # Word-based chunking with section header metadata
│   ├── retrieve.py      # HybridRetriever: BM25 + Chroma, α-weighted fusion
│   ├── generate.py      # Streaming generation: Ollama or Anthropic SDK
│   └── pipeline.py      # Orchestration + per-query SQLite logging
├── ui/
│   └── app.py           # Streamlit: SSE consumer, sources panel, α slider
├── evals/
│   └── run_evals.py     # Wraps /api/v1/query as a callable for llm-eval-harness
├── demo/
│   └── corpus/          # ML fundamentals corpus — covers all eval case topics
├── docs/
│   ├── adr-01-hybrid-search.md    # Why BM25+vector, with eval numbers
│   ├── adr-02-api-ui-split.md     # Why FastAPI+Streamlit vs monolith
│   └── scale-design.md            # 100k docs · 1M docs · 10k queries/day
├── Makefile             # make demo · make eval · make docker-up
├── Dockerfile
└── docker-compose.yml
```

---

## What I'd Do With More Time

- **Cross-encoder reranking:** rerank top-20 hybrid results to top-5 using `cross-encoder/ms-marco-MiniLM-L-6-v2`; typically +5–8% Accuracy@4 at ~150ms added latency
- **Async ingest:** `POST /ingest` returns `job_id` immediately; Celery worker processes in background; status at `GET /ingest/{job_id}`
- **Multi-tenant collections:** one Chroma collection per tenant; JWT middleware extracts `tenant_id`; per-tenant delete, access control, usage metering
- **LangFuse integration:** ship latency and judge scores to LangFuse automatically after each eval run — replaces manual SQLite queries with a dashboard
- **Hosted deployment:** FastAPI on Railway, UI on Streamlit Community Cloud, Qdrant Cloud for the vector store — total cost ~$30/month at demo scale
