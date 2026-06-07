# Setup and Usage

## Key Concepts

**Hybrid BM25 + vector search:** Two complementary retrieval strategies. BM25 uses exact keyword matching (fast, precise on technical terms like "LoRA"). Vector search uses semantic similarity (good for paraphrases and concepts). Hybrid fusion combines both — a query "What is LoRA?" hits BM25 on the exact acronym, while vector search finds semantically related chunks. This project weights vector 70% + BM25 30% (α=0.7) and measures the tradeoff with evals.

**Alpha weighting:** The hyperparameter controlling hybrid fusion balance. α=1.0 means vector-only (semantic), α=0.0 means BM25-only (keyword). This project exposes α as a slider in the UI so users see the difference live — dense retrieval vs hybrid vs keyword-based, all on the same query.

**Streaming (SSE):** Server-Sent Events — the server sends tokens to the browser as they arrive, instead of waiting for the full response. Token-by-token streaming is the UX threshold for LLM applications. Without it, users see a blank screen for 3–5 seconds before any output. Implemented via FastAPI `StreamingResponse` and consumed by the Streamlit frontend.

**Observability from day one:** Every query writes to SQLite with latency breakdown (retrieval time, generation time) and chunk scores (BM25, vector, combined). This is not optional instrumentation — it's the mechanism for measuring whether changes help or hurt. No observability = no way to validate improvements.

**Chunking at 256 words:** A measured decision, not a convention. This project ran chunk-size experiments (128, 256, 512 words) against 20 eval queries. 256 words maximizes retrieval precision; 512 dilutes embeddings and hurts quality. The experiment methodology transfers to any new corpus.

---

## Prerequisites

- Python 3.10+
- Ollama running locally with `qwen2.5-coder:7b` (default LLM, no API key needed)
- For Anthropic: `ANTHROPIC_API_KEY` in `.env`

## Quick Start

```bash
# 1. Configure
cp .env.example .env
# Default: LLM_PROVIDER=ollama (no key needed)
# For Anthropic: set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY

# 2. Install
pip install -r requirements.txt

# 3. Pull Ollama model
ollama pull qwen2.5-coder:7b

# 4. Start API (Terminal 1)
uvicorn api.main:app --reload --port 8000

# 5. Ingest corpus (Terminal 2)
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"corpus_dir": "./demo/corpus"}'

# 6. Start UI (Terminal 3)
streamlit run ui/app.py
```

Open http://localhost:8501. Try:
- "What is LoRA?" (exact acronym: BM25 wins over vector)
- "Explain the attention mechanism" (semantic: vector wins)
- "What is the bias-variance tradeoff?" (concept: hybrid wins)

Shortcut: `make demo` runs steps 1 to 6 automatically.

## Docker

```bash
export ANTHROPIC_API_KEY=sk-ant-...
docker compose up
make docker-ingest   # loads demo corpus into the running stack
```

API docs: http://localhost:8000/docs (full OpenAPI schema, try-it-out UI).

## Eval Integration

Requires [llm-eval-harness](https://github.com/selizondo/llm-eval-harness) sibling repo and `ANTHROPIC_API_KEY` for the judge.

```bash
# Baseline: record current performance
python evals/run_evals.py \
  --cases ../llm-eval-harness/evals/cases/rag_qa.jsonl \
  --tag rag_app_v1

# After a change: compare
python evals/run_evals.py \
  --cases ../llm-eval-harness/evals/cases/rag_qa.jsonl \
  --tag rag_app_v2 \
  --compare <run_id>

make eval-smoke    # 3-case sanity check, ~30 seconds
```

## Sample API Response

```json
{
  "question": "What is LoRA?",
  "answer": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique...",
  "sources": [
    {
      "chunk": "LoRA stands for Low-Rank Adaptation...",
      "source": "ml_interview_qa.md",
      "bm25_score": 8.432,
      "vector_score": 0.78,
      "combined_score": 0.824
    }
  ],
  "latency_ms": {"retrieval": 45, "generation": 2340}
}
```

## Code Layout

```
rag-pipeline-app/
├── api/
│   ├── main.py          # FastAPI app, lifespan startup (model + BM25 index load)
│   ├── routes.py        # /query, /query/stream, /ingest, /health, /queries
│   └── models.py        # Pydantic request/response schemas
├── core/
│   ├── ingest.py        # Word-based chunking with section header metadata
│   ├── retrieve.py      # HybridRetriever: BM25 + Chroma, alpha-weighted fusion
│   ├── generate.py      # Streaming generation: Ollama or Anthropic SDK
│   └── pipeline.py      # Orchestration + per-query SQLite logging
├── ui/
│   └── app.py           # Streamlit: SSE consumer, sources panel, alpha slider
├── evals/
│   └── run_evals.py     # Wraps /api/v1/query as callable for llm-eval-harness
├── demo/
│   └── corpus/          # ML fundamentals corpus covering all eval case topics
├── Makefile             # make demo, make eval, make docker-up
├── Dockerfile
└── docker-compose.yml
```

## Observability Queries

```sql
-- Where did latency go?
SELECT question, latency_ms, retrieval_ms, generation_ms
FROM queries ORDER BY latency_ms DESC LIMIT 10;

-- What did retrieval return for a given query?
SELECT r.rank, r.source, r.combined_score, r.bm25_score, r.vector_score
FROM retrievals r JOIN queries q ON r.query_id = q.id
WHERE q.id = <query_id>;
```

REST endpoint: `GET /api/v1/queries?limit=20`
