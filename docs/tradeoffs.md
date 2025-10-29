# Architectural Tradeoffs

Decisions made during build, with the reasoning and explicit scale/complexity boundaries.

---

## Hybrid BM25 + Vector Search

**Decision:** Combine BM25 (lexical) and vector (semantic) scores with configurable linear interpolation:
```
score(d, q) = α · vector_score(d, q) + (1 − α) · bm25_score(d, q)
```
Default `α = 0.7` (70% vector, 30% BM25), tunable per-request via API.

**Why:** Pure vector search has a known failure mode on exact keyword lookups — acronyms, model names, error codes. Two concrete cases: "What is LoRA?" retrieved chunks about low-rank matrix decomposition but missed the literal LoRA definition; "BM25 vs TF-IDF" ranked exact-match content below semantically similar IR content.

**Measured result:** α=1.0 (pure vector): 79% Accuracy@4, 4% hallucination. α=0.7 (hybrid): 83% Accuracy@4, 3% hallucination. α=0.0 (pure BM25): 61% Accuracy@4, 8% hallucination.

**Alternatives considered and rejected:**
- SPLADE/ColBERT — state-of-the-art recall but 5–10× slower, requires separate model inference
- Cross-encoder reranking — adds 200–500ms per query; reserved for future optimization

**Scale boundary:** BM25 holds all chunks in RAM (`rank_bm25`). At 50k chunks × 256 words: ~400MB, acceptable for demo. At 1M chunks: ~8GB RAM — replace with Elasticsearch/OpenSearch (inverted index, O(log n) lookup vs O(n) scan). Keep `HybridRetriever` interface stable; swap the BM25 backend only.

---

## FastAPI Backend + Streamlit Frontend Split

**Decision:** Two independent processes — FastAPI owns all AI logic; Streamlit is a pure presentation layer making HTTP requests.

**Why:** A monolith (everything in Streamlit) can't be tested independently, restarts the embedding model on every Streamlit rerun, and locks out any other client. The split makes `core/` and `api/` testable without a browser; any future client (eval harness, CLI, agent) calls the same `/api/v1/query` endpoint.

**Communication:** HTTP/SSE — backend streams tokens via Server-Sent Events; frontend consumes in streaming mode.

**Alternatives considered:** FastAPI + Next.js (overkill for portfolio, requires JS build pipeline); FastAPI + Gradio (less control over layout).

**Tradeoff:** Two processes to manage locally. Mitigated by `make dev`. SSE over HTTP adds ~5ms per request — imperceptible.

**Scale boundary:** At portfolio scale (1 instance), shared `obs.db` and `chroma_db` via volume is fine. Multi-replica deployment requires care: move to PostgreSQL + Qdrant, add tenant isolation.

---

## Word-Based Chunking (256 words, 32-word overlap)

**Decision:** Split on whitespace boundaries; no sentence-awareness.

**Why:** Simple, predictable, no NLP pipeline dependency. For a demo corpus of ML concepts, sentences are generally short enough that mid-sentence splits are rare.

**Tradeoff:** Not evaluated against sentence-boundary chunking. The rag-pipeline-from-scratch and rag-ragas-eval projects show sentence-boundary chunking improves context_recall on long documents. This corpus is short-form Q&A so the gap is likely smaller.

**Documented gap:** No measurement of word-based vs sentence-boundary on this corpus.

---

## Synchronous Ingest Endpoint

**Decision:** `POST /api/v1/ingest` blocks until all chunks are embedded and stored.

**Why:** Demo-scale corpus (hundreds of files). Embedding completes in seconds; request timeout is not a concern.

**Scale boundary:** At 50k+ chunks, embedding takes 30–120 seconds — request timeout. Fix: accept ingest → return `job_id` immediately → process via Celery + Redis → poll `GET /api/v1/ingest/{job_id}`.

---

## SQLite for Observability

**Decision:** Single `obs.db` SQLite file for query logging.

**Why:** Zero infrastructure. Self-contained, portable, queryable with standard SQL. Sufficient for single-user demo and development analysis.

**Scale boundary:** SQLite is single-writer. Multi-instance deployments (load-balanced FastAPI) require PostgreSQL. At >10k queries/day, ship logs to Datadog/Honeycomb for real-time alerting.

---

## Scale Boundaries Summary

| Component | Demo limit | Upgrade path |
|---|---|---|
| BM25 in RAM | ~50k chunks (~400MB) | Elasticsearch / OpenSearch |
| Chroma embedded | ~50k chunks, 1 instance | Chroma HTTP mode → Qdrant / Weaviate |
| Sync ingest | <1k files | Celery + Redis async job queue |
| SQLite obs.db | 1 writer | PostgreSQL |
| LLM cost (Haiku) | $0 at demo scale | $375/month at 10k q/day uncached; $175/month at 60% cache hit |
| Ollama local | 1 concurrent user | vLLM or managed endpoint for multi-user |

---

## What Was Cut

| Cut | Reason | Upgrade trigger |
|---|---|---|
| Cross-encoder reranking | +200–500ms per query; overkill at demo scale | Production latency budget allows it |
| Sentence-boundary chunking | Word-based is simpler; corpus is short-form Q&A | Long-document corpus where resolution content is buried |
| Multi-tenant auth | Single-user demo | Real product with multiple users |
| Next.js frontend | Requires JS build pipeline; Streamlit sufficient for demo | Production-grade UI needed |
