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

**Alternatives considered:** FastAPI + Next.js (overkill for this scope, requires JS build pipeline); FastAPI + Gradio (less control over layout).

**Tradeoff:** Two processes to manage locally. Mitigated by `make dev`. SSE over HTTP adds ~5ms per request — imperceptible.

**Scale boundary:** At single-instance scale, shared `obs.db` and `chroma_db` via volume is fine. Multi-replica deployment requires care: move to PostgreSQL + Qdrant, add tenant isolation.

---

## ADR-03: Chunking Strategy — Word-Based vs Sentence-Boundary

**Decision:** Word-based chunking with fixed size (256 words) and overlap (32 words). No sentence-awareness.

**Current approach (`core/ingest.py → _word_chunks`):**
```python
words = text.split()           # naive whitespace split
chunk = " ".join(words[start:end])
start += chunk_size - overlap
```

**Alternative considered:** Sentence-boundary chunking via spaCy (`nlp(text).sents`) or NLTK (`sent_tokenize`). Preserves sentence integrity so every chunk is a semantically complete unit.

**Why word-based was chosen:**
1. No additional NLP dependency (spaCy model download ~12MB; NLTK punkt data).
2. Chunk sizes are strictly predictable — simplifies BM25 index memory estimation (400MB at 50k chunks).
3. Demo corpus is short-form Q&A paragraphs where mid-sentence splits are infrequent.

**Known failure mode:** Word-based can split mid-sentence, burying key content between two chunks. Example: "LoRA stands for Low-Rank [chunk boundary] Adaptation" yields two chunks neither of which fully defines LoRA. Sentence-boundary chunking eliminates this class of failure.

**Measurement plan:** Compare faithfulness and context_recall between strategies using RAGAS once the eval pipeline is operational:
```
make eval-save   # word-based baseline → artifacts/eval/latest_run.json
# switch core/ingest.py to sentence-boundary chunker
make eval-save   # compare metrics
```
Hypothesis: sentence-boundary improves faithfulness on long-document corpora (>500 words/doc); impact is smaller on this short-form Q&A corpus.

**Overhead of sentence-boundary:** spaCy `en_core_web_sm` adds ~150ms/doc at N=10k docs (15 min extra ingest time). Acceptable at batch-ingest scale; negligible at demo scale.

**Status:** Not yet measured on this corpus. Open measurement task tracked in `docs/STAFF_REVIEW.md`.

---

## Synchronous Ingest Endpoint

**Decision:** `POST /api/v1/ingest` blocks until all chunks are embedded and stored.

**Why:** Demo-scale corpus (hundreds of files). Embedding completes in seconds; request timeout is not a concern.

**Scale boundary:** At 50k+ chunks, embedding takes 30–120 seconds — request timeout. Fix: accept ingest → return `job_id` immediately → process via Celery + Redis → poll `GET /api/v1/ingest/{job_id}`.

---

## SQLite for Observability

**Decision:** Single `obs.db` SQLite file for query logging.

**Why:** Zero infrastructure. Self-contained, portable, queryable with standard SQL. Sufficient for single-user demo and development analysis.

**Scale boundary — single writer:** SQLite serialises all writes through a file lock. A single FastAPI instance is fine (writes are short: one INSERT per query). The breakpoint is **horizontal scaling**: two or more FastAPI replicas writing concurrently will hit `database is locked` errors at ~100+ concurrent writers. Fix: move to PostgreSQL with asyncpg + SQLAlchemy async (`async_engine = create_async_engine("postgresql+asyncpg://...")`). The `_conn()` context manager in `core/pipeline.py` is the only write site — swap the driver there. At >10k queries/day, supplement with Datadog/Honeycomb for real-time alerting.

**Inline pointer:** See `core/pipeline.py → _conn()` for the single write site to replace.

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
