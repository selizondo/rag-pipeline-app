# Design and Tradeoffs

Decisions made during build, with evaluation evidence, alternatives considered, and the scale boundary where each choice breaks down.

---

## ADR-01: Hybrid BM25 + Vector Search

**Decision:** Combine BM25 (lexical) and vector (semantic) scores with configurable linear interpolation:

```
score(d, q) = alpha * vector_score(d, q) + (1 - alpha) * bm25_score(d, q)
```

Default alpha=0.7 (70% vector, 30% BM25), tunable per-request via API and UI slider. BM25 index built at startup from all Chroma chunks; rebuilt automatically after `/ingest`.

**Why:** Pure vector search has a well-known failure mode on exact keyword queries. Acronyms, model names, API names, and specific method signatures embed poorly because they depend on token frequency, not semantic closeness. Two concrete cases:
- "What is LoRA?" retrieved chunks about "low-rank matrix decomposition" but missed the chunk defining LoRA (Low-Rank Adaptation): the acronym did not embed near its expansion.
- "BM25 vs TF-IDF" retrieved semantically similar IR content but ranked below the exact-match chunk.

**Eval evidence (llm-eval-harness):**

| alpha | Accuracy@4 | Hallucination rate |
|-------|------------|-------------------|
| 1.0 (vector only) | 79% | 4% |
| 0.7 (default) | 83% | 3% |
| 0.5 | 80% | 4% |
| 0.0 (BM25 only) | 61% | 8% |

4pp improvement at alpha=0.7 justified adding `rank_bm25`. The UI alpha slider makes this tradeoff observable at runtime.

**Alternatives rejected:**

| Option | Reason rejected |
|--------|----------------|
| Pure vector (status quo) | ~15% of eval cases fail on exact keyword queries |
| SPLADE/ColBERT sparse-dense fusion | Requires separate model inference, 5 to 10x slower |
| Cross-encoder reranking | Adds 200 to 500ms per query; reserved for future optimization |

**Tradeoffs:**
- BM25 index holds all chunk texts in RAM. At 50k chunks x 256 words: ~400MB. Acceptable at demo scale.
- O(n) BM25 scoring on every query. At 10k chunks: ~2ms. At 1M chunks: ~200ms, needs approximation.

**Scale boundary:** At 1M+ chunks, replace in-process `rank_bm25` with Elasticsearch or OpenSearch. Keep the `HybridRetriever` interface stable: only the BM25 backend changes.

---

## ADR-02: FastAPI Backend + Streamlit Frontend Split

**Decision:** Two independent processes: FastAPI owns all AI logic; Streamlit is a pure presentation layer making HTTP requests.

**Why:** Streamlit's execution model re-runs the entire script on every user interaction. That means the embedding model loads, the BM25 index rebuilds, and the Chroma client reconnects on every query. Startup time in a monolithic script: 4 to 6 seconds per query. The FastAPI backend starts once, holds all expensive state in memory for the lifetime of the process, and responds to queries in 45 to 500ms.

**Secondary benefits:**
- `core/` and `api/` are independently testable without a browser
- The eval harness, CLI, and any future agent call the same `/api/v1/query` endpoint with no code duplication
- OpenAPI schema auto-generated at `/docs`; machine-readable contract for API consumers
- Two-container Docker deployment is clean: one container per service, shared volume for Chroma and SQLite state

**Communication:** HTTP and Server-Sent Events. Backend streams tokens via SSE; frontend consumes in streaming mode. Added latency: ~5ms per request. Imperceptible in practice.

**Alternatives rejected:**

| Option | Reason rejected |
|--------|----------------|
| Monolith (everything in Streamlit) | Embedding model restarts on every rerun, no testable API |
| FastAPI + Next.js | Overkill for this scope; requires JS build pipeline |
| FastAPI + Gradio | Less layout control, weaker auth story |

---

## Chunking at 256 Words

**Decision:** 256-word chunks with 32-word overlap.

**Eval evidence:**

| Chunk size | Accuracy@4 | Hallucination rate | Why |
|------------|------------|-------------------|-----|
| 128 words | 74% | 5% | Too granular: fragments context |
| 256 words | 83% | 3% | Best balance |
| 512 words | 58% | 17% | Dilutes embedding: retrieval becomes imprecise |

The 512-word result is counterintuitive. Larger chunks average over more content, so embeddings stop representing any specific concept precisely. Retrieval returns broadly related chunks, and the LLM fills gaps with hallucinations.

**Reusable methodology:** Run this experiment on any new corpus before committing to a chunk size. The template: define the eval metric, sweep sizes (128, 256, 512), measure. The result will be corpus-specific; the measurement discipline transfers.

---

## Observability as First-Class Architecture

**Decision:** Every query writes to SQLite: total latency, retrieval latency, generation latency, chunks retrieved, and per-chunk scores (BM25, vector, combined).

**Why:** Without query-level observability, changes to retrieval configuration (alpha, chunk size, retrieval depth) are guesses. With it, a wrong answer can be diagnosed as a retrieval failure (right chunks not returned) or a generation failure (right chunks, still wrong answer). These require different fixes. The observability schema enables this distinction.

**Production upgrade path:** Ship logs to Datadog or Honeycomb for latency histograms; use Langfuse or Arize for LLM-specific trace-level observability. SQLite is sufficient at single-user demo scale.

---

## Scale Boundaries

| Bottleneck | Demo limit | Production fix |
|------------|-----------|----------------|
| BM25 in RAM | ~50k chunks (~400MB) | Elasticsearch or OpenSearch |
| Single Chroma instance | Single user, no replication | Chroma HTTP server, then Qdrant or Weaviate |
| Synchronous ingest | 50k docs = minutes blocked | Return job_id, process via Celery + Redis |
| SQLite observability | Single user | Postgres, ship to Datadog or Honeycomb |
| Multi-tenancy | Single shared corpus | tenant_id metadata filter or per-tenant Chroma collection |

See docs/product.md for the full cost model at 10k and 100k queries/day.

---

## What Was Cut

| Cut | Reason | Upgrade trigger |
|-----|--------|-----------------|
| Cross-encoder reranking | 200 to 500ms added latency | Quality plateau on hybrid, latency SLA allows it |
| Async ingest | Complexity for demo scope | Corpus size makes synchronous ingest time out |
| Multi-tenant collections | Single user for demo | Shared deployment with multiple users |
| Langfuse integration | Separate task (stashed) | LLM-specific trace observability needed |
| Hosted deployment | Out of scope for portfolio | Demo available to external reviewers |
