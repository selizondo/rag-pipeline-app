# System Design: RAG at Scale

This document addresses: how would you evolve this system to handle 100k → 1M documents, multiple tenants, and 10k queries/day?

---

## Current Architecture (Demo Scale)

```
Client → Streamlit UI → FastAPI (1 instance) → [BM25 index in RAM + Chroma on disk] → Ollama/Claude
                                                          ↓
                                                    SQLite (obs.db)
```

Works up to: ~50k chunks, single user, <1k queries/day.

---

## Scale Bottlenecks and Solutions

### 1. BM25 Index in RAM

**Bottleneck:** BM25 holds all chunk texts in RAM. At 1M chunks × 256 words: ~8 GB RAM.

**Solution:**
- Replace in-process `rank_bm25` with **Elasticsearch** or **OpenSearch** — proven inverted index at billion-document scale
- BM25 query goes from O(n) RAM scan → O(log n) index lookup
- ElasticSearch's `bool/should` query with `match` + `knn` supports hybrid search natively since v8.x

**Migration path:** keep the `HybridRetriever` interface stable; swap the BM25 backend without changing the API or pipeline.

### 2. Single Chroma Instance

**Bottleneck:** Chroma runs embedded (in-process) with file-backed storage. No horizontal scale.

**Solution:**
- **Short term:** Chroma's [HTTP server mode](https://docs.trychroma.com/usage-guide#running-chroma-in-clientserver-mode) — separate process, same API
- **Medium term:** Migrate to **Qdrant** or **Weaviate** — purpose-built vector DBs with replication, partitioning, and filtering
- **Long term:** For 100M+ vectors, consider **Pinecone** (managed) or **pgvector** (if already on Postgres) for cost-at-scale reasons

### 3. Multi-Tenant Collections

**Current:** Single `rag_corpus` collection shared by all users.

**Solution:**
- Add `tenant_id` to metadata on all chunks
- Filter on `tenant_id` at retrieval time: `collection.query(..., where={"tenant_id": user_id})`
- For full isolation: one Chroma collection per tenant — enables per-tenant delete, access control, and usage metering
- Auth: JWT middleware in FastAPI, tenant extracted from token, injected into retrieval calls

### 4. Async Ingest

**Current:** Ingest is synchronous — the HTTP request blocks until all chunks are embedded and stored.

**Bottleneck:** Embedding 50k chunks takes 30–120 seconds. Request timeout.

**Solution:**
- Accept ingest request → return a `job_id` immediately
- Process in background via **Celery** + Redis (or Dramatiq for simpler setup)
- Poll status at `GET /api/v1/ingest/{job_id}`
- For streaming ingestion (live documents), use a Kafka consumer pattern

### 5. LLM Latency and Cost at 10k Queries/Day

At 10k queries/day with Claude Haiku (`claude-haiku-4-5` at $1.00/$5.00 per 1M tokens):
- Average query: ~500 input tokens (context + question) + ~150 output tokens
- Daily cost: 10k × (500 × $0.001 + 150 × $0.005) / 1000 ≈ **$12.50/day**, $375/month

**Levers:**
- **Prompt caching:** 5-chunk context is largely stable — cache the system prompt + static context prefix with `cache_control` breakpoints. Saves ~60% of input token cost on cache hits.
- **Tiered routing:** route simple factual queries to Haiku, complex multi-step reasoning to Sonnet. Detect via query length + keyword heuristics.
- **Answer caching:** for high-frequency identical questions (FAQ patterns), cache the last answer in Redis with TTL = 1 hour. Hit rate on typical knowledge bases: 15–30%.

### 6. Observability

**Current:** SQLite `obs.db` with queries + retrievals tables.

**Production:**
- Ship logs to **Datadog** or **Honeycomb** — latency histogram by query type, p99 tail tracking
- **LangFuse** or **Arize** for LLM-specific observability: token usage, judge scores, retrieval quality over time
- Alert on: hallucination_rate > 10% (from eval harness), p95 latency > 5s, LLM API error rate > 1%

---

## Cost at 10k Queries/Day (summary)

| Component | At demo scale | At 10k q/day |
|-----------|--------------|--------------|
| LLM (Haiku, no caching) | ~$0 | ~$375/month |
| LLM (Haiku, 60% cache hit) | ~$0 | ~$175/month |
| Vector DB (Chroma local) | $0 | $0–$200/month (managed) |
| BM25 (in-RAM) | $0 | $100–300/month (Elasticsearch) |
| Embedding (all-MiniLM local) | $0 | $0 (self-hosted) or $20/month |
| **Total** | **$0** | **~$300–700/month** |

At 100k queries/day, Haiku becomes expensive (~$3.7k/month uncached). At that scale, consider fine-tuned smaller models or self-hosted open models (Llama 3.x via Ollama/vLLM).
