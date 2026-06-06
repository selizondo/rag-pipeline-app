# Product and Cost

This document frames the project for a technical business reviewer: what organizational risk it addresses, how the system earns trust, what it costs to operate at scale, and when a team should build a custom RAG system versus using a managed solution.

---

## The Business Problem

Most AI knowledge assistants fail in production because the team validated them in demo conditions: clean conceptual queries, a small corpus, and a developer looking at the output. Production exposes three failure modes that demos hide:

1. **Exact-keyword queries fail silently.** Acronyms, product names, and method signatures embed poorly in vector space. The model retrieves semantically related but irrelevant chunks and generates a plausible-sounding wrong answer. The user sees an answer; no alert fires.
2. **Wrong answers have no root cause.** Without query-level observability (what was retrieved, what scored highest), debugging a bad answer requires re-running the query and guessing. Was it retrieval? Was it the LLM? Different causes, different fixes.
3. **Cost compounds invisibly.** At 10k queries/day with Claude Haiku, uncached costs reach $375/month. Teams that ship without measuring per-query cost discover this on the cloud bill, not in the architecture review.

This system addresses all three with hybrid retrieval measured against a baseline, observability wired in from day one, and a documented cost model at scale.

---

## Trust Surface

**What can go wrong:**

- Vector-only retrieval misses exact-keyword queries (~15% of realistic ML Q&A queries in eval). The model generates confident answers from loosely related chunks.
- A corpus update (new documents ingested) changes chunk distribution. The BM25 index is rebuilt from Chroma on ingest. If rebuild fails silently, BM25 scores the old corpus.
- Template injection in the query string. The FastAPI query endpoint accepts arbitrary text and injects it directly into the LLM prompt.
- SQLite `obs.db` grows unbounded. At 10k queries/day, 1M rows accumulates in weeks. No eviction policy.

**How this system addresses each:**

- Hybrid retrieval (alpha=0.7) raises Accuracy@4 from 79% to 83% on the eval case set. The 4pp gain is concentrated on exact-keyword queries. The alpha slider allows live comparison between retrieval strategies.
- BM25 index rebuild is triggered automatically after every `/ingest` call. If the rebuild fails, the API returns an error. The health endpoint (`GET /api/v1/health`) reports index state.
- The LLM prompt does not expose system instructions or prior conversation content to user query text. Prompt injection is a known gap: no sanitization layer is implemented. This is documented in docs/failures.md.
- SQLite cleanup is documented but not implemented. For production, partition `obs.db` by time and archive to cold storage.

**What is not addressed here:** Multi-tenant data isolation, per-user access control, authentication, rate limiting. This is a single-user demo architecture. For shared deployment, the upgrade path is documented in docs/engineering.md.

---

## Cost Model

### At demo scale

| Component | Cost |
|-----------|------|
| LLM (Ollama local) | $0 |
| Vector DB (Chroma local) | $0 |
| BM25 (in-RAM) | $0 |
| Embedding (all-MiniLM local) | $0 |
| SQLite observability | $0 |
| **Total** | **$0** |

### At 10k queries/day (Claude Haiku)

| Component | Monthly cost |
|-----------|-------------|
| LLM: Haiku, no caching (~500 input + 150 output tokens/query) | ~$375 |
| LLM: Haiku, 60% prompt cache hit rate | ~$175 |
| Vector DB: Chroma managed or Qdrant Cloud | $0 to $200 |
| BM25: Elasticsearch managed | $100 to $300 |
| Embedding: all-MiniLM self-hosted | $0 |
| Observability: Datadog or Honeycomb | $50 to $150 |
| **Total range** | **~$300 to $700/month** |

### At 100k queries/day

Haiku becomes ~$3.7k/month uncached. The inflection point for switching to self-hosted open models (Llama 3.x via vLLM): when API costs exceed infrastructure costs for a GPU server. At 100k queries/day and 150ms p50 generation, a single A10G GPU handles approximately 6 to 7 concurrent requests, enough for this volume at ~$500 to $800/month self-hosted.

**Cost levers:**
- Prompt caching: cache the static context prefix. Saves ~60% on cache hits.
- Answer caching: Redis with TTL=1 hour for high-frequency identical queries. Hit rate on typical knowledge bases: 15 to 30%.
- Tiered routing: Haiku for simple factual queries, Sonnet for complex multi-step reasoning.

---

## Market Context

The RAG tooling market (LlamaIndex, LangChain, Vertex AI Search, Azure AI Search) has standardized on hybrid retrieval as the production default. Most teams implement it via a managed platform without understanding what alpha weighting does or when it helps.

The gap: teams that adopt hybrid retrieval from a platform get a default alpha value but not the methodology for tuning it. They cannot answer "should we use more BM25 for our corpus?" without running an experiment. This project provides that experiment, with the alpha sweep and eval harness wired together.

The observability gap is larger. Most RAG implementations log queries but not chunk scores. Without per-chunk scoring, the question "did retrieval fail or did the LLM fail?" cannot be answered from logs alone. This system logs both.

---

## Deployment Constraints

**Startup time:** The FastAPI backend loads the embedding model (~400MB) and builds the BM25 index from Chroma on startup. Cold start: 15 to 30 seconds. For production, use a warm instance or a readiness probe before routing traffic.

**Single-process bottleneck:** One FastAPI process, synchronous query handling. At 10k queries/day spread over 8 hours, average load is ~0.35 queries/second: a single process handles this easily. At peak (bursts of 5 to 10 concurrent queries), add `uvicorn --workers 4` or run behind a load balancer.

**Synchronous ingest:** `/ingest` blocks until all chunks are embedded and indexed. At 1000 documents (256 words each, ~40ms per embedding): ~40 seconds. For larger corpora, return a `job_id` immediately and process asynchronously.

**Docker:** The `docker-compose.yml` deploys API and UI as separate containers with a shared volume for Chroma and SQLite. For team deployment, migrate Chroma to its HTTP server mode so the volume is not a single point of failure.

---

## Build vs Buy

**Build (custom RAG) when:**

- The team needs to understand the alpha tuning methodology and own the eval loop, not just accept platform defaults
- Corpus is small enough that Chroma local is sufficient and managed vector DB costs are not justified
- The observability story needs to be custom: per-query chunk scores, retrieval vs generation latency split, integration with an existing observability stack
- The team wants to extend the pipeline: custom reranking, metadata filters, multi-index retrieval

**Use a managed RAG platform (Vertex AI Search, Azure AI Search, Elastic App Search) when:**

- The team needs enterprise access control, audit logs, and compliance features built in
- Corpus is large (1M+ documents) and the team does not want to operate Elasticsearch or Qdrant
- The team's queries are general-purpose and platform default tuning is acceptable

**The judgment call:** Managed platforms abstract away alpha tuning and chunk size decisions. Teams that adopt them without understanding the tradeoffs get a black box. The first time a managed platform performs poorly on a specific query type, the team has no lever to pull. Building the custom pipeline first builds the understanding needed to evaluate managed platforms critically, not just adopt them by default.
