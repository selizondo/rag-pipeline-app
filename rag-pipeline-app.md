## Staff-Level Review: rag-pipeline-app

### Executive Summary

This is a **strong portfolio project** that demonstrates solid ML systems design principles and production-thinking architecture. The hybrid BM25+vector retrieval with measured evaluation results, clear API contracts, and explicit scale documentation are all Staff-level signals. However, it lacks unit tests, has some unhandled failure modes in the LLM layer, and misses opportunities for advanced observability patterns seen in other portfolio projects.

**Overall:** Production-ready architecture, development-stage robustness. Suitable as a portfolio narrative for "how to structure a RAG system for production," but needs test coverage and failure-scenario documentation for a production deployment claim.

---

### Architecture & Design

#### Core Strengths

**1. Contract-First API Design** ✅
The system front-loads HTTP boundaries and data models. [api/models.py](api/models.py) defines request/response schemas with Pydantic validation before any logic:
- `QueryRequest` enforces k ∈ [1, 20], alpha ∈ [0, 1], temperature ∈ [0, 1], question length ∈ [3, 1000]
- `QueryResponse` mandates sources, chunks with scoring details, and QueryMeta (latency breakdown)
- Ingest contract is separate: `IngestRequest` with path validation, `IngestResponse` with file count

This forces clients to think about the contract early. Good signal.

**2. HTTP Boundary Between Retrieval & Presentation** ✅
[api/main.py](api/main.py) creates a clean separation:
- `core/` (retrieval, generation, pipeline orchestration) is testable without a browser
- `ui/` (Streamlit frontend) is pure presentation layer; calls `/api/v1/query/stream` via SSE
- Eval harness plugs into the same HTTP endpoint — no special instrumentation needed

This design choice is justified in [docs/adr-02-api-ui-split.md](docs/adr-02-api-ui-split.md). The ~5ms SSE overhead is minimal and the benefit (API reuse) is substantial.

**3. Hybrid Retrieval Evaluated Before Shipping** ✅
[docs/adr-01-hybrid-search.md](docs/adr-01-hybrid-search.md) is excellent:
- **Problem identified:** Pure vector search fails on acronyms and exact keywords (~15% of realistic queries). Concrete case: "What is LoRA?" misses the definition.
- **Baseline measured:** α=1.0 (vector-only) → 79% Accuracy@4, 4% hallucination rate
- **Improvement measured:** α=0.7 (hybrid default) → 83% Accuracy@4, 3% hallucination
- **Default justified:** 0.7 balances semantic queries (higher α) and exact lookup (lower α)

The ADR **doesn't claim α=0.7 is optimal globally** — it explains the tradeoff and makes the slider tunable in the UI. Staff-level thinking.

#### Architecture Diagram (from README)
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
   [HybridRetriever]      [stream_answer]
   BM25 + Chroma          Ollama/Claude
       │
       ▼
   [SQLite obs.db]
```

---

### Failure Mode Analysis

#### Documented Failure Modes ⚠️

**Good:** ADR-01 identifies and measures the vector-only failure (acronyms). The hybrid approach with α=0.7 is a direct mitigation.

**Missing:**
1. **No test scenarios for edge cases:**
   - What if corpus is empty? (Code: [retrieve.py](core/retrieve.py#L56) returns [], [routes.py](api/routes.py#L67) accepts it silently)
   - What if `k > len(corpus)`? (Chroma.query handles this, but no test)
   - What if BM25 index fails to rebuild after ingest? (No signal to caller; health endpoint is best-effort)

2. **No LLM error handling:**
   - [generate.py#_stream_ollama](core/generate.py#L60): `resp.raise_for_status()` crashes the request if Ollama is down
   - [generate.py#_stream_anthropic](core/generate.py#L77): No try/except for API errors (rate limits, auth failure)
   - No retry logic or exponential backoff

3. **No degradation signals in response:**
   - Query returns full chunks even if BM25 failed to rebuild
   - No field like `"warnings": ["bm25_index_not_rebuilt", "ollama_unreachable"]`
   - Contrast: [rag-eval-pipeline](../rag-eval-pipeline) includes `"retrieval_strategy"` in every response so observers know what was actually used

4. **Silent failures on malformed context:**
   - If a chunk's source or section metadata is malformed (e.g., section field is `None`), [generate.py#_build_prompt](core/generate.py#L35) builds a broken header
   - No sanitization or fallback

#### Scale Boundaries (Excellent Documentation) ✅

[docs/scale-design.md](docs/scale-design.md) is thorough:
- **BM25 in RAM:** Current ~400 MB at 50k chunks; breaks at ~1M chunks without a service like Elasticsearch
- **Chroma single-process:** Scales to 100M vectors via managed Chroma or Qdrant; local mode caps at ~50k for demo
- **Multi-tenant:** Recommends `where` clause filtering on `tenant_id` at retrieval time
- **Async ingest:** Current sync endpoint would timeout on 50k+ chunks; suggests Celery + Redis
- **LLM cost:** 10k q/day with Haiku = ~$375/month uncached, ~$175 with 60% cache-hit
- **Observability:** Current SQLite is single-writer; suggests Datadog/Honeycomb at scale

This is excellent. Boundaries are explicit; tradeoffs are clear.

---

### Observability

#### Response Fields ✅

Every `/api/v1/query` response includes:
```json
{
  "answer": "...",
  "sources": [
    {"source": "ml_fundamentals.md", "section": "Gradient Descent", "score": 0.82}
  ],
  "chunks": [
    {
      "doc_id": "ml_fundamentals.md::3",
      "source": "ml_fundamentals.md",
      "section": "Gradient Descent",
      "text": "...",
      "combined_score": 0.82,
      "bm25_score": 0.75,
      "vector_score": 0.88
    }
  ],
  "meta": {
    "query_id": 42,
    "latency_ms": 1234.5,
    "retrieval_ms": 45.2,
    "generation_ms": 1189.3,
    "num_chunks": 5
  }
}
```

**Good:** Latency breakdown (retrieve vs generation), per-chunk scoring, query ID for log correlation.

#### Observability Gaps ❌

1. **No degradation flags:**
   - No `"fallback_used": "bm25"` (would signal hybrid degradation)
   - No `"warnings": ["bm25_index_stale"]` (BM25 rebuild failed silently)
   - No `"retrieval_source": "vector|bm25|fallback"` (unlike rag-eval-pipeline)

2. **No LLM token usage:**
   - Claude returns `usage_metadata` in stream output; not captured or returned to caller
   - Could track input/output tokens for cost/latency analysis

3. **SQLite query log has no error tracking:**
   - Schema: `queries(id, ts, question, k, alpha, latency_ms, retrieval_ms, generation_ms, num_chunks, answer_len)`
   - Missing: `failed` (bool), `error_reason` (str), `fallback_used` (str)
   - Means you can't query "how often does generation fail?" without parsing logs

4. **Health endpoint is best-effort:**
   ```python
   try:
       count = pipeline.retriever._collection.count()
       bm25_indexed = pipeline.retriever._bm25 is not None
   except Exception:
       count = 0
       bm25_indexed = False
   ```
   Exception is swallowed; caller doesn't know *why* collection is empty. Could be:
   - Chroma client crashed
   - BM25 rebuild failed
   - Collection never ingested
   All map to `bm25_indexed=False`

---

### ML Data Quality Lens

#### Dataset & Evaluation ❌

1. **Ground truth construction not documented:**
   - Demo corpus ([demo/corpus/ml_fundamentals.md](demo/corpus/ml_fundamentals.md)) is hand-written reference material, not evaluation labels
   - No contrast with real eval datasets (e.g., SQuAD, ASQA, or domain-specific QA pairs)

2. **No baseline metric reported in repo:**
   - ADR-01 reports α=0.7 achieves "83% Accuracy@4" but **doesn't commit the eval run to the repo**
   - No way to reproduce the numbers without running the llm-eval-harness against the sibling project
   - Contrast: [rag-eval-pipeline](../rag-eval-pipeline) commits `artifacts/eval/` with RAGAS scores

3. **Eval harness wired but no results logged:**
   - [evals/run_evals.py](evals/run_evals.py) has `--cases ../llm-eval-harness/evals/cases/rag_qa.jsonl` but:
     - No eval cases committed to this repo
     - Depends on sibling project's llm-eval-harness
     - Would need `make eval` to run; no CI/CD result logged

4. **No metric selection rationale:**
   - Why Accuracy@4? Why not NDCG@5 or MRR?
   - No discussion of cold-start bias (if eval includes new-user queries, BM25 may unfairly help)

#### RAG-Specific Data Quality Issues ❌

1. **No faithfulness vs relevancy split:**
   - ADR-01 reports Accuracy (mix of both); doesn't measure RAGAS faithfulness separately
   - Contrast: [rag-eval-pipeline/docs/eval_results.md](../rag-eval-pipeline/docs/eval_results.md) reports dense faithfulness=0.358, BM25=0.625 (BM25 wins on grounding!)
   - Missing insight: BM25 may trade relevancy for faithfulness

2. **No adaptive fallback threshold tuning:**
   - Chunking strategy is fixed (word-based, 256 words, 32-word overlap)
   - No evaluation of: sentence-boundary chunking, fixed-size tokens, or resolution-first ranking (mentioned in rag-eval-pipeline)
   - No measurement of impact

3. **Version-scoped retrieval not implemented:**
   - Chunks don't carry a `version` field (e.g., `"version": "v1"` for A/B testing)
   - [rag-eval-pipeline](../rag-eval-pipeline) shows 35% improvement from version-scoped filtering (0.358 → 0.485 faithfulness)
   - Not tested here

#### Chunking Strategy Rationale ⚠️

[core/ingest.py](core/ingest.py) uses word-based chunking:
```python
def _word_chunks(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int]]:
    """Split text into overlapping chunks by word count."""
    words = text.split()  # naive split on whitespace
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
```

**Issues:**
- Naive tokenization (splits on whitespace only); doesn't handle punctuation well
- Fixed-size chunks split sentences; can bury key content mid-chunk
- No evaluation against sentence-boundary chunking (which [rag-eval-pipeline](../rag-eval-pipeline) uses via `sentence_chunker.py`)

**No measurement:** No commit message or ADR explaining why word-based was chosen over alternatives.

---

### Test Coverage & Testability

#### No Unit Tests ❌

```bash
$ find rag-pipeline-app -name "test_*.py" -o -name "*_test.py"
# (empty)
```

**Critical gaps:**

1. **Retrieval logic untested:**
   ```python
   # [retrieve.py#query] — no tests for:
   - empty corpus (should return [])
   - k > corpus size (should return all)
   - alpha normalization edge cases (divide-by-zero if all scores are 0)
   - query with special characters or very long text
   ```

2. **API contract untested:**
   - Malformed requests (alpha=-0.1, k=0, question="x", temperature=2.0)
   - Empty corpus + query (should return 200 with empty chunks, not 500)
   - Concurrent ingest while querying (race condition in BM25 rebuild?)

3. **LLM backend untested:**
   - Ollama timeout (no mock, can't test retry logic that doesn't exist)
   - Anthropic API error (missing error handler)
   - Streaming token loss (what if SSE connection drops mid-generation?)

4. **Failure scenarios untested:**
   - Chroma collection missing (BM25 rebuilt but Chroma broke)
   - BM25 index corrupted (tokenization error on a chunk)
   - SQLite obs.db locked by another process

**Consequence:** These failure modes will only be discovered in production or during manual testing.

---

### Production Readiness Checklist

| Signal | Present? | Severity | Notes |
|--------|----------|----------|-------|
| **Contract-first API** | ✅ Yes | — | Excellent models.py |
| **HTTP boundary** | ✅ Yes | — | Separates logic from presentation |
| **Failure modes documented** | ⚠️ Partial | Medium | ADR-01 covers hybrid search; missing LLM error docs |
| **Non-fatal degradation** | ⚠️ Partial | Medium | BM25 handles empty corpus; LLM errors crash |
| **Observability in response** | ✅ Yes | — | Good latency breakdown, per-chunk scores |
| **Degradation signals** | ❌ No | Medium | No "fallback_used", "warnings" fields |
| **Scale boundaries documented** | ✅ Yes | — | Excellent docs/scale-design.md |
| **Baselines measured** | ✅ Yes | — | ADR-01 reports α tuning results |
| **Eval reproducible** | ⚠️ Partial | Medium | Depends on sibling llm-eval-harness; no committed results |
| **Unit tests** | ❌ No | High | Zero tests; edge cases untested |
| **Multi-instance safe** | ❌ No | High | SQLite obs.db is single-writer |
| **LLM error handling** | ❌ No | High | Ollama/Claude errors crash the request |

---

### Design Tradeoffs (Well-Articulated)

1. **BM25 in RAM vs Elasticsearch** ✅
   - Trade: 400 MB RAM @ 50k chunks vs managed service cost
   - Decision: RAM is OK for demo; Elasticsearch deferred to scale-design.md
   - Staff signal: Boundary is explicit

2. **FastAPI + Streamlit split vs monolith** ✅
   - Trade: HTTP overhead (~5ms SSE per request) vs API reuse + testability
   - Decision: HTTP overhead is negligible; API reuse wins
   - Staff signal: Tradeoff is quantified in ADR-02

3. **Token streaming via SSE vs blocking** ✅
   - Trade: Complexity of event-stream format vs UX (users don't wait)
   - Decision: UX wins; users stop waiting after 3–5 sec per README
   - Staff signal: Problem statement is concrete

4. **Word-based chunking vs sentence-boundary** ⚠️
   - Trade: Simple implementation vs better semantic boundaries
   - Decision: No explicit decision documented; word-based assumed
   - Missing: Eval result comparing both approaches

---

### Recommendations

#### High Priority (Production Blocking) 🔴

1. **Add unit tests for failure scenarios** (2–3 days)
   - Empty corpus: `test_empty_retriever_returns_empty()`
   - Malformed API request: `test_query_invalid_alpha()`, `test_query_empty_question()`
   - LLM backend error: `test_stream_ollama_timeout()`, `test_stream_anthropic_auth_error()`
   - Target: 50+ tests covering core + api modules

2. **Handle LLM backend errors gracefully** (1 day)
   - Wrap `_stream_ollama()` and `_stream_anthropic()` in try/except
   - Yield error metadata: `{"type": "error", "reason": "ollama_timeout"}`
   - Add exponential backoff retry on transient errors (429, 503)
   - Document retry policy in API contract

3. **Add degradation signals to response** (1 day)
   - Add `"warnings": [...]` field to QueryResponse
   - Include: `"bm25_index_missing"`, `"fallback_to_pure_vector"`, `"no_retrieval_results"`
   - Callers can alert on `len(warnings) > 0` without parsing logs

#### Medium Priority (Production Recommended) 🟡

4. **Document failure scenarios in committed tests** (2 days)
   - Create `tests/test_failure_scenarios.py` similar to [rag-eval-pipeline/tests/test_failure_scenarios.py](../rag-eval-pipeline/tests/test_failure_scenarios.py)
   - Run demo with corpus missing, Ollama down, BM25 corruption
   - Should succeed gracefully or fail with clear error messages

5. **Commit eval results to repo** (1 day)
   - Run `make eval` against llm-eval-harness
   - Commit results to `artifacts/eval/latest_run.json`
   - Update README with Accuracy@4, Hallucination rate, α tuning baseline
   - Enable CI to alert if accuracy regresses

6. **Add observability field to QueryMeta** (4 hours)
   ```python
   class QueryMeta(BaseModel):
       # existing fields...
       retrieval_strategy: str = "hybrid"  # "hybrid", "bm25_only", "vector_only"
       fallback_reason: str | None = None  # "empty_corpus", "vector_low_score", etc.
   ```

7. **Measure chunking strategy impact** (3 days)
   - Compare word-based (current) vs sentence-boundary chunking
   - Run eval on both; report NDCG@5, retrieval latency
   - Document tradeoff in ADR-03 or update README

#### Low Priority (Nice-to-Have) 🟢

8. **Support multi-instance deployments**
   - Move SQLite obs.db to PostgreSQL with async writes
   - Allow horizontal scaling of FastAPI (load-balanced behind Nginx)
   - Document in scale-design.md

9. **Add adaptive threshold for BM25 fallback**
   - If max(vector_scores) < 0.3, use pure BM25 instead of hybrid
   - Empirically tune threshold from eval data
   - Reference: [rag-eval-pipeline/retrieval/strategies.py#AdaptiveRetriever](../rag-eval-pipeline/retrieval/strategies.py)

10. **Support version-scoped filtering**
    - Add `version` field to Chunk metadata
    - Allow queries with `?version=v1` to filter chunks pre-retrieval
    - Enables A/B testing of document versions

---

### Strengths Summary (What to Highlight in Interview)

1. **Measured decision-making:** ADR-01 reports concrete eval results (79% vs 83% accuracy) justifying α=0.7. Most engineers don't measure before committing to a design choice.

2. **Clear scale boundaries:** docs/scale-design.md explicitly addresses "how would this break at 1M docs?" Shows production-system thinking.

3. **API-first architecture:** HTTP boundary between retrieval and UI enables testing, reuse, and future client diversity (CLI, mobile, agents). Not a monolith.

4. **Streaming UI without blocking:** Handles the user-experience threshold (users stop waiting after ~3s). Shows ops thinking.

5. **Structured observability:** Every response includes query_id, latency breakdown, per-chunk scores. Enables debugging without log digging.

---

### Narrative for Portfolio

**30-second pitch:**
> "I built a RAG system that separates retrieval logic from presentation via a FastAPI backend, enabling independent testing and future client diversity. I implemented hybrid BM25+vector search after measuring that pure vector failed on 15% of queries (acronyms like 'LoRA'). The system logs latency per stage, chunk scores, and query artifacts to SQLite for observability. It streams tokens to the UI via SSE so users don't wait. I documented explicit scale boundaries for when to migrate from in-memory BM25 to Elasticsearch and from local Chroma to managed vector DBs."

**When asked about handling failure:**
> "The hybrid retrieval provides non-fatal degradation — if BM25 can't index for any reason, the system falls back to vector-only. But I know I'm missing better error handling for LLM backends (Ollama timeout, Claude auth failure) and I'm not yet tracking those signals in the response metadata. That's on my roadmap."

**When asked about eval rigor:**
> "I wired the system to an eval harness and measured α tuning: pure vector achieved 79% accuracy, hybrid at 0.7 achieved 83%, pure BM25 achieved 61%. But I haven't committed those results to the repo yet, so they're not reproducible without running against the harness. The system is ready for ops work, but needs test coverage for edge cases."

---

### Summary Table

| Category | Rating | Notes |
|----------|--------|-------|
| **Architecture** | ⭐⭐⭐⭐⭐ | Contract-first, HTTP boundary, scale-aware |
| **Failure handling** | ⭐⭐⭐ | Good docs, missing LLM error handlers, no tests |
| **Observability** | ⭐⭐⭐⭐ | Latency breakdown, per-chunk scores; missing degradation flags |
| **Production readiness** | ⭐⭐⭐ | Good design, missing unit tests and multi-instance support |
| **Data quality rigor** | ⭐⭐⭐ | Measured baselines, missing faithfulness split, no version-scoped eval |
| **Code quality** | ⭐⭐⭐⭐ | Clean separation, type hints; magic numbers, silent failures |
| **Scale design** | ⭐⭐⭐⭐⭐ | Excellent documentation of boundaries and tradeoffs |
| **Overall (Portfolio)** | ⭐⭐⭐⭐ | Staff-level architecture + mid-level execution; ready for "structuring RAG systems" discussion