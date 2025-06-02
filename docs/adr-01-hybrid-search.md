# ADR-01: Hybrid BM25 + Vector Search

**Status:** Accepted  
**Date:** 2025-05-08

---

## Context

The baseline RAG pipeline ([rag-pipeline-from-scratch](../../rag-pipeline-from-scratch)) uses pure vector search (cosine similarity over all-MiniLM-L6-v2 embeddings). This works well for semantic queries but has a known failure mode: exact keyword lookups — acronyms, model names, API names, method signatures, specific error codes — score poorly because they depend on token frequency, not semantic closeness.

Two concrete failure cases we observed:

1. Query: "What is LoRA?" — pure vector search retrieved chunks about "low-rank matrix decomposition" but missed the chunk that literally defined LoRA (Low-Rank Adaptation), because the acronym didn't embed closely to its expansion.
2. Query: "BM25 vs TF-IDF" — vector search returned semantically similar IR content but ranked below the exact-match chunk where BM25 was defined.

## Decision

Use **hybrid retrieval**: combine BM25 (lexical) and vector (semantic) scores with a configurable linear interpolation:

```
score(d, q) = α · vector_score(d, q) + (1 − α) · bm25_score(d, q)
```

Default `α = 0.7` (70% vector, 30% BM25), tunable per-request via the API.

BM25 index is built once at startup from all chunks stored in Chroma. Rebuilt automatically after `/ingest`.

## Alternatives Considered

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Pure vector (status quo) | Simple, no extra dependency | Poor on exact keyword queries | Rejected |
| Sparse-dense fusion (SPLADE/ColBERT) | State-of-the-art recall | Requires separate model inference, 5–10× slower | Too heavy for this stage |
| Cross-encoder reranking | High precision at top-k | Adds 200–500ms per query | Reserved for future optimization |
| **BM25 + vector (chosen)** | Strong recall improvement, no extra model | BM25 needs full corpus in RAM | Accepted |

## Consequences

**Positive:**
- Handles exact keyword queries that pure vector misses (~15% of the eval cases improved by ≥1 correctness point in internal testing).
- `α` slider in the UI lets interviewers see the tradeoff live — a strong demo artifact.
- BM25 implementation is pure Python (`rank_bm25`); no GPU or separate service required.

**Negative:**
- BM25 index holds all chunk texts in RAM. At 50k chunks × 256 words, estimated ~400 MB. Acceptable for demo scale; would need an inverted index service (Elasticsearch, OpenSearch) at production scale (1M+ docs).
- `O(n)` BM25 scoring across the full corpus on every query. At 10k chunks: ~2ms, negligible. At 1M chunks: ~200ms, need an approximation layer.

## Tuning Notes

The default `α = 0.7` was selected based on the eval harness ([llm-eval-harness](../../llm-eval-harness)):
- `α = 1.0` (pure vector): Accuracy@4 = 79%, Hallucination rate = 4%
- `α = 0.7` (hybrid default): Accuracy@4 = 83%, Hallucination rate = 3%
- `α = 0.0` (pure BM25): Accuracy@4 = 61%, Hallucination rate = 8%

Semantic queries (most ML concept questions) respond better to higher `α`. Exact lookup queries (API names, paper titles) respond better to lower `α`. The 0.7 default balances both.
