# Failure Scenarios

Failure modes for the RAG pipeline app. "Handled" means a non-fatal path exists. "Documented gap" means the failure is understood but detection is not yet implemented.

---

## Failure 1: Ollama Unavailable (Handled)

**What breaks:** `generate.py` streams tokens from the Ollama HTTP endpoint. If Ollama isn't running, the connection fails.

**Status:** Handled — `_stream_ollama` wraps the HTTP call in typed try/except blocks. Yields error tokens to the SSE stream rather than raising. The SSE connection stays open; the UI displays the error inline.

**Error types caught:**
- `ConnectionError` → `"[ERROR: Ollama not reachable at {url}. Is it running?]"`
- `Timeout` → `"[ERROR: Ollama request timed out after 120s...]"`
- `HTTPError` → `"[ERROR: Ollama returned HTTP {code}: {text}]"`
- Generic `Exception` → `"[ERROR: Ollama generation failed: {e}]"`

**Why error tokens instead of exceptions:** A generator that raises mid-stream has already sent HTTP headers — the SSE connection is open. An exception at that point produces a broken stream. Yielding `[ERROR: ...]` lets the client display it.

---

## Failure 2: Anthropic API Errors (Handled)

**What breaks:** `generate.py` `_stream_anthropic` uses `client.messages.stream()`. Auth failures, rate limits, and network issues can occur.

**Status:** Handled — typed exception handlers for `AuthenticationError`, `RateLimitError`, `APIConnectionError`, and generic `Exception`. All yield informative error tokens.

---

## Failure 3: Chroma Collection Empty at Startup (Partially Handled)

**What breaks:** `HybridRetriever.__init__` calls `_build_bm25_index()`, which fetches all docs from Chroma. If ingest hasn't been run, the collection is empty — BM25 index is not built (`self._bm25 = None`). A subsequent query call to `retrieve()` hits `self._bm25.get_scores(...)` on `None`.

**Status:** Partially handled — `_build_bm25_index` checks `if not docs` and returns early, leaving `self._bm25 = None`. The `retrieve()` call will hit an `AttributeError`. No ingest-not-run check at the API layer.

**Detection (planned):** In `_build_bm25_index`, if docs is empty, either raise `RuntimeError("Collection is empty — run ingest first")` or return a degraded retriever that falls back to vector-only search.

---

## Failure 4: Embedding Model Not Downloaded (Documented Gap)

**What breaks:** `HybridRetriever.__init__` calls `SentenceTransformer(embed_model)`. On first run, the model is downloaded from HuggingFace. If the download fails (no network, HuggingFace unavailable), the retriever fails to initialize.

**Status:** Documented gap — no retry or clear error message.

**Detection (planned):** Wrap model load in try/except; provide a clear error message with cache path and environment variable hint (`SENTENCE_TRANSFORMERS_HOME`).

---

## Failure 5: Concurrent Ingest + Query (Documented Gap)

**What breaks:** If ingest is running while a query arrives, the Chroma collection is in a partially-written state. The BM25 index (built at `HybridRetriever` startup) is also stale — it doesn't reflect in-flight ingest operations.

**Status:** Documented gap. Acceptable for the demo (single-user, sequential ingest-then-query). Production path: document ingest as a separate job that requires a retriever restart, or use Chroma's thread-safe client.

---

## Failure 6: SSE Connection Drop Mid-Stream (Documented Gap)

**What breaks:** If the client disconnects while the generator is still producing tokens (user closes tab, network drop), FastAPI continues generating tokens until the `stream_answer` generator is exhausted. This wastes API calls for Anthropic backend.

**Status:** Documented gap — no disconnection detection.

**Detection (planned):** Check `await request.is_disconnected()` in the FastAPI route between token yields. If disconnected, break the generator loop and abort the LLM call.
