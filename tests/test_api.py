"""
Unit tests for API failure modes.

Covers:
  - Malformed requests → 422 (FastAPI validation), not 500
  - Ollama timeout → graceful error response, not 500
  - Empty question string → 422

These tests use FastAPI's TestClient with a mocked pipeline so no
Ollama, Chroma, or sentence-transformers are needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# App fixture: patch HybridRetriever and RAGPipeline before importing app
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """
    Return a TestClient with all external dependencies mocked out.

    WHY patch at import time: api.main creates the HybridRetriever inside
    the lifespan context manager. Patching chromadb and SentenceTransformer
    before the import prevents the real clients from being instantiated.
    """
    with (
        patch("chromadb.PersistentClient"),
        patch("sentence_transformers.SentenceTransformer"),
    ):
        # Provide a mock pipeline that the lifespan and routes will use
        mock_pipeline = MagicMock()
        mock_pipeline.retriever._collection.count.return_value = 0
        mock_pipeline.retriever._bm25 = None

        with patch("core.retrieve.HybridRetriever"), \
             patch("core.pipeline.RAGPipeline", return_value=mock_pipeline):
            import importlib

            import api.main as main_module
            importlib.reload(main_module)

            from fastapi.testclient import TestClient as _TC
            app = main_module.app

            with _TC(app) as tc:
                yield tc, mock_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_query_stream_response(tokens=("Hello", " world"), meta=None):
    """Return a list of items a real pipeline.query_stream would yield."""
    if meta is None:
        meta = {
            "__meta__": {
                "query_id": 1,
                "latency_ms": 100.0,
                "retrieval_ms": 10.0,
                "generation_ms": 90.0,
                "num_chunks": 0,
                "retrieval_strategy": "hybrid",
            },
            "sources": [],
            "chunks": [],
        }
    return list(tokens) + [meta]


# ---------------------------------------------------------------------------
# Malformed API requests → 422
# ---------------------------------------------------------------------------

class TestMalformedRequests:
    """FastAPI's Pydantic validation should return 422, never 500."""

    def test_missing_question_field_returns_422(self, client):
        tc, _ = client
        resp = tc.post("/api/v1/query", json={"k": 5, "alpha": 0.7})
        assert resp.status_code == 422, (
            f"Expected 422 for missing 'question', got {resp.status_code}"
        )

    def test_question_too_short_returns_422(self, client):
        """min_length=3 in QueryRequest — one-char question must fail validation."""
        tc, _ = client
        resp = tc.post("/api/v1/query", json={"question": "x"})
        assert resp.status_code == 422, (
            f"Expected 422 for question='x' (too short), got {resp.status_code}"
        )

    def test_alpha_out_of_range_returns_422(self, client):
        """alpha must be in [0, 1] — value 1.5 must be rejected."""
        tc, _ = client
        resp = tc.post("/api/v1/query", json={"question": "What is LoRA?", "alpha": 1.5})
        assert resp.status_code == 422, (
            f"Expected 422 for alpha=1.5, got {resp.status_code}"
        )

    def test_k_zero_returns_422(self, client):
        """k must be >= 1 — k=0 must be rejected."""
        tc, _ = client
        resp = tc.post("/api/v1/query", json={"question": "What is LoRA?", "k": 0})
        assert resp.status_code == 422, (
            f"Expected 422 for k=0, got {resp.status_code}"
        )

    def test_temperature_out_of_range_returns_422(self, client):
        """temperature must be in [0, 1] — value 2.0 must be rejected."""
        tc, _ = client
        resp = tc.post("/api/v1/query", json={"question": "What is LoRA?", "temperature": 2.0})
        assert resp.status_code == 422, (
            f"Expected 422 for temperature=2.0, got {resp.status_code}"
        )

    def test_empty_question_string_returns_422(self, client):
        """Empty string passes min_length=3 guard only if it's 3+ chars;
        the route also raises 422 for blank (whitespace-only) questions."""
        tc, _ = client
        resp = tc.post("/api/v1/query", json={"question": "   "})
        # Pydantic rejects len<3; whitespace-only "   " is len=3 but routes.py
        # raises HTTPException(422) for strip()==""
        assert resp.status_code in (422, 422), (
            f"Expected 422 for blank question, got {resp.status_code}"
        )


# ---------------------------------------------------------------------------
# Ollama timeout mock → graceful error, not 500
# ---------------------------------------------------------------------------

class TestOllamaTimeout:
    """
    When Ollama times out, stream_answer yields an [ERROR: ...] token.
    The API route should complete the response (200 or stream) with that
    error token rather than returning 500.

    WHY test the generator behaviour directly (not via HTTP):
        The SSE streaming route sends headers before yielding tokens.
        A mid-stream exception produces a broken response that TestClient
        cannot easily inspect. Instead, we test the generator layer directly
        to confirm the error token is yielded and no exception propagates.
    """

    def test_ollama_timeout_yields_error_token_not_exception(self):
        import requests as req_lib

        with patch("requests.post") as mock_post:
            mock_post.side_effect = req_lib.exceptions.Timeout("timed out")

            from core.generate import _stream_ollama

            # Provide a minimal fake chunk list (generate needs chunks for prompt)
            fake_chunk = MagicMock()
            fake_chunk.source = "test.md"
            fake_chunk.section = "Intro"
            fake_chunk.text = "Some context text."

            tokens = list(_stream_ollama("What is LoRA?", [fake_chunk], temperature=0.2))

        assert len(tokens) == 1, f"Expected 1 error token, got {tokens}"
        assert tokens[0].startswith("[ERROR:"), (
            f"Expected error token starting with '[ERROR:', got: {tokens[0]}"
        )
        assert "timed out" in tokens[0].lower() or "timeout" in tokens[0].lower(), (
            f"Expected timeout mention in error token, got: {tokens[0]}"
        )

    def test_ollama_connection_error_yields_error_token(self):
        import requests as req_lib

        with patch("requests.post") as mock_post:
            mock_post.side_effect = req_lib.exceptions.ConnectionError("refused")

            from core.generate import _stream_ollama

            fake_chunk = MagicMock()
            fake_chunk.source = "test.md"
            fake_chunk.section = ""
            fake_chunk.text = "Context."

            tokens = list(_stream_ollama("Test question?", [fake_chunk], temperature=0.2))

        assert len(tokens) == 1
        assert tokens[0].startswith("[ERROR:")

    def test_ollama_http_error_yields_error_token(self):
        import requests as req_lib

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_response.text = "Service unavailable"
            mock_post.side_effect = req_lib.exceptions.HTTPError(
                "503 Server Error", response=mock_response
            )

            from core.generate import _stream_ollama

            fake_chunk = MagicMock()
            fake_chunk.source = "test.md"
            fake_chunk.section = ""
            fake_chunk.text = "Context."

            tokens = list(_stream_ollama("Test?", [fake_chunk], temperature=0.2))

        assert len(tokens) == 1
        assert tokens[0].startswith("[ERROR:")


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        tc, mock_pipeline = client
        mock_pipeline.retriever._collection.count.return_value = 42
        mock_pipeline.retriever._bm25 = MagicMock()  # not None = indexed

        resp = tc.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_health_with_empty_corpus(self, client):
        """Health endpoint must return 200 even with an empty corpus."""
        tc, mock_pipeline = client
        mock_pipeline.retriever._collection.count.return_value = 0
        mock_pipeline.retriever._bm25 = None

        resp = tc.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["collection_count"] == 0
        assert body["bm25_indexed"] is False
