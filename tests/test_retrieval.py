"""
Unit tests for retrieval failure modes and edge cases.

These tests use mocks and in-memory state; they do NOT require
Ollama, Chroma, or sentence-transformers to be running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal stubs so we can import HybridRetriever without a real Chroma/ST
# ---------------------------------------------------------------------------

@dataclass
class _FakeChunk:
    text: str
    doc_id: str
    source: str = "test.md"
    section: str = ""
    chunk_index: int = 0
    char_start: int = 0
    metadata: dict = field(default_factory=dict)


def _make_retriever(docs: list[dict[str, Any]]):
    """Return a HybridRetriever wired to an in-memory fake collection."""
    with (
        patch("chromadb.PersistentClient") as mock_chroma,
        patch("sentence_transformers.SentenceTransformer") as mock_st,
    ):
        # Fake Chroma collection
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": [d["id"] for d in docs],
            "documents": [d["text"] for d in docs],
            "metadatas": [d.get("meta", {}) for d in docs],
            "embeddings": [[0.0] * 384 for _ in docs],
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Fake embed model: returns a zero vector per input
        import numpy as np
        mock_st.return_value.encode.return_value = np.zeros((1, 384))

        from core.retrieve import HybridRetriever

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._client = mock_chroma.return_value
        retriever._collection = mock_collection
        retriever._embed_model = mock_st.return_value
        retriever._bm25 = None
        retriever._all_docs = []
        retriever._build_bm25_index()

        return retriever, mock_collection


# ---------------------------------------------------------------------------
# Test: empty corpus
# ---------------------------------------------------------------------------

class TestEmptyCorpus:
    """Retrieval against an empty corpus must return [] without raising."""

    def test_query_empty_corpus_returns_empty_list(self):
        retriever, _ = _make_retriever([])
        results = retriever.query("What is LoRA?", k=5, alpha=0.7)
        assert results == [], f"Expected [], got {results}"

    def test_query_empty_corpus_no_exception(self):
        retriever, _ = _make_retriever([])
        try:
            retriever.query("test query", k=5, alpha=0.7)
        except Exception as exc:
            raise AssertionError(f"query() raised unexpectedly: {exc}") from exc

    def test_k_larger_than_corpus_returns_at_most_corpus_size(self):
        docs = [
            {"id": "doc::0", "text": "Neural networks are universal approximators.", "meta": {"source": "nn.md", "section": ""}},
            {"id": "doc::1", "text": "Gradient descent minimizes the loss function.", "meta": {"source": "nn.md", "section": ""}},
        ]
        retriever, mock_collection = _make_retriever(docs)

        # Mock the Chroma query response for vector search
        import numpy as np
        mock_collection.query.return_value = {
            "ids": [["doc::0", "doc::1"]],
            "distances": [[0.1, 0.2]],
            "documents": [["Neural networks are universal approximators.", "Gradient descent minimizes the loss function."]],
            "metadatas": [[{"source": "nn.md", "section": ""}, {"source": "nn.md", "section": ""}]],
        }
        retriever._embed_model.encode.return_value = np.zeros((1, 384))

        # k=10 but corpus only has 2 docs → must not crash, must return ≤ 2
        results = retriever.query("What is a neural network?", k=10, alpha=0.7)
        assert len(results) <= 2, f"Expected ≤2 results but got {len(results)}"


# ---------------------------------------------------------------------------
# Test: BM25 rebuild failure → fall back to dense retrieval
# ---------------------------------------------------------------------------

class TestBM25RebuildFailure:
    """If BM25 rebuild raises, the retriever should not crash.

    The system currently calls _build_bm25_index() inside add_chunks().
    This test verifies that a failure during rebuild leaves _bm25 = None
    (vector-only degraded mode) rather than propagating an exception.
    """

    def test_bm25_rebuild_exception_does_not_propagate(self):
        retriever, mock_collection = _make_retriever([])

        # Force the collection.get() call (used inside _build_bm25_index) to raise
        mock_collection.get.side_effect = RuntimeError("Chroma internal error")

        try:
            retriever._build_bm25_index()
        except RuntimeError:
            raise AssertionError(
                "_build_bm25_index() should not propagate RuntimeError; "
                "it should degrade to vector-only (bm25=None)"
            )

    def test_bm25_none_after_rebuild_failure(self):
        retriever, mock_collection = _make_retriever([])

        # _bm25 was None initially; after a failed rebuild it should still be None
        _initial_bm25 = retriever._bm25
        mock_collection.get.side_effect = RuntimeError("Chroma internal error")

        try:
            retriever._build_bm25_index()
        except Exception:
            pass  # swallowed or propagated; either way check state below

        # If the exception propagated, test_bm25_rebuild_exception_does_not_propagate
        # will catch it. Here we just confirm _bm25 was never set to a non-None value.
        assert retriever._bm25 is None, (
            "Expected _bm25 to remain None after failed rebuild, "
            f"but got {retriever._bm25}"
        )
