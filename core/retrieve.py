"""
Hybrid retrieval: BM25 + Chroma vector search, combined with configurable alpha.

score = alpha * vector_score + (1 - alpha) * bm25_score

BM25 index is built once on startup from all chunks stored in Chroma.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    text: str
    doc_id: str
    source: str
    section: str
    chunk_index: int
    bm25_score: float
    vector_score: float
    combined_score: float
    metadata: dict


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(
        self,
        chroma_path: str,
        collection_name: str = "rag_corpus",
        embed_model: str = "all-MiniLM-L6-v2",
    ):
        self._client = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed_model = SentenceTransformer(embed_model)
        self._bm25: BM25Okapi | None = None
        self._all_docs: list[dict[str, Any]] = []
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        result = self._collection.get(include=["documents", "metadatas", "embeddings"])
        ids = result["ids"]
        docs = result["documents"] or []
        metas = result["metadatas"] or []

        if not docs:
            self._bm25 = None
            self._all_docs = []
            return

        self._all_docs = [
            {
                "id": ids[i],
                "text": docs[i],
                "meta": metas[i] if metas else {},
            }
            for i in range(len(docs))
        ]
        tokenized = [_tokenize(d["text"]) for d in self._all_docs]
        self._bm25 = BM25Okapi(tokenized)

    def rebuild_index(self) -> None:
        self._build_bm25_index()

    def query(
        self,
        query_text: str,
        k: int = 5,
        alpha: float = 0.7,
    ) -> list[RetrievedChunk]:
        query_text = query_text.strip()
        if not self._all_docs or not query_text:
            return []

        # --- vector search ---
        q_emb = self._embed_model.encode([query_text]).tolist()
        vec_result = self._collection.query(
            query_embeddings=q_emb,
            n_results=min(k * 3, len(self._all_docs)),
            include=["documents", "metadatas", "distances"],
        )
        # Chroma cosine distances: 0 = identical, 2 = opposite → similarity = 1 - dist
        vec_ids = vec_result["ids"][0]
        vec_dists = vec_result["distances"][0]
        vec_docs = vec_result["documents"][0]
        vec_metas = vec_result["metadatas"][0]

        vec_scores: dict[str, float] = {}
        vec_texts: dict[str, str] = {}
        vec_metamap: dict[str, dict] = {}
        max_sim = 0.0
        for i, doc_id in enumerate(vec_ids):
            sim = max(0.0, 1.0 - vec_dists[i])
            vec_scores[doc_id] = sim
            vec_texts[doc_id] = vec_docs[i]
            vec_metamap[doc_id] = vec_metas[i] if vec_metas else {}
            if sim > max_sim:
                max_sim = sim
        if max_sim > 0:
            vec_scores = {k: v / max_sim for k, v in vec_scores.items()}

        # --- BM25 search ---
        tokens = _tokenize(query_text)
        bm25_raw = self._bm25.get_scores(tokens)
        max_bm25 = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
        bm25_norm = [s / max_bm25 for s in bm25_raw]

        bm25_map: dict[str, float] = {
            self._all_docs[i]["id"]: bm25_norm[i]
            for i in range(len(self._all_docs))
        }

        # --- combine ---
        candidate_ids = set(vec_ids) | {d["id"] for d in self._all_docs}
        scored: list[tuple[float, str]] = []
        for doc_id in candidate_ids:
            v = vec_scores.get(doc_id, 0.0)
            b = bm25_map.get(doc_id, 0.0)
            combined = alpha * v + (1 - alpha) * b
            scored.append((combined, doc_id))

        scored.sort(reverse=True)
        top_k = scored[:k]

        results: list[RetrievedChunk] = []
        doc_lookup = {d["id"]: d for d in self._all_docs}
        for combined_score, doc_id in top_k:
            if doc_id not in doc_lookup and doc_id not in vec_texts:
                continue
            text = vec_texts.get(doc_id) or doc_lookup.get(doc_id, {}).get("text", "")
            meta = vec_metamap.get(doc_id) or doc_lookup.get(doc_id, {}).get("meta", {})
            results.append(
                RetrievedChunk(
                    text=text,
                    doc_id=doc_id,
                    source=meta.get("source", ""),
                    section=meta.get("section", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    bm25_score=bm25_map.get(doc_id, 0.0),
                    vector_score=vec_scores.get(doc_id, 0.0),
                    combined_score=combined_score,
                    metadata=meta,
                )
            )
        return results

    def add_chunks(self, chunks: list) -> None:
        """Add new chunks (Chunk dataclasses or dicts) to Chroma and rebuild BM25 index."""
        def _get(c, key, default=""):
            return getattr(c, key, None) if hasattr(c, key) else c.get(key, default)

        texts = [_get(c, "text") for c in chunks]
        ids = [_get(c, "doc_id") for c in chunks]
        metas = [
            {
                "source": _get(c, "source"),
                "section": _get(c, "section"),
                "chunk_index": _get(c, "chunk_index", 0),
            }
            for c in chunks
        ]
        embeddings = self._embed_model.encode(texts).tolist()
        self._collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metas,
            embeddings=embeddings,
        )
        self._build_bm25_index()
