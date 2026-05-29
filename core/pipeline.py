"""
Orchestration: retrieve → generate, with observability logging.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

# Bumped in config when corpus is re-ingested; surfaced in every response so callers
# can correlate result quality shifts with corpus updates without parsing logs.
CORPUS_VERSION = os.getenv("CORPUS_VERSION", "v1")

from .generate import stream_answer
from .retrieve import HybridRetriever, RetrievedChunk


@dataclass
class QueryResult:
    question: str
    chunks: list[RetrievedChunk]
    answer_tokens: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    query_id: int = -1

    @property
    def answer(self) -> str:
        return "".join(self.answer_tokens)

    @property
    def sources(self) -> list[dict]:
        seen: set[str] = set()
        out = []
        for c in self.chunks:
            key = c.source
            if key not in seen:
                seen.add(key)
                out.append({"source": c.source, "section": c.section, "score": round(c.combined_score, 3)})
        return out


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        db_path: str = "obs.db",
        k: int = 5,
        alpha: float = 0.7,
    ):
        self.retriever = retriever
        self.k = k
        self.alpha = alpha
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT DEFAULT (datetime('now')),
                    question    TEXT,
                    k           INTEGER,
                    alpha       REAL,
                    latency_ms  REAL,
                    retrieval_ms REAL,
                    generation_ms REAL,
                    num_chunks  INTEGER,
                    answer_len  INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS retrievals (
                    query_id    INTEGER,
                    rank        INTEGER,
                    doc_id      TEXT,
                    source      TEXT,
                    section     TEXT,
                    bm25_score  REAL,
                    vector_score REAL,
                    combined_score REAL
                )
            """)

    @contextmanager
    def _conn(self):
        # SQLite single-writer boundary: this is the only write site in the codebase.
        # One FastAPI instance is fine. Horizontal scaling (≥2 replicas) causes
        # "database is locked" errors at ~100+ concurrent writers.
        # Upgrade path: replace sqlite3.connect() here with asyncpg + SQLAlchemy async.
        # See docs/tradeoffs.md § "SQLite for Observability" for the full migration plan.
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _log_query(self, result: QueryResult) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO queries (question, k, alpha, latency_ms, retrieval_ms,
                   generation_ms, num_chunks, answer_len)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.question,
                    self.k,
                    self.alpha,
                    round(result.latency_ms, 1),
                    round(result.retrieval_ms, 1),
                    round(result.generation_ms, 1),
                    len(result.chunks),
                    len(result.answer),
                ),
            )
            query_id = cur.lastrowid
            conn.executemany(
                """INSERT INTO retrievals
                   (query_id, rank, doc_id, source, section, bm25_score, vector_score, combined_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        query_id,
                        i + 1,
                        c.doc_id,
                        c.source,
                        c.section,
                        round(c.bm25_score, 4),
                        round(c.vector_score, 4),
                        round(c.combined_score, 4),
                    )
                    for i, c in enumerate(result.chunks)
                ],
            )
        return query_id

    def query_stream(
        self,
        question: str,
        temperature: float = 0.2,
    ) -> Generator[str | dict, None, None]:
        """
        Yields:
          - str tokens as they stream from the LLM
          - a final dict {"__meta__": ..., "sources": ..., "chunks": ...}
        """
        t0 = time.perf_counter()

        t_ret = time.perf_counter()
        chunks = self.retriever.query(question, k=self.k, alpha=self.alpha)
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        result = QueryResult(question=question, chunks=chunks, retrieval_ms=retrieval_ms)

        t_gen = time.perf_counter()
        for token in stream_answer(question, chunks, temperature):
            result.answer_tokens.append(token)
            yield token
        result.generation_ms = (time.perf_counter() - t_gen) * 1000

        result.latency_ms = (time.perf_counter() - t0) * 1000
        query_id = self._log_query(result)
        result.query_id = query_id

        yield {
            "__meta__": {
                "query_id": query_id,
                "latency_ms": round(result.latency_ms, 1),
                "retrieval_ms": round(result.retrieval_ms, 1),
                "generation_ms": round(result.generation_ms, 1),
                "num_chunks": len(chunks),
                "retrieval_strategy": "hybrid",
                "corpus_version": CORPUS_VERSION,
            },
            "sources": result.sources,
            "chunks": [
                {
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "section": c.section,
                    "text": c.text[:200],
                    "combined_score": round(c.combined_score, 3),
                    "bm25_score": round(c.bm25_score, 3),
                    "vector_score": round(c.vector_score, 3),
                }
                for c in chunks
            ],
        }

    def recent_queries(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, ts, question, latency_ms, num_chunks, answer_len "
                "FROM queries ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "ts": r[1],
                "question": r[2],
                "latency_ms": r[3],
                "num_chunks": r[4],
                "answer_len": r[5],
            }
            for r in rows
        ]
