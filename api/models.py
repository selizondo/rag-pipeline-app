from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Vector weight (0=BM25 only, 1=vector only)")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class SourceRef(BaseModel):
    source: str
    section: str
    score: float


class ChunkRef(BaseModel):
    doc_id: str
    source: str
    section: str
    text: str
    combined_score: float
    bm25_score: float
    vector_score: float


class QueryMeta(BaseModel):
    query_id: int
    latency_ms: float
    retrieval_ms: float
    generation_ms: float
    num_chunks: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    chunks: list[ChunkRef]
    meta: QueryMeta


class IngestRequest(BaseModel):
    corpus_dir: str = Field(..., description="Absolute path to directory with .md/.txt files")
    chunk_size: int = Field(default=256, ge=64, le=1024)
    overlap: int = Field(default=32, ge=0, le=256)


class IngestResponse(BaseModel):
    num_chunks: int
    files_processed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    collection_count: int
    bm25_indexed: bool
