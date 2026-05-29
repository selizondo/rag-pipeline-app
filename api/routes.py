from __future__ import annotations

import json
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from core.ingest import ingest_directory
from core.pipeline import RAGPipeline

from .models import (
    ChunkRef,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryMeta,
    QueryRequest,
    QueryResponse,
    SourceRef,
)

router = APIRouter()


def get_pipeline() -> RAGPipeline:
    from api.main import pipeline
    return pipeline


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health(pipeline: RAGPipeline = Depends(get_pipeline)):
    try:
        count = pipeline.retriever._collection.count()
        bm25_indexed = pipeline.retriever._bm25 is not None
    except Exception:
        count = 0
        bm25_indexed = False
    return HealthResponse(
        status="ok",
        collection_count=count,
        bm25_indexed=bm25_indexed,
    )


# ---------------------------------------------------------------------------
# /query  (streaming SSE)
# ---------------------------------------------------------------------------

@router.post("/query/stream")
def query_stream(req: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    pipeline.k = req.k
    pipeline.alpha = req.alpha

    def event_generator():
        for item in pipeline.query_stream(req.question, temperature=req.temperature):
            if isinstance(item, str):
                yield f"data: {json.dumps({'type': 'token', 'text': item})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'done', **item})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    pipeline.k = req.k
    pipeline.alpha = req.alpha

    answer_parts: list[str] = []
    final_meta = None

    for item in pipeline.query_stream(req.question, temperature=req.temperature):
        if isinstance(item, str):
            answer_parts.append(item)
        else:
            final_meta = item

    if final_meta is None:
        raise HTTPException(status_code=500, detail="Generation failed")

    return QueryResponse(
        answer="".join(answer_parts),
        sources=[SourceRef(**s) for s in final_meta["sources"]],
        chunks=[ChunkRef(**c) for c in final_meta["chunks"]],
        meta=QueryMeta(**final_meta["__meta__"]),
    )


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    if not os.path.isdir(req.corpus_dir):
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.corpus_dir}")

    chunks = ingest_directory(req.corpus_dir, req.chunk_size, req.overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="No .md/.txt files found in directory")

    pipeline.retriever.add_chunks(chunks)

    files = len({c.source for c in chunks})
    return IngestResponse(
        num_chunks=len(chunks),
        files_processed=files,
        message=f"Ingested {len(chunks)} chunks from {files} files",
    )


# ---------------------------------------------------------------------------
# /queries  (observability)
# ---------------------------------------------------------------------------

@router.get("/queries")
def recent_queries(limit: int = 20, pipeline: RAGPipeline = Depends(get_pipeline)):
    return pipeline.recent_queries(limit=limit)
