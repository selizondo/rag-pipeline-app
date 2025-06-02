"""FastAPI application entry point."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.retrieve import HybridRetriever
from core.pipeline import RAGPipeline

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION = os.getenv("COLLECTION_NAME", "rag_corpus")
OBS_DB = os.getenv("OBS_DB_PATH", "./obs.db")

pipeline: RAGPipeline = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    retriever = HybridRetriever(
        chroma_path=CHROMA_PATH,
        collection_name=COLLECTION,
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        db_path=OBS_DB,
        k=int(os.getenv("DEFAULT_K", "5")),
        alpha=float(os.getenv("DEFAULT_ALPHA", "0.7")),
    )
    yield
    # cleanup (nothing to close for sqlite/chroma)


app = FastAPI(
    title="RAG App API",
    version="1.0.0",
    description="Hybrid BM25+vector RAG pipeline with streaming and observability",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes import router  # noqa: E402  (after app is created)
app.include_router(router, prefix="/api/v1")
