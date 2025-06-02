"""
Enhanced ingestion: word-based chunking with per-chunk metadata.
Stores source filename, section header, and chunk index alongside text.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    doc_id: str
    source: str          # filename
    section: str         # nearest ## heading, or ""
    chunk_index: int
    char_start: int
    metadata: dict = field(default_factory=dict)


def _word_chunks(text: str, size: int, overlap: int) -> list[tuple[str, int]]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk_words = words[start:end]
        char_start = len(" ".join(words[:start]))
        chunks.append((" ".join(chunk_words), char_start))
        if end == len(words):
            break
        start += size - overlap
    return chunks


def _current_section(text_before: str) -> str:
    headers = re.findall(r"^#{1,3} (.+)$", text_before, re.MULTILINE)
    return headers[-1].strip() if headers else ""


def ingest_file(
    path: str,
    chunk_size: int = 256,
    overlap: int = 32,
) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        text = f.read()

    source = os.path.basename(path)
    raw_chunks = _word_chunks(text, chunk_size, overlap)
    chunks = []
    for i, (chunk_text, char_start) in enumerate(raw_chunks):
        section = _current_section(text[:char_start])
        doc_id = f"{source}::{i}"
        chunks.append(
            Chunk(
                text=chunk_text,
                doc_id=doc_id,
                source=source,
                section=section,
                chunk_index=i,
                char_start=char_start,
            )
        )
    return chunks


def ingest_directory(
    corpus_dir: str,
    chunk_size: int = 256,
    overlap: int = 32,
    extensions: tuple[str, ...] = (".md", ".txt"),
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for fname in sorted(os.listdir(corpus_dir)):
        if not any(fname.endswith(ext) for ext in extensions):
            continue
        fpath = os.path.join(corpus_dir, fname)
        all_chunks.extend(ingest_file(fpath, chunk_size, overlap))
    return all_chunks
