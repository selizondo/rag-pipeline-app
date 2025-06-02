"""
Streaming generation over retrieved context.

Supports two backends selected via LLM_PROVIDER env var:
  - "ollama" (default): local Ollama, model configurable via OLLAMA_MODEL
  - "anthropic": Claude via Anthropic SDK, model via ANTHROPIC_MODEL
"""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieve import RetrievedChunk

_SYSTEM = (
    "You are a knowledgeable AI/ML assistant. Answer the question using only the "
    "provided context. If the context does not contain enough information, say so — "
    "do not guess or hallucinate. Be concise and precise."
)

_PROMPT_TMPL = """\
Context:
{context}

Question: {question}

Answer:"""


def _build_prompt(question: str, chunks: list["RetrievedChunk"]) -> str:
    ctx_parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c.source}"
        if c.section:
            header += f" § {c.section}"
        ctx_parts.append(f"{header}\n{c.text}")
    context = "\n\n".join(ctx_parts)
    return _PROMPT_TMPL.format(context=context, question=question)


def stream_answer(
    question: str,
    chunks: list["RetrievedChunk"],
    temperature: float = 0.2,
) -> Generator[str, None, None]:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "anthropic":
        yield from _stream_anthropic(question, chunks, temperature)
    else:
        yield from _stream_ollama(question, chunks, temperature)


def _stream_ollama(
    question: str,
    chunks: list["RetrievedChunk"],
    temperature: float,
) -> Generator[str, None, None]:
    import requests

    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    url = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
    prompt = _build_prompt(question, chunks)

    resp = requests.post(
        url,
        json={
            "model": model,
            "prompt": f"{_SYSTEM}\n\n{prompt}",
            "stream": True,
            "options": {"temperature": temperature},
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    import json
    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        token = data.get("response", "")
        if token:
            yield token
        if data.get("done"):
            break


def _stream_anthropic(
    question: str,
    chunks: list["RetrievedChunk"],
    temperature: float,
) -> Generator[str, None, None]:
    import anthropic

    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    client = anthropic.Anthropic()
    prompt = _build_prompt(question, chunks)

    with client.messages.stream(
        model=model,
        max_tokens=1024,
        temperature=temperature,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text
