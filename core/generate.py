"""
Streaming generation over retrieved context.

Supports two backends selected via LLM_PROVIDER env var:
  - "ollama" (default): local Ollama, model configurable via OLLAMA_MODEL
  - "anthropic": Claude via Anthropic SDK, model via ANTHROPIC_MODEL

WHY streaming (Generator instead of returning a full string):
    LLM generation can take 5-30 seconds for longer answers. Without streaming,
    the user sees a blank screen for the entire duration, then the answer appears
    all at once. Research shows users abandon requests after ~3-5 seconds of
    silence. Streaming yields tokens as they are generated, so the user sees
    text appearing immediately — perceived latency drops dramatically even if
    total generation time is the same.

    The FastAPI route (routes.py) converts the Generator into a Server-Sent
    Events (SSE) stream that the Streamlit UI consumes token-by-token.

Error handling:
    Both _stream_ollama and _stream_anthropic yield an error token on failure
    rather than raising. WHY: a generator that raises mid-stream has already
    sent HTTP headers to the client — the SSE connection is open. Raising an
    exception at that point results in a broken stream with no error message
    visible to the user. Yielding "[ERROR: ...]" lets the client display it.
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
    """
    Stream tokens from a local Ollama model.

    WHY requests with stream=True instead of blocking:
        Ollama's /api/generate returns newline-delimited JSON objects, one per
        token. Consuming them with iter_lines() lets us yield each token as it
        arrives rather than waiting for the full response — this is what makes
        the streaming UI feel responsive.

    On error: yields an error token rather than raising so the SSE stream
    closes gracefully instead of producing a broken HTTP response.
    """
    import json

    import requests

    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    url = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
    prompt = _build_prompt(question, chunks)

    try:
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
    except requests.exceptions.ConnectionError:
        yield f"[ERROR: Ollama not reachable at {os.getenv('OLLAMA_URL', 'http://localhost:11434')}. Is it running?]"
        return
    except requests.exceptions.Timeout:
        yield "[ERROR: Ollama request timed out after 120s. Try a shorter question or a smaller model.]"
        return
    except requests.exceptions.HTTPError as e:
        yield f"[ERROR: Ollama returned HTTP {e.response.status_code}: {e.response.text[:200]}]"
        return
    except Exception as e:
        yield f"[ERROR: Ollama generation failed: {e}]"
        return

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
    """
    Stream tokens from Claude via the Anthropic SDK.

    WHY client.messages.stream() instead of messages.create():
        stream() is a context manager that opens an SSE connection to Anthropic's
        API. text_stream yields decoded text tokens as they arrive. This gives the
        same progressive display behaviour as Ollama streaming.

    On error: yields an error token (see module docstring for why).
    """
    import anthropic

    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024"))
    client = anthropic.Anthropic()
    prompt = _build_prompt(question, chunks)

    try:
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text
    except anthropic.AuthenticationError:
        yield "[ERROR: Anthropic API key is invalid or not set. Check ANTHROPIC_API_KEY.]"
    except anthropic.RateLimitError:
        yield "[ERROR: Anthropic rate limit hit. Wait a moment and retry.]"
    except anthropic.APIConnectionError:
        yield "[ERROR: Could not connect to Anthropic API. Check network connectivity.]"
    except Exception as e:
        yield f"[ERROR: Anthropic generation failed: {e}]"
