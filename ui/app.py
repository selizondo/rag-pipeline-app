"""
Streamlit frontend — streams token-by-token from the FastAPI backend.

Run:  streamlit run ui/app.py
      (FastAPI must be running on API_URL, default http://localhost:8000)
"""

from __future__ import annotations

import json
import os
import time
from typing import Generator

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stream_query(question: str, k: int, alpha: float, temperature: float) -> Generator[dict, None, None]:
    """Yield token dicts and a final done dict from the SSE stream."""
    payload = {"question": question, "k": k, "alpha": alpha, "temperature": temperature}
    try:
        with requests.post(
            f"{API_URL}/query/stream",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    yield data
    except requests.exceptions.ConnectionError:
        yield {"type": "error", "text": "Cannot connect to API server. Is it running?"}
    except Exception as e:
        yield {"type": "error", "text": str(e)}


def get_health() -> dict:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return {"status": "unreachable", "collection_count": 0, "bm25_indexed": False}


def format_latency(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms/1000:.1f} s"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    k = st.slider("Retrieval depth (k)", min_value=1, max_value=20, value=5)
    alpha = st.slider(
        "Vector weight (α)",
        min_value=0.0, max_value=1.0, value=0.7, step=0.05,
        help="0 = BM25 only · 1 = vector only · 0.7 = recommended",
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.divider()
    st.subheader("System Status")
    health = get_health()
    status_color = "🟢" if health.get("status") == "ok" else "🔴"
    st.write(f"{status_color} API: **{health.get('status', 'unknown')}**")
    st.write(f"📚 Chunks indexed: **{health.get('collection_count', 0):,}**")
    bm25_ok = health.get("bm25_indexed", False)
    st.write(f"{'✅' if bm25_ok else '⚠️'} BM25 index: **{'ready' if bm25_ok else 'not built'}**")

    st.divider()
    st.subheader("Ingest Corpus")
    corpus_dir = st.text_input("Corpus directory", value="./demo/corpus")
    if st.button("Ingest"):
        with st.spinner("Ingesting..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ingest",
                    json={"corpus_dir": os.path.abspath(corpus_dir)},
                    timeout=120,
                )
                result = resp.json()
                if resp.ok:
                    st.success(result.get("message", "Done"))
                else:
                    st.error(result.get("detail", "Error"))
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.caption("RAG App · rag-pipeline-app")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("🤖 RAG Assistant")
st.caption("Hybrid BM25 + vector search · Streaming · Observability")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    st.write(f"**{src['source']}**" + (f" § {src['section']}" if src.get("section") else "") + f"  ·  score `{src['score']}`")
        if msg.get("meta"):
            m = msg["meta"]
            st.caption(
                f"⏱ {format_latency(m['latency_ms'])} total  "
                f"(retrieval {format_latency(m['retrieval_ms'])} · generation {format_latency(m['generation_ms'])})  "
                f"· {m['num_chunks']} chunks"
            )

# Input
if question := st.chat_input("Ask a question about AI/ML..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        sources_placeholder = st.empty()
        meta_placeholder = st.empty()

        full_answer = ""
        sources = []
        meta_data = None
        error = None

        for event in stream_query(question, k=k, alpha=alpha, temperature=temperature):
            etype = event.get("type")
            if etype == "token":
                full_answer += event.get("text", "")
                answer_placeholder.markdown(full_answer + "▌")
            elif etype == "done":
                answer_placeholder.markdown(full_answer)
                sources = event.get("sources", [])
                meta_data = event.get("__meta__", {})

                if sources:
                    with sources_placeholder.expander("Sources", expanded=True):
                        for src in sources:
                            label = f"**{src['source']}**"
                            if src.get("section"):
                                label += f" § {src['section']}"
                            st.write(f"{label}  ·  score `{src['score']}`")

                if meta_data:
                    meta_placeholder.caption(
                        f"⏱ {format_latency(meta_data['latency_ms'])} total  "
                        f"(retrieval {format_latency(meta_data['retrieval_ms'])} · "
                        f"generation {format_latency(meta_data['generation_ms'])})  "
                        f"· {meta_data['num_chunks']} chunks"
                    )
            elif etype == "error":
                error = event.get("text", "Unknown error")
                answer_placeholder.error(error)
                break

        if not error:
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources,
                "meta": meta_data,
            })


# ---------------------------------------------------------------------------
# Observability tab
# ---------------------------------------------------------------------------

st.divider()
with st.expander("📊 Query History (last 20)", expanded=False):
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        # Fetch query history via a simple approach — show from sidebar state
        st.info("Run `SELECT * FROM queries ORDER BY id DESC` on obs.db for full history.")
    except Exception:
        st.warning("API unavailable")
