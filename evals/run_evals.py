"""
Eval integration — wires the llm-eval-harness to this RAG pipeline.

Usage:
    python evals/run_evals.py --cases ../llm-eval-harness/evals/cases/rag_qa.jsonl
    python evals/run_evals.py --cases ../llm-eval-harness/evals/cases/rag_qa.jsonl --compare <run_id>
    python evals/run_evals.py --list-runs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add llm-eval-harness to path — assumes sibling directory layout
HARNESS_DIR = Path(__file__).parent.parent.parent / "llm-eval-harness"
if HARNESS_DIR.exists():
    sys.path.insert(0, str(HARNESS_DIR))
else:
    print(f"WARNING: eval harness not found at {HARNESS_DIR}")
    print("  Clone it: git clone https://github.com/selizondo/llm-eval-harness.git")

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")


def rag_model_fn(question: str) -> str:
    """Callable that wraps the local FastAPI /query endpoint."""
    resp = requests.post(
        f"{API_URL}/query",
        json={"question": question, "k": 5, "alpha": 0.7, "temperature": 0.1},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["answer"]


def main():
    parser = argparse.ArgumentParser(description="Run evals against the RAG App pipeline")
    parser.add_argument("--cases", required=False, help="Path to .jsonl test cases file")
    parser.add_argument("--tag", default="rag_app", help="Run tag for comparison")
    parser.add_argument("--compare", default=None, help="Run ID to compare against")
    parser.add_argument("--limit", type=int, default=None, help="Max cases to run (smoke test)")
    parser.add_argument("--list-runs", action="store_true", help="List all historical runs")
    parser.add_argument("--show-cases", default=None, metavar="RUN_ID", help="Show per-case breakdown")
    args = parser.parse_args()

    try:
        from evals.harness import run_eval
        from evals.metrics import compute_summary
        from evals.dashboard import print_summary, print_cases
    except ImportError as e:
        print(f"ERROR: Could not import eval harness: {e}")
        print(f"  Ensure llm-eval-harness is at: {HARNESS_DIR}")
        sys.exit(1)

    if args.list_runs:
        from evals.harness import list_runs
        list_runs()
        return

    if args.show_cases:
        from evals.harness import show_cases
        show_cases(args.show_cases)
        return

    if not args.cases:
        parser.error("--cases is required unless using --list-runs or --show-cases")

    # Verify API is reachable
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        health = resp.json()
        if health.get("collection_count", 0) == 0:
            print("WARNING: Chroma collection is empty — ingest a corpus first.")
            print(f"  POST {API_URL}/ingest  {{\"corpus_dir\": \"./demo/corpus\"}}")
    except Exception as e:
        print(f"ERROR: Cannot reach API at {API_URL}: {e}")
        print("  Start it with: uvicorn api.main:app --reload")
        sys.exit(1)

    run_id = run_eval(
        cases_path=args.cases,
        model_fn=rag_model_fn,
        model_tag=args.tag,
        limit=args.limit,
        compare_run_id=args.compare,
    )

    summary = compute_summary(run_id, compare_run_id=args.compare)
    print_summary(summary)

    print(f"\nRun ID: {run_id}")
    print(f"  python evals/run_evals.py --show-cases {run_id}")


if __name__ == "__main__":
    main()
