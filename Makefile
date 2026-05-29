.PHONY: bootstrap test dev api ui demo eval eval-save eval-compare eval-smoke clean

# --- Local development ---

bootstrap:
	UV_PROJECT_ENVIRONMENT=.venv uv sync

test:
	uv run pytest

api:
	uv run uvicorn api.main:app --reload --port 8000

ui:
	uv run streamlit run ui/app.py --server.port 8501

dev:
	@echo "Starting API and UI in background..."
	uv run uvicorn api.main:app --reload --port 8000 &
	sleep 3
	uv run streamlit run ui/app.py --server.port 8501

# --- Demo (one-command local stack) ---

demo: bootstrap
	@echo ""
	@echo "=============================================="
	@echo "  RAG App Demo"
	@echo "  API  → http://localhost:8000/docs"
	@echo "  UI   → http://localhost:8501"
	@echo "=============================================="
	@echo ""
	@echo "Step 1: starting API..."
	uv run uvicorn api.main:app --reload --port 8000 &
	sleep 3
	@echo "Step 2: ingesting demo corpus..."
	curl -s -X POST http://localhost:8000/api/v1/ingest \
	  -H "Content-Type: application/json" \
	  -d "{\"corpus_dir\": \"$$(pwd)/demo/corpus\"}" | python3 -m json.tool
	@echo ""
	@echo "Step 3: starting UI (Ctrl+C to stop)..."
	uv run streamlit run ui/app.py --server.port 8501

# --- Docker ---

docker-build:
	docker compose build

docker-up:
	ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) docker compose up

docker-down:
	docker compose down

docker-ingest:
	curl -s -X POST http://localhost:8000/api/v1/ingest \
	  -H "Content-Type: application/json" \
	  -d '{"corpus_dir": "/app/demo/corpus"}' | python3 -m json.tool

# --- Evals ---

eval:
	uv run python evals/run_evals.py \
	  --cases ../llm-eval-harness/evals/cases/rag_qa.jsonl \
	  --tag rag_app_v1

eval-save:
	mkdir -p artifacts/eval
	uv run python evals/run_evals.py \
	  --cases ../llm-eval-harness/evals/cases/rag_qa.jsonl \
	  --tag rag_app_v1 \
	  --output artifacts/eval/latest_run.json

eval-compare:
	@echo "Usage: make eval-compare COMPARE=<run_id>"
	uv run python evals/run_evals.py \
	  --cases ../llm-eval-harness/evals/cases/rag_qa.jsonl \
	  --tag rag_app_v1 \
	  --compare $(COMPARE)

eval-smoke:
	uv run python evals/run_evals.py \
	  --cases ../llm-eval-harness/evals/cases/rag_qa.jsonl \
	  --tag smoke \
	  --limit 3

# --- Cleanup ---

clean:
	rm -rf chroma_db obs.db __pycache__ **/__pycache__ .pytest_cache
