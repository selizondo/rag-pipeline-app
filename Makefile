.PHONY: install dev api ui demo test eval clean

# --- Local development ---

install:
	pip install -r requirements.txt

api:
	uvicorn api.main:app --reload --port 8000

ui:
	streamlit run ui/app.py --server.port 8501

dev:
	@echo "Starting API and UI in background..."
	uvicorn api.main:app --reload --port 8000 &
	sleep 3
	streamlit run ui/app.py --server.port 8501

# --- Demo (one-command local stack) ---

demo: install
	@echo ""
	@echo "=============================================="
	@echo "  RAG App Demo"
	@echo "  API  → http://localhost:8000/docs"
	@echo "  UI   → http://localhost:8501"
	@echo "=============================================="
	@echo ""
	@echo "Step 1: starting API..."
	uvicorn api.main:app --reload --port 8000 &
	sleep 3
	@echo "Step 2: ingesting demo corpus..."
	curl -s -X POST http://localhost:8000/api/v1/ingest \
	  -H "Content-Type: application/json" \
	  -d "{\"corpus_dir\": \"$$(pwd)/demo/corpus\"}" | python3 -m json.tool
	@echo ""
	@echo "Step 3: starting UI (Ctrl+C to stop)..."
	streamlit run ui/app.py --server.port 8501

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
	python evals/run_evals.py \
	  --cases ../llm-eval-harness/evals/cases/rag_qa.jsonl \
	  --tag rag_app_v1

eval-compare:
	@echo "Usage: make eval-compare COMPARE=<run_id>"
	python evals/run_evals.py \
	  --cases ../llm-eval-harness/evals/cases/rag_qa.jsonl \
	  --tag rag_app_v1 \
	  --compare $(COMPARE)

eval-smoke:
	python evals/run_evals.py \
	  --cases ../llm-eval-harness/evals/cases/rag_qa.jsonl \
	  --tag smoke \
	  --limit 3

# --- Cleanup ---

clean:
	rm -rf chroma_db obs.db __pycache__ **/__pycache__ .pytest_cache
