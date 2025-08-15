"""
Environment loading: project-level .env overrides the workspace master .env.

Load order (later values win):
  1. career/.env  — shared API keys across all projects
  2. rag-pipeline-app/.env — project-specific overrides (optional)
"""

from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # rag-pipeline-app/
_MASTER_ENV = _PROJECT_ROOT.parent.parent / ".env"      # career/.env


def load_env() -> None:
    load_dotenv(_MASTER_ENV)             # shared keys first
    load_dotenv(_PROJECT_ROOT / ".env")  # local overrides master
