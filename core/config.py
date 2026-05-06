"""
Environment loading: loads .env from the project root.
"""

from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # rag-pipeline-app/


def load_env() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")
