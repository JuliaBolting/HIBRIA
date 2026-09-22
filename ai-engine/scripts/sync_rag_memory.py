"""Reenvia ao FAISS evidências aprovadas que ficaram pendentes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")

from pipeline.persistence.rag_memory_service import RagMemoryService
from pipeline.retrieval.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    summary = RagMemoryService().sync_pending(
        VectorStore(),
        limit=max(1, args.limit),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
