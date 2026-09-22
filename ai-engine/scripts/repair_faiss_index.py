"""Diagnostica ou reconstrói o FAISS a partir do metadata.json.

Por padrão não altera nada. Use --repair para reconstruir. O script sempre
cria backup antes da troca e --drop-fakebr remove o corpus Fake.Br que o
retriever da HÍBRIA já ignora como evidência factual.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

DEFAULT_STORE = ROOT_DIR / "data" / "vector_store"


def is_fakebr(item: dict) -> bool:
    metadata = item.get("metadata", {}) or {}
    return bool(
        metadata.get("source_type") == "labeled_dataset"
        or metadata.get("corpus") == "fake.br"
        or str(item.get("source", "")).lower().startswith("fake.br")
        or str(item.get("url", "")).lower().startswith("fakebr://")
    )


def load_metadata(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("metadata.json não contém uma lista")
    return [item for item in payload if isinstance(item, dict)]


def diagnose(store_path: Path, metadata: list[dict]) -> tuple[int, int]:
    import faiss

    index_path = store_path / "index.faiss"
    index = faiss.read_index(str(index_path))
    vectors = int(index.ntotal)
    rows = len(metadata)
    print(f"Vetores FAISS: {vectors}")
    print(f"Linhas metadata.json: {rows}")
    print(f"Vetores órfãos/diferença: {vectors - rows}")
    print(f"Itens Fake.Br: {sum(1 for item in metadata if is_fakebr(item))}")
    print("Consistência:", "OK" if vectors == rows else "INCONSISTENTE")
    return vectors, rows


def rebuild(store_path: Path, metadata: list[dict], drop_fakebr: bool) -> None:
    import faiss

    from pipeline.analysis.embeddings import EmbeddingModel

    selected = [item for item in metadata if item.get("text")]
    if drop_fakebr:
        selected = [item for item in selected if not is_fakebr(item)]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = store_path / "backups" / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(store_path / "index.faiss", backup_dir / "index.faiss")
    shutil.copy2(store_path / "metadata.json", backup_dir / "metadata.json")

    model = EmbeddingModel("multilingual-minilm")
    texts = [str(item["text"]).strip() for item in selected]
    vectors = model.embed_batch(
        texts,
        normalize=True,
        batch_size=64,
        show_progress=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(model.dims)
    if len(vectors):
        index.add(vectors)

    repaired_metadata = []
    for position, item in enumerate(selected):
        repaired = dict(item)
        repaired["faiss_id"] = position
        repaired_metadata.append(repaired)

    temp_index = store_path / "index.faiss.repairing"
    temp_metadata = store_path / "metadata.json.repairing"
    faiss.write_index(index, str(temp_index))
    temp_metadata.write_text(
        json.dumps(repaired_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_index.replace(store_path / "index.faiss")
    temp_metadata.replace(store_path / "metadata.json")

    print(f"Backup: {backup_dir}")
    print(f"Reconstruído: {len(repaired_metadata)} vetores/metadados")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--drop-fakebr", action="store_true")
    args = parser.parse_args()

    index_path = args.store / "index.faiss"
    metadata_path = args.store / "metadata.json"
    if not index_path.exists() or not metadata_path.exists():
        raise SystemExit(f"Índice incompleto em {args.store}")

    metadata = load_metadata(metadata_path)
    diagnose(args.store, metadata)
    if not args.repair:
        print("Nenhuma alteração feita. Para corrigir, acrescente --repair.")
        return

    rebuild(args.store, metadata, drop_fakebr=args.drop_fakebr)
    repaired = load_metadata(metadata_path)
    diagnose(args.store, repaired)


if __name__ == "__main__":
    main()
