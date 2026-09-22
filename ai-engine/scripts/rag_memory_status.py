"""Mostra o estado conjunto do PostgreSQL e do FAISS."""

from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")


def print_postgres() -> None:
    database_url = os.getenv("HIBRIA_DATABASE_URL") or os.getenv("DATABASE_URL")
    print("POSTGRESQL")
    if not database_url:
        print("  não configurado (HIBRIA_DATABASE_URL/DATABASE_URL ausente)")
        return

    try:
        import psycopg2

        query = """
            SELECT
                (SELECT COUNT(*) FROM analises) AS analises,
                (SELECT COUNT(*) FROM claims_analise) AS claims,
                (SELECT COUNT(*) FROM evidencias_rag) AS evidencias,
                (SELECT COUNT(*) FROM claim_evidencias) AS relacoes,
                (SELECT COUNT(*) FROM evidencias_rag WHERE aprovada_para_rag) AS aprovadas,
                (SELECT COUNT(*) FROM evidencias_rag WHERE faiss_indexada_em IS NOT NULL) AS indexadas,
                (SELECT COUNT(*) FROM evidencias_rag
                  WHERE aprovada_para_rag AND faiss_indexada_em IS NULL) AS pendentes;
        """
        with psycopg2.connect(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                row = cursor.fetchone()
        labels = (
            "análises",
            "claims",
            "evidências únicas",
            "relações claim-evidência",
            "evidências aprovadas",
            "evidências indexadas",
            "evidências pendentes",
        )
        for label, value in zip(labels, row):
            print(f"  {label}: {value}")
    except Exception as exc:
        print(f"  erro: {exc}")


def print_faiss() -> None:
    print("\nFAISS")
    try:
        from pipeline.retrieval.vector_store import VectorStore

        store = VectorStore()
        stats = store.stats()
        metadata_count = len(getattr(store, "_metadata", []))
        print(f"  vetores: {stats['total_chunks']}")
        print(f"  metadados: {metadata_count}")
        print(f"  documentos únicos: {stats['unique_documents']}")
        print(f"  dimensões: {stats['embedding_dims']}")
        print(f"  tamanho: {stats['index_size_mb']} MB")
        print(
            "  consistência: "
            + ("OK" if stats["total_chunks"] == metadata_count else "INCONSISTENTE")
        )
        print(f"  fontes: {stats['sources']}")
    except Exception as exc:
        print(f"  erro: {exc}")


if __name__ == "__main__":
    print_postgres()
    print_faiss()
