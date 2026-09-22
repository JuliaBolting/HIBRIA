from __future__ import annotations

import logging
import threading

from pipeline.retrieval.vector_store import Document

from .rag_memory_repository import RagMemoryRepository

logger = logging.getLogger(__name__)


class RagMemoryService:
    """Sincroniza a memória estruturada do PostgreSQL com o índice FAISS."""

    _index_lock = threading.Lock()

    def __init__(self, repository: RagMemoryRepository | None = None) -> None:
        self.repository = repository or RagMemoryRepository()

    def persist_and_index(self, analysis_id: str | None, result, vector_store) -> dict:
        summary = {
            "enabled": self.repository.is_configured,
            "claims_saved": 0,
            "evidence_links_saved": 0,
            "approved_evidences": 0,
            "indexed_evidences": 0,
            "reconciled_evidences": 0,
            "failed_indexing": 0,
        }
        if not self.repository.is_configured:
            return summary
        if not analysis_id:
            summary["error"] = "análise não persistida; memória RAG não foi atualizada"
            return summary

        try:
            persisted = self.repository.persist_analysis(analysis_id, result)
        except Exception as exc:
            logger.warning("[rag_memory] persistência estruturada falhou: %s", exc)
            summary["error"] = str(exc)
            return summary

        summary.update(
            {
                "claims_saved": persisted.claims_saved,
                "evidence_links_saved": persisted.evidence_links_saved,
                "approved_evidences": persisted.approved_evidences,
            }
        )

        if vector_store is None:
            summary["pending_faiss"] = len(persisted.pending_candidates)
            summary["error"] = "FAISS indisponível"
            return summary

        self._index_candidates(
            persisted.pending_candidates,
            vector_store,
            summary,
        )
        return summary

    def sync_pending(self, vector_store, limit: int = 100) -> dict:
        """Tenta novamente evidências aprovadas que ficaram pendentes."""
        summary = {
            "enabled": self.repository.is_configured,
            "candidates": 0,
            "indexed_evidences": 0,
            "reconciled_evidences": 0,
            "failed_indexing": 0,
            "pending_faiss": 0,
        }
        if not self.repository.is_configured or vector_store is None:
            return summary

        candidates = self.repository.pending_candidates(limit=limit)
        summary["candidates"] = len(candidates)
        self._index_candidates(candidates, vector_store, summary)
        return summary

    def _index_candidates(self, candidates, vector_store, summary: dict) -> None:
        # O lock evita duas gravações simultâneas no index.faiss no mesmo
        # processo. Em produção, mantenha um único worker para esta instância.
        with self._index_lock:
            for candidate in candidates:
                base_doc_id = f"rag-evidence-{candidate.evidence_id}"
                try:
                    if vector_store.contains_document(base_doc_id):
                        self.repository.mark_indexed(
                            candidate.evidence_id,
                            base_doc_id,
                        )
                        summary["reconciled_evidences"] += 1
                        continue

                    vector_store.add_document(
                        Document(
                            text=candidate.text,
                            source=candidate.source,
                            url=candidate.url,
                            doc_id=base_doc_id,
                            published_at=candidate.published_at,
                            metadata={
                                **candidate.metadata,
                                "evidence_database_id": candidate.evidence_id,
                            },
                        )
                    )
                    self.repository.mark_indexed(
                        candidate.evidence_id,
                        base_doc_id,
                    )
                    summary["indexed_evidences"] += 1
                except Exception as exc:
                    logger.warning(
                        "[rag_memory] indexação FAISS falhou para %s: %s",
                        candidate.evidence_id,
                        exc,
                    )
                    self.repository.mark_index_failed(
                        candidate.evidence_id,
                        str(exc),
                    )
                    summary["failed_indexing"] += 1

        summary["pending_faiss"] = max(
            0,
            len(candidates)
            - summary["indexed_evidences"]
            - summary["reconciled_evidences"]
        )
