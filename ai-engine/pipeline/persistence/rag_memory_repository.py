from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
import re
from typing import Any

from .analysis_repository import AnalysisRepository

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "sim", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _score(value: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _json(payload: Any):
    import psycopg2.extras

    return psycopg2.extras.Json(
        payload,
        dumps=lambda value: json.dumps(value, ensure_ascii=False, default=str),
    )


@dataclass(frozen=True)
class RagEvidenceCandidate:
    evidence_id: str
    text: str
    source: str
    url: str
    published_at: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RagPersistenceResult:
    claims_saved: int
    evidence_links_saved: int
    approved_evidences: int
    pending_candidates: tuple[RagEvidenceCandidate, ...]


class RagMemoryRepository:
    """Persiste a memória auditável do RAG e controla a fila do FAISS."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = (
            database_url
            if database_url is not None
            else os.getenv("HIBRIA_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or ""
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.database_url)

    @staticmethod
    def _evidence_key(url: str, text: str) -> str:
        normalized_url = (
            AnalysisRepository.normalize_url(url) if url.strip() else ""
        )
        return _sha256(f"{normalized_url}|{_clean_text(text)}")

    @staticmethod
    def _source_reputation(cursor, domain: str) -> float | None:
        if not domain:
            return None

        cursor.execute(
            """
            SELECT fr.score_reputacao
            FROM fontes_reputacao fr
            LEFT JOIN aliases_fontes_reputacao afr
              ON afr.fonte_id = fr.id
             AND afr.ativo = TRUE
            WHERE fr.dominio_canonico = %(domain)s
               OR afr.dominio_alias = %(domain)s
            ORDER BY fr.data_ultima_verificacao DESC
            LIMIT 1;
            """,
            {"domain": domain.lower()},
        )
        row = cursor.fetchone()
        if not row:
            return None
        return float(row[0])

    @classmethod
    def _approval(
        cls,
        evidence,
        source_reputation: float | None,
    ) -> tuple[bool, str]:
        """Aplica a política que impede conteúdo fraco de contaminar o FAISS."""
        if not _env_flag("HIBRIA_RAG_MEMORY_ENABLED", True):
            return False, "memória RAG desativada"

        layer = _clean_text(getattr(evidence, "retrieval_layer", "")).lower()
        source_type = _clean_text(getattr(evidence, "source_type", "")).lower()
        source = _clean_text(getattr(evidence, "source", "")).lower()
        url = _clean_text(getattr(evidence, "url", "")).lower()
        text = _clean_text(getattr(evidence, "text", ""))
        stance = _clean_text(getattr(evidence, "stance", "")).lower()
        metadata = getattr(evidence, "metadata", {}) or {}
        similarity = _score(metadata.get("stance_similarity"))

        if layer == "vector_store":
            return False, "já veio do FAISS"
        if layer in {"ai_fallback", "wikipedia"}:
            return False, f"camada {layer} não é prova primária"
        if source_type in {"encyclopedia", "labeled_dataset"}:
            return False, f"tipo de fonte {source_type} não é elegível"
        if "fake.br" in source or url.startswith("fakebr:"):
            return False, "dataset rotulado não é evidência externa"

        min_chars = _env_int("HIBRIA_RAG_MEMORY_MIN_CHARS", 80)
        if len(text) < max(1, min_chars):
            return False, "trecho curto demais"
        if not url:
            return False, "evidência sem URL rastreável"
        if stance not in {"support", "contradict"}:
            return False, "stance não confirma nem contradiz"

        min_similarity = _env_float("HIBRIA_RAG_MEMORY_MIN_SIMILARITY", 0.55)
        if similarity is None or similarity < min_similarity:
            return False, "similaridade final abaixo do limite"

        trusted = bool(getattr(evidence, "trusted_source", False))
        is_factcheck = layer == "factcheck" or source_type == "fact_check"
        if trusted:
            return True, "fonte marcada como confiável"
        if is_factcheck:
            return True, "evidência de serviço de fact-check"

        require_reputation = _env_flag(
            "HIBRIA_RAG_MEMORY_REQUIRE_SOURCE_REPUTATION",
            True,
        )
        if not require_reputation:
            return True, "política permite fonte ainda não avaliada"

        min_reputation = _env_float(
            "HIBRIA_RAG_MEMORY_MIN_SOURCE_REPUTATION",
            0.80,
        )
        if source_reputation is not None and source_reputation >= min_reputation:
            return True, "domínio possui reputação suficiente"

        return False, "fonte sem reputação suficiente"

    def persist_analysis(self, analysis_id: str, result) -> RagPersistenceResult:
        if not self.is_configured or not analysis_id:
            return RagPersistenceResult(0, 0, 0, ())

        import psycopg2

        claim_sql = """
            INSERT INTO claims_analise (
                analise_id, claim_id_pipeline, ordem, texto,
                texto_normalizado, hash_claim, sujeito, confianca,
                entidades_json, metadata_json
            ) VALUES (
                %(analysis_id)s, %(pipeline_id)s, %(position)s, %(text)s,
                %(normalized)s, %(claim_hash)s, %(subject)s, %(confidence)s,
                %(entities)s, %(metadata)s
            )
            ON CONFLICT (analise_id, claim_id_pipeline)
            DO UPDATE SET
                ordem = EXCLUDED.ordem,
                texto = EXCLUDED.texto,
                texto_normalizado = EXCLUDED.texto_normalizado,
                sujeito = EXCLUDED.sujeito,
                confianca = EXCLUDED.confianca,
                entidades_json = EXCLUDED.entidades_json,
                metadata_json = EXCLUDED.metadata_json
            RETURNING id;
        """
        evidence_sql = """
            INSERT INTO evidencias_rag (
                evidence_key, url, url_normalizada, hash_url, titulo,
                trecho, hash_conteudo, fonte, dominio, tipo_fonte,
                data_publicacao, fonte_confiavel, aprovada_para_rag,
                motivo_aprovacao, metadata_json
            ) VALUES (
                %(evidence_key)s, %(url)s, %(normalized_url)s, %(url_hash)s,
                %(title)s, %(text)s, %(content_hash)s, %(source)s,
                %(domain)s, %(source_type)s, %(published_at)s,
                %(trusted)s, %(approved)s, %(approval_reason)s, %(metadata)s
            )
            ON CONFLICT (evidence_key)
            DO UPDATE SET
                titulo = COALESCE(NULLIF(EXCLUDED.titulo, ''), evidencias_rag.titulo),
                fonte = COALESCE(NULLIF(EXCLUDED.fonte, ''), evidencias_rag.fonte),
                dominio = COALESCE(NULLIF(EXCLUDED.dominio, ''), evidencias_rag.dominio),
                tipo_fonte = COALESCE(NULLIF(EXCLUDED.tipo_fonte, ''), evidencias_rag.tipo_fonte),
                data_publicacao = COALESCE(EXCLUDED.data_publicacao, evidencias_rag.data_publicacao),
                fonte_confiavel = evidencias_rag.fonte_confiavel OR EXCLUDED.fonte_confiavel,
                aprovada_para_rag = evidencias_rag.aprovada_para_rag OR EXCLUDED.aprovada_para_rag,
                motivo_aprovacao = CASE
                    WHEN EXCLUDED.aprovada_para_rag THEN EXCLUDED.motivo_aprovacao
                    ELSE evidencias_rag.motivo_aprovacao
                END,
                metadata_json = evidencias_rag.metadata_json || EXCLUDED.metadata_json,
                ultima_observacao_em = NOW(),
                quantidade_usos = evidencias_rag.quantidade_usos + 1
            RETURNING id, aprovada_para_rag, faiss_doc_id;
        """
        link_sql = """
            INSERT INTO claim_evidencias (
                analise_id, claim_id, evidencia_id, evidence_id_pipeline,
                camada_recuperacao, ordem, similaridade_retriever,
                similaridade_final, stance, stance_confianca,
                stance_motivo, metadata_json
            ) VALUES (
                %(analysis_id)s, %(claim_id)s, %(evidence_id)s,
                %(pipeline_evidence_id)s, %(layer)s, %(position)s,
                %(retriever_similarity)s, %(final_similarity)s,
                %(stance)s, %(stance_confidence)s, %(stance_reason)s,
                %(metadata)s
            )
            ON CONFLICT (analise_id, claim_id, evidencia_id)
            DO UPDATE SET
                ordem = EXCLUDED.ordem,
                similaridade_retriever = EXCLUDED.similaridade_retriever,
                similaridade_final = EXCLUDED.similaridade_final,
                stance = EXCLUDED.stance,
                stance_confianca = EXCLUDED.stance_confianca,
                stance_motivo = EXCLUDED.stance_motivo,
                metadata_json = EXCLUDED.metadata_json;
        """

        claims_saved = 0
        links_saved = 0
        approved_ids: set[str] = set()
        pending: dict[str, RagEvidenceCandidate] = {}

        with psycopg2.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                for claim_position, retrieval in enumerate(
                    getattr(result, "retrieval_results", None) or [],
                    start=1,
                ):
                    claim = getattr(retrieval, "claim", None)
                    if claim is None:
                        continue

                    pipeline_claim_id = _clean_text(
                        getattr(claim, "claim_id", "")
                    ) or f"claim-{claim_position}"
                    claim_text = _clean_text(getattr(claim, "text", ""))
                    if not claim_text:
                        continue

                    claim_metadata = {
                        "entity_types": getattr(claim, "entity_types", {}) or {},
                        "predicate": getattr(claim, "predicate", ""),
                        "claim_type": str(getattr(claim, "claim_type", "")),
                        "has_numbers": bool(getattr(claim, "has_numbers", False)),
                        "keywords": getattr(claim, "keywords", []) or [],
                        "layers_used": getattr(retrieval, "layers_used", []) or [],
                        "layers_failed": getattr(retrieval, "layers_failed", {}) or {},
                        "retrieval_time": getattr(retrieval, "retrieval_time", None),
                    }
                    cursor.execute(
                        claim_sql,
                        {
                            "analysis_id": analysis_id,
                            "pipeline_id": pipeline_claim_id,
                            "position": claim_position,
                            "text": claim_text,
                            "normalized": _clean_text(
                                getattr(claim, "normalized", "")
                            ),
                            "claim_hash": _sha256(claim_text.lower()),
                            "subject": _clean_text(getattr(claim, "subject", "")),
                            "confidence": _score(getattr(claim, "confidence", None)),
                            "entities": _json(getattr(claim, "entities", []) or []),
                            "metadata": _json(claim_metadata),
                        },
                    )
                    claim_database_id = str(cursor.fetchone()[0])
                    claims_saved += 1

                    for evidence_position, evidence in enumerate(
                        getattr(retrieval, "evidences", []) or [],
                        start=1,
                    ):
                        text = _clean_text(getattr(evidence, "text", ""))
                        if not text:
                            continue

                        url = _clean_text(getattr(evidence, "url", ""))
                        normalized_url = (
                            AnalysisRepository.normalize_url(url) if url else ""
                        )
                        domain = _clean_text(getattr(evidence, "domain", "")).lower()
                        reputation = self._source_reputation(cursor, domain)
                        approved, approval_reason = self._approval(
                            evidence,
                            reputation,
                        )
                        evidence_metadata = dict(
                            getattr(evidence, "metadata", {}) or {}
                        )
                        evidence_metadata["source_reputation"] = reputation

                        cursor.execute(
                            evidence_sql,
                            {
                                "evidence_key": self._evidence_key(url, text),
                                "url": url or None,
                                "normalized_url": normalized_url or None,
                                "url_hash": _sha256(normalized_url) if normalized_url else None,
                                "title": _clean_text(getattr(evidence, "title", "")),
                                "text": text,
                                "content_hash": _sha256(text),
                                "source": _clean_text(getattr(evidence, "source", "")),
                                "domain": domain,
                                "source_type": _clean_text(
                                    getattr(evidence, "source_type", "")
                                ),
                                "published_at": getattr(evidence, "published_at", None),
                                "trusted": bool(
                                    getattr(evidence, "trusted_source", False)
                                ),
                                "approved": approved,
                                "approval_reason": approval_reason,
                                "metadata": _json(evidence_metadata),
                            },
                        )
                        evidence_row = cursor.fetchone()
                        evidence_database_id = str(evidence_row[0])
                        stored_approved = bool(evidence_row[1])
                        faiss_doc_id = evidence_row[2]
                        if stored_approved:
                            approved_ids.add(evidence_database_id)

                        final_similarity = _score(
                            evidence_metadata.get("stance_similarity")
                        )
                        cursor.execute(
                            link_sql,
                            {
                                "analysis_id": analysis_id,
                                "claim_id": claim_database_id,
                                "evidence_id": evidence_database_id,
                                "pipeline_evidence_id": _clean_text(
                                    getattr(evidence, "evidence_id", "")
                                ),
                                "layer": _clean_text(
                                    getattr(evidence, "retrieval_layer", "")
                                ),
                                "position": evidence_position,
                                "retriever_similarity": _score(
                                    getattr(evidence, "similarity", None)
                                ),
                                "final_similarity": final_similarity,
                                "stance": _clean_text(
                                    getattr(evidence, "stance", "")
                                ) or None,
                                "stance_confidence": _score(
                                    evidence_metadata.get("stance_confidence")
                                ),
                                "stance_reason": evidence_metadata.get("stance_reason"),
                                "metadata": _json(evidence_metadata),
                            },
                        )
                        links_saved += 1

                        if stored_approved and not faiss_doc_id:
                            pending[evidence_database_id] = RagEvidenceCandidate(
                                evidence_id=evidence_database_id,
                                text=text,
                                source=_clean_text(
                                    getattr(evidence, "source", "")
                                ) or domain,
                                url=url,
                                published_at=getattr(evidence, "published_at", None),
                                metadata={
                                    **evidence_metadata,
                                    "domain": domain,
                                    "source_type": _clean_text(
                                        getattr(evidence, "source_type", "")
                                    ),
                                    "trusted_source": bool(
                                        getattr(evidence, "trusted_source", False)
                                    ),
                                    "retrieval_layer": _clean_text(
                                        getattr(evidence, "retrieval_layer", "")
                                    ),
                                    "stance": _clean_text(
                                        getattr(evidence, "stance", "")
                                    ),
                                    "approved_for_rag": True,
                                    "rag_memory": True,
                                },
                            )

        return RagPersistenceResult(
            claims_saved=claims_saved,
            evidence_links_saved=links_saved,
            approved_evidences=len(approved_ids),
            pending_candidates=tuple(pending.values()),
        )

    def mark_indexed(self, evidence_id: str, faiss_doc_id: str) -> None:
        self._mark_index_attempt(evidence_id, faiss_doc_id=faiss_doc_id, error=None)

    def pending_candidates(self, limit: int = 100) -> tuple[RagEvidenceCandidate, ...]:
        """Recupera evidências aprovadas que ainda não chegaram ao FAISS."""
        if not self.is_configured:
            return ()

        import psycopg2
        import psycopg2.extras

        query = """
            SELECT id, trecho, fonte, dominio, url, data_publicacao,
                   tipo_fonte, fonte_confiavel, metadata_json
            FROM evidencias_rag
            WHERE aprovada_para_rag = TRUE
              AND faiss_indexada_em IS NULL
            ORDER BY primeira_observacao_em
            LIMIT %(limit)s;
        """
        with psycopg2.connect(self.database_url) as connection:
            with connection.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cursor:
                cursor.execute(query, {"limit": max(1, int(limit))})
                rows = cursor.fetchall()

        candidates = []
        for row in rows:
            metadata = row["metadata_json"] or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            candidates.append(
                RagEvidenceCandidate(
                    evidence_id=str(row["id"]),
                    text=row["trecho"],
                    source=row["fonte"] or row["dominio"] or "Evidência RAG",
                    url=row["url"] or "",
                    published_at=row["data_publicacao"],
                    metadata={
                        **metadata,
                        "domain": row["dominio"] or "",
                        "source_type": row["tipo_fonte"] or "external_document",
                        "trusted_source": bool(row["fonte_confiavel"]),
                        "approved_for_rag": True,
                        "rag_memory": True,
                    },
                )
            )
        return tuple(candidates)

    def mark_index_failed(self, evidence_id: str, error: str) -> None:
        self._mark_index_attempt(evidence_id, faiss_doc_id=None, error=error)

    def _mark_index_attempt(
        self,
        evidence_id: str,
        *,
        faiss_doc_id: str | None,
        error: str | None,
    ) -> None:
        if not self.is_configured:
            return

        import psycopg2

        query = """
            UPDATE evidencias_rag
            SET tentativas_indexacao = tentativas_indexacao + 1,
                faiss_doc_id = COALESCE(%(faiss_doc_id)s, faiss_doc_id),
                faiss_indexada_em = CASE
                    WHEN %(faiss_doc_id)s IS NOT NULL THEN NOW()
                    ELSE faiss_indexada_em
                END,
                ultimo_erro_indexacao = %(error)s
            WHERE id = %(evidence_id)s;
        """
        try:
            with psycopg2.connect(self.database_url) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        query,
                        {
                            "evidence_id": evidence_id,
                            "faiss_doc_id": faiss_doc_id,
                            "error": error[:2000] if error else None,
                        },
                    )
        except Exception as exc:
            logger.warning("[rag_memory] falha ao atualizar estado FAISS: %s", exc)
