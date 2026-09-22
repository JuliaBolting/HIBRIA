from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import logging
import os
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedAnalysis:
    analysis_id: str
    data: dict[str, Any]
    analyzed_at: datetime | None
    request_count: int


class AnalysisRepository:
    """Persistência e recuperação do cache de análises no PostgreSQL."""

    TRACKING_PARAMETERS = {
        "fbclid",
        "gclid",
        "dclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
    }

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
    def pipeline_version() -> str:
        return os.getenv("HIBRIA_PIPELINE_VERSION", "1.0.0").strip() or "1.0.0"

    @classmethod
    def normalize_url(cls, value: str) -> str:
        parts = urlsplit((value or "").strip())
        scheme = parts.scheme.lower() or "https"
        hostname = (parts.hostname or "").lower()

        if parts.port and not (
            (scheme == "http" and parts.port == 80)
            or (scheme == "https" and parts.port == 443)
        ):
            netloc = f"{hostname}:{parts.port}"
        else:
            netloc = hostname

        path = re.sub(r"/{2,}", "/", parts.path or "/")
        if path != "/":
            path = path.rstrip("/")

        query_items = [
            (key, item)
            for key, item in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_")
            and key.lower() not in cls.TRACKING_PARAMETERS
        ]
        query = urlencode(sorted(query_items))

        return urlunsplit((scheme, netloc, path, query, ""))

    @staticmethod
    def normalize_content(value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip()

    @staticmethod
    def sha256(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def cache_identity(cls, url: str, content: str) -> dict[str, str]:
        normalized_url = cls.normalize_url(url)
        normalized_content = cls.normalize_content(content)
        return {
            "normalized_url": normalized_url,
            "url_hash": cls.sha256(normalized_url),
            "content_hash": cls.sha256(normalized_content),
            "pipeline_version": cls.pipeline_version(),
        }

    def get(self, url: str, content: str) -> CachedAnalysis | None:
        if not self.is_configured:
            return None

        identity = self.cache_identity(url, content)
        query = """
            UPDATE analises
            SET ultima_consulta_em = NOW(),
                quantidade_consultas = quantidade_consultas + 1
            WHERE hash_url = %(url_hash)s
              AND hash_conteudo = %(content_hash)s
              AND versao_pipeline = %(pipeline_version)s
            RETURNING id, resultado_json, data_analise, quantidade_consultas;
        """

        try:
            import psycopg2
            import psycopg2.extras

            with psycopg2.connect(self.database_url) as connection:
                with connection.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                ) as cursor:
                    cursor.execute(query, identity)
                    row = cursor.fetchone()

            if not row:
                return None

            data = row["resultado_json"]
            if isinstance(data, str):
                data = json.loads(data)

            return CachedAnalysis(
                analysis_id=str(row["id"]),
                data=deepcopy(data),
                analyzed_at=row["data_analise"],
                request_count=int(row["quantidade_consultas"]),
            )
        except Exception as exc:
            logger.warning(
                "[analysis_repository] consulta ao cache falhou: %s",
                exc,
            )
            return None

    def save(
        self,
        *,
        url: str,
        title: str,
        content: str,
        score: float | None,
        classification: str | None,
        explanation: str | None,
        processing_time: float | None,
        data: dict[str, Any],
    ) -> str | None:
        if not self.is_configured:
            return None

        identity = self.cache_identity(url, content)
        query = """
            INSERT INTO analises (
                url_original,
                url_normalizada,
                hash_url,
                titulo,
                conteudo,
                hash_conteudo,
                versao_pipeline,
                score_final,
                classificacao_final,
                explicacao,
                tempo_processamento_segundos,
                resultado_json
            ) VALUES (
                %(url)s,
                %(normalized_url)s,
                %(url_hash)s,
                %(title)s,
                %(content)s,
                %(content_hash)s,
                %(pipeline_version)s,
                %(score)s,
                %(classification)s,
                %(explanation)s,
                %(processing_time)s,
                %(data)s
            )
            ON CONFLICT (hash_url, hash_conteudo, versao_pipeline)
            DO UPDATE SET
                ultima_consulta_em = NOW(),
                quantidade_consultas = analises.quantidade_consultas + 1
            RETURNING id;
        """

        def json_dumps(payload: Any) -> str:
            return json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            )

        try:
            import psycopg2
            import psycopg2.extras

            params = {
                **identity,
                "url": url,
                "title": title,
                "content": content,
                "score": score,
                "classification": classification,
                "explanation": explanation,
                "processing_time": processing_time,
                "data": psycopg2.extras.Json(data, dumps=json_dumps),
            }

            with psycopg2.connect(self.database_url) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, params)
                    analysis_id = cursor.fetchone()[0]

            return str(analysis_id)
        except Exception as exc:
            logger.warning(
                "[analysis_repository] gravação da análise falhou: %s",
                exc,
            )
            return None

    def update_result(self, analysis_id: str, data: dict[str, Any]) -> bool:
        """Atualiza o JSON após a sincronização da memória RAG."""
        if not self.is_configured or not analysis_id:
            return False

        query = """
            UPDATE analises
            SET resultado_json = %(data)s
            WHERE id = %(analysis_id)s;
        """

        try:
            import psycopg2
            import psycopg2.extras

            payload = psycopg2.extras.Json(
                data,
                dumps=lambda value: json.dumps(
                    value,
                    ensure_ascii=False,
                    default=str,
                ),
            )
            with psycopg2.connect(self.database_url) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        query,
                        {"analysis_id": analysis_id, "data": payload},
                    )
                    return cursor.rowcount == 1
        except Exception as exc:
            logger.warning(
                "[analysis_repository] atualização do resultado falhou: %s",
                exc,
            )
            return False
