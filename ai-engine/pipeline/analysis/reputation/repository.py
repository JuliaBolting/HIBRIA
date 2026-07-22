from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import tempfile
import threading
from typing import Any

from .config import DATABASE_URL, JSON_STORAGE_PATH
from .models import SourceReputation

logger = logging.getLogger(__name__)


class SourceReputationRepository:
    """
    Persistência da reputação.

    Quando PostgreSQL está configurado, cada fonte é salva em uma transação
    individual. Sem banco, usa JSON persistente e grava atomicamente após cada
    fonte, permitindo interromper e retomar o script de semeadura.
    """

    _json_lock = threading.Lock()

    def __init__(self, database_url: str | None = None, json_path: Path | None = None) -> None:
        self.database_url = DATABASE_URL if database_url is None else database_url
        self.json_path = json_path or JSON_STORAGE_PATH

    @property
    def backend(self) -> str:
        return "postgresql" if self.database_url else "json"

    def init_schema(self) -> dict[str, Any]:
        if not self.database_url:
            data = self._read_json()
            self._write_json(data)
            return {
                "initialized": True,
                "backend": "json",
                "path": str(self.json_path),
            }

        import psycopg2

        migrations_dir = Path(__file__).resolve().parents[4] / "database" / "migrations"
        migrations = sorted(migrations_dir.glob("*.sql"))
        if not migrations:
            raise FileNotFoundError(
                f"Nenhuma migração SQL foi encontrada em {migrations_dir}"
            )

        with psycopg2.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                for migration in migrations:
                    cursor.execute(migration.read_text(encoding="utf-8"))
        return {
            "initialized": True,
            "backend": "postgresql",
            "migrations": [item.name for item in migrations],
        }

    def get_by_domain_or_alias(self, domain: str) -> SourceReputation | None:
        domain = (domain or "").strip().lower()
        if not domain:
            return None
        if self.database_url:
            value = self._get_postgres(domain)
            if value is not None:
                return value
        return self._get_json(domain)

    def save(self, reputation: SourceReputation) -> None:
        if self.database_url:
            try:
                self._save_postgres(reputation)
                return
            except Exception as exc:
                logger.warning(
                    "[reputation_repository] gravação no PostgreSQL falhou; usando JSON: %s",
                    exc,
                )
        self._save_json(reputation)

    def record_run(self, payload: dict[str, Any]) -> None:
        if self.database_url:
            try:
                self._record_run_postgres(payload)
                return
            except Exception as exc:
                logger.warning(
                    "[reputation_repository] histórico no PostgreSQL falhou; usando JSON: %s",
                    exc,
                )
        with self._json_lock:
            data = self._read_json()
            runs = data.setdefault("runs", [])
            runs.append(payload)
            # Histórico local limitado para não crescer indefinidamente.
            data["runs"] = runs[-2000:]
            self._write_json(data)

    # ------------------------------------------------------------------
    # JSON persistente
    # ------------------------------------------------------------------

    def _empty_json(self) -> dict[str, Any]:
        return {"version": 1, "sources": {}, "aliases": {}, "runs": []}

    def _read_json(self) -> dict[str, Any]:
        if not self.json_path.exists():
            return self._empty_json()
        try:
            data = json.loads(self.json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return self._empty_json()
        data.setdefault("version", 1)
        data.setdefault("sources", {})
        data.setdefault("aliases", {})
        data.setdefault("runs", [])
        return data

    def _write_json(self, data: dict[str, Any]) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=self.json_path.parent,
            suffix=".tmp",
        ) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        temp_path.replace(self.json_path)

    def _get_json(self, domain: str) -> SourceReputation | None:
        data = self._read_json()
        canonical = data.get("aliases", {}).get(domain, domain)
        payload = data.get("sources", {}).get(canonical)
        return SourceReputation.from_dict(payload) if payload else None

    def _save_json(self, reputation: SourceReputation) -> None:
        with self._json_lock:
            data = self._read_json()
            domain = reputation.identity.canonical_domain
            data["sources"][domain] = reputation.to_dict()
            for alias in {
                reputation.identity.requested_domain,
                *reputation.identity.aliases,
            }:
                alias = (alias or "").strip().lower()
                if alias and alias != domain:
                    data["aliases"][alias] = domain
            self._write_json(data)

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------

    def _get_postgres(self, domain: str) -> SourceReputation | None:
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            return None

        query = """
            SELECT f.payload_json
            FROM fontes_reputacao f
            LEFT JOIN aliases_fontes_reputacao a ON a.fonte_id = f.id AND a.ativo = TRUE
            WHERE LOWER(f.dominio_canonico) = LOWER(%s)
               OR LOWER(a.dominio_alias) = LOWER(%s)
            ORDER BY f.data_ultima_verificacao DESC NULLS LAST
            LIMIT 1;
        """
        try:
            with psycopg2.connect(self.database_url) as connection:
                with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(query, (domain, domain))
                    row = cursor.fetchone()
                    if not row:
                        return None
                    payload = row["payload_json"]
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    return SourceReputation.from_dict(payload)
        except Exception as exc:
            logger.warning("[reputation_repository] consulta ao PostgreSQL falhou: %s", exc)
            return None

    def _save_postgres(self, reputation: SourceReputation) -> None:
        import psycopg2
        import psycopg2.extras

        payload = reputation.to_dict()
        upsert = """
            INSERT INTO fontes_reputacao (
                dominio_canonico, nome_fonte, status_avaliacao, nota_total,
                score_reputacao, classificacao, origem, metodo_avaliacao,
                precisa_revisao, data_avaliacao, data_ultima_verificacao,
                proxima_reavaliacao, payload_json
            ) VALUES (
                %(domain)s, %(name)s, %(status)s, %(note)s,
                %(score)s, %(classification)s, %(origin)s, %(method)s,
                %(needs_review)s, %(evaluated_at)s, NOW(), %(expires_at)s,
                %(payload)s
            )
            ON CONFLICT (dominio_canonico) DO UPDATE SET
                nome_fonte = EXCLUDED.nome_fonte,
                status_avaliacao = EXCLUDED.status_avaliacao,
                nota_total = EXCLUDED.nota_total,
                score_reputacao = EXCLUDED.score_reputacao,
                classificacao = EXCLUDED.classificacao,
                origem = EXCLUDED.origem,
                metodo_avaliacao = EXCLUDED.metodo_avaliacao,
                precisa_revisao = EXCLUDED.precisa_revisao,
                data_avaliacao = EXCLUDED.data_avaliacao,
                data_ultima_verificacao = NOW(),
                proxima_reavaliacao = EXCLUDED.proxima_reavaliacao,
                payload_json = EXCLUDED.payload_json
            RETURNING id;
        """
        params = {
            "domain": reputation.identity.canonical_domain,
            "name": reputation.identity.source_name,
            "status": reputation.status,
            "note": reputation.note,
            "score": reputation.score,
            "classification": reputation.classification,
            "origin": reputation.origin,
            "method": reputation.method,
            "needs_review": reputation.needs_review,
            "evaluated_at": reputation.evaluated_at,
            "expires_at": reputation.expires_at,
            "payload": psycopg2.extras.Json(payload),
        }

        # Uma fonte por transação: se a execução for interrompida, as fontes
        # anteriores permanecem gravadas e o script pode ser retomado.
        with psycopg2.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(upsert, params)
                source_id = cursor.fetchone()[0]

                cursor.execute("DELETE FROM aliases_fontes_reputacao WHERE fonte_id = %s", (source_id,))
                for alias in {
                    reputation.identity.requested_domain,
                    *reputation.identity.aliases,
                }:
                    alias = (alias or "").strip().lower()
                    if not alias or alias == reputation.identity.canonical_domain:
                        continue
                    cursor.execute(
                        """
                        INSERT INTO aliases_fontes_reputacao (dominio_alias, fonte_id, tipo_alias, ativo)
                        VALUES (%s, %s, 'automatic_redirect_or_canonical', TRUE)
                        ON CONFLICT (dominio_alias) DO UPDATE SET fonte_id = EXCLUDED.fonte_id, ativo = TRUE;
                        """,
                        (alias, source_id),
                    )

                cursor.execute("DELETE FROM criterios_reputacao_fonte WHERE fonte_id = %s", (source_id,))
                cursor.execute("DELETE FROM evidencias_reputacao_fonte WHERE fonte_id = %s", (source_id,))

                for key, criterion in reputation.criteria.items():
                    cursor.execute(
                        """
                        INSERT INTO criterios_reputacao_fonte (
                            fonte_id, criterio, peso_maximo, pontos_obtidos,
                            status, justificativa
                        ) VALUES (%s, %s, %s, %s, %s, %s);
                        """,
                        (
                            source_id,
                            key,
                            criterion.max_points,
                            criterion.points,
                            criterion.status,
                            criterion.justification,
                        ),
                    )
                    for evidence in criterion.evidences:
                        cursor.execute(
                            """
                            INSERT INTO evidencias_reputacao_fonte (
                                fonte_id, criterio, provedor, tipo_evidencia,
                                titulo, url, trecho, metadata_json
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                            """,
                            (
                                source_id,
                                key,
                                evidence.provider,
                                evidence.evidence_type,
                                evidence.title,
                                evidence.url,
                                evidence.snippet,
                                psycopg2.extras.Json(evidence.metadata),
                            ),
                        )

    def _record_run_postgres(self, payload: dict[str, Any]) -> None:
        import psycopg2
        import psycopg2.extras

        query = """
            INSERT INTO execucoes_avaliacao_reputacao (
                dominio_solicitado, dominio_canonico, gatilho, status,
                provedores_json, quantidade_consultas, quantidade_evidencias,
                mensagem_erro, iniciada_em, finalizada_em
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        with psycopg2.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (
                    payload.get("requested_domain", ""),
                    payload.get("canonical_domain"),
                    payload.get("trigger", "unknown"),
                    payload.get("status", "unknown"),
                    psycopg2.extras.Json(payload.get("providers", [])),
                    payload.get("query_count", 0),
                    payload.get("evidence_count", 0),
                    payload.get("error"),
                    payload.get("started_at"),
                    payload.get("finished_at"),
                ))
