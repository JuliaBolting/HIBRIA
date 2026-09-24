from __future__ import annotations

import os
from urllib.parse import urlparse


def _domain(value: str) -> str:
    parsed = urlparse(value if "://" in value else f"https://{value}")
    domain = (parsed.hostname or "").lower()
    return domain.removeprefix("www.")


class SourceTrustResolver:
    """Resolve confiança de fontes usando a reputação dinâmica do PostgreSQL.

    `HIBRIA_TRUSTED_DOMAINS` permanece apenas como compatibilidade opcional.
    Sem reputação avaliada, a evidência continua visível, mas não é marcada como
    confiável nem entra como prova factual principal no agregador.
    """

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = (
            database_url
            if database_url is not None
            else os.getenv("HIBRIA_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or ""
        )
        self.minimum = self._env_float(
            "HIBRIA_EVIDENCE_MIN_SOURCE_REPUTATION", 0.80
        )
        self.static_domains = {
            _domain(item.strip())
            for item in os.getenv("HIBRIA_TRUSTED_DOMAINS", "").split(",")
            if item.strip()
        }
        self._cache: dict[str, float | None] = {}

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(os.getenv(name, str(default)))))
        except ValueError:
            return default

    def reputation(self, value: str) -> float | None:
        domain = _domain(value)
        if not domain:
            return None
        if domain in self._cache:
            return self._cache[domain]
        if domain in self.static_domains:
            self._cache[domain] = 1.0
            return 1.0
        if not self.database_url:
            self._cache[domain] = None
            return None

        try:
            import psycopg2

            with psycopg2.connect(self.database_url) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT fr.score_reputacao
                        FROM fontes_reputacao fr
                        LEFT JOIN aliases_fontes_reputacao afr
                          ON afr.fonte_id = fr.id AND afr.ativo = TRUE
                        WHERE (fr.dominio_canonico = %(domain)s
                               OR afr.dominio_alias = %(domain)s)
                          AND fr.status_avaliacao = 'evaluated'
                          AND (fr.proxima_reavaliacao IS NULL
                               OR fr.proxima_reavaliacao > NOW())
                        ORDER BY fr.data_ultima_verificacao DESC
                        LIMIT 1;
                        """,
                        {"domain": domain},
                    )
                    row = cursor.fetchone()
            score = float(row[0]) if row else None
        except Exception:
            score = None

        self._cache[domain] = score
        return score

    def resolve(self, value: str) -> tuple[bool, float | None]:
        score = self.reputation(value)
        return score is not None and score >= self.minimum, score
