from __future__ import annotations

import os
from pipeline.cancellation import checkpoint
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
        if os.getenv("HIBRIA_ALLOW_STATIC_TRUST", "false").lower() != "true":
            self.static_domains = set()
        self._cache: dict[str, float | None] = {}
        try:
            self._evaluations_left = max(0, min(10, int(os.getenv("HIBRIA_REFERENCE_REPUTATION_BUDGET", "2"))))
        except ValueError:
            self._evaluations_left = 2

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(os.getenv(name, str(default)))))
        except ValueError:
            return default

    def reputation(self, value: str) -> float | None:
        checkpoint()
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
                        SELECT fr.score_reputacao, fr.status_avaliacao
                        FROM fontes_reputacao fr

                        WHERE fr.dominio_canonico = %(domain)s
                          AND (fr.proxima_reavaliacao IS NULL
                               OR fr.proxima_reavaliacao > NOW())
                        ORDER BY fr.data_ultima_verificacao DESC
                        LIMIT 1;
                        """,
                        {"domain": domain},
                    )
                    row = cursor.fetchone()
            score = float(row[0]) if row and row[1] == "evaluated" else None
            if row is None and self._evaluations_left > 0:
                from pipeline.analysis.reputation.service import SourceReputationService
                from pipeline.analysis.reputation.repository import SourceReputationRepository
                from pipeline.analysis.reputation.config import DYNAMIC_ENABLED
                if DYNAMIC_ENABLED:
                    self._evaluations_left -= 1
                    checkpoint()
                    service = SourceReputationService(repository=SourceReputationRepository(database_url=self.database_url))
                    assessed = service.get_or_evaluate(domain, trigger="evidence")
                    if assessed.status == "evaluated" and assessed.identity.canonical_domain == domain:
                        score = float(assessed.score)
        except Exception:
            score = None

        self._cache[domain] = score
        return score

    def resolve(self, value: str) -> tuple[bool, float | None]:
        score = self.reputation(value)
        return score is not None and score >= self.minimum, score
