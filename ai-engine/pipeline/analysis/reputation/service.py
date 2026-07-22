from __future__ import annotations

from datetime import datetime, timezone
import logging

from .collector import ReputationEvidenceCollector
from .config import DYNAMIC_ENABLED, METHOD_NAME
from .evaluator import ReputationEvaluator
from .identity import SourceIdentityResolver, domain_from_url
from .models import SourceReputation
from .repository import SourceReputationRepository

logger = logging.getLogger(__name__)


def _is_expired(value: SourceReputation) -> bool:
    if not value.expires_at:
        return value.status in {"insufficient_evidence", "failed", "not_evaluated"}
    try:
        expires = datetime.fromisoformat(value.expires_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= expires


class SourceReputationService:
    """Serviço único usado pelo pipeline e pelo script do Fake.Br."""

    def __init__(
        self,
        repository: SourceReputationRepository | None = None,
        identity_resolver: SourceIdentityResolver | None = None,
        collector: ReputationEvidenceCollector | None = None,
        evaluator: ReputationEvaluator | None = None,
    ) -> None:
        self.repository = repository or SourceReputationRepository()
        self.identity_resolver = identity_resolver or SourceIdentityResolver()
        self.collector = collector or ReputationEvidenceCollector()
        self.evaluator = evaluator or ReputationEvaluator()

    def get_or_evaluate(
        self,
        url_or_domain: str,
        *,
        trigger: str = "pipeline",
        force: bool = False,
        metadata: dict | None = None,
    ) -> SourceReputation:
        requested_domain = domain_from_url(url_or_domain)

        if requested_domain and not force:
            stored = self.repository.get_by_domain_or_alias(requested_domain)
            if stored is not None and not _is_expired(stored):
                stored.metadata = {**stored.metadata, "storage_hit": True, "storage_backend": self.repository.backend}
                return stored

        started_at = datetime.now(timezone.utc).isoformat()
        identity = self.identity_resolver.resolve(url_or_domain)

        if identity.canonical_domain and not force:
            stored = self.repository.get_by_domain_or_alias(identity.canonical_domain)
            if stored is not None and not _is_expired(stored):
                stored.metadata = {**stored.metadata, "storage_hit": True, "storage_backend": self.repository.backend}
                return stored

        if not identity.canonical_domain:
            result = SourceReputation(
                identity=identity,
                status="failed",
                note=None,
                score=0.5,
                classification="avaliação insuficiente",
                criteria={},
                origin=trigger,
                method=METHOD_NAME,
                needs_review=True,
                requires_external_evidence_weight=True,
                reason="Não foi possível identificar o domínio da fonte.",
                metadata={"storage_backend": self.repository.backend, **(metadata or {})},
            )
            return result

        if not DYNAMIC_ENABLED and trigger == "pipeline":
            return SourceReputation(
                identity=identity,
                status="not_evaluated",
                note=None,
                score=0.5,
                classification="fonte ainda não avaliada",
                criteria={},
                origin=trigger,
                method=METHOD_NAME,
                needs_review=True,
                requires_external_evidence_weight=True,
                reason="A avaliação dinâmica de fontes está desativada no .env.",
                metadata={"storage_backend": self.repository.backend, **(metadata or {})},
            )

        try:
            collection = self.collector.collect(identity)
            result = self.evaluator.evaluate(
                identity,
                collection,
                origin=trigger,
                metadata={"storage_hit": False, "storage_backend": self.repository.backend, **(metadata or {})},
            )
            # Tanto avaliações completas quanto insuficientes são gravadas. As
            # insuficientes expiram rapidamente e serão pesquisadas novamente.
            self.repository.save(result)
            self.repository.record_run({
                "requested_domain": identity.requested_domain,
                "canonical_domain": identity.canonical_domain,
                "trigger": trigger,
                "status": result.status,
                "providers": result.providers_attempted,
                "query_count": result.query_count,
                "evidence_count": result.evidence_count,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            return result
        except Exception as exc:
            logger.exception("[reputation] avaliação dinâmica falhou")
            result = SourceReputation(
                identity=identity,
                status="failed",
                note=None,
                score=0.5,
                classification="avaliação insuficiente",
                criteria={},
                origin=trigger,
                method=METHOD_NAME,
                needs_review=True,
                requires_external_evidence_weight=True,
                reason=f"A avaliação da fonte falhou: {type(exc).__name__}: {exc}",
                metadata={"storage_backend": self.repository.backend, **(metadata or {})},
            )
            self.repository.record_run({
                "requested_domain": identity.requested_domain,
                "canonical_domain": identity.canonical_domain,
                "trigger": trigger,
                "status": "failed",
                "providers": [],
                "query_count": 0,
                "evidence_count": 0,
                "error": str(exc),
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            return result
