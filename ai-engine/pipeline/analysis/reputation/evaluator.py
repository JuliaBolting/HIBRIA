from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta, timezone

from .collector import CollectionResult
from .config import (
    CACHE_TTL_DAYS,
    INSUFFICIENT_TTL_DAYS,
    METHOD_NAME,
    MIN_CONFIRMED_CRITERIA,
    MIN_EVIDENCE_URLS,
)
from .criteria import CRITERIA, ReputationCriterion
from .interpreter import GeminiEvidenceInterpreter
from .models import CriterionEvaluation, SourceIdentity, SourceReputation


STRONG_NEGATIVE_HISTORY_TERMS = (
    "recorrência de desinformação",
    "recorrencia de desinformacao",
    "publicou notícias falsas",
    "publicou noticias falsas",
    "rede de desinformação",
    "rede de desinformacao",
    "reincidente em desinformação",
    "reincidente em desinformacao",
    "veículo de desinformação",
    "veiculo de desinformacao",
)


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text.lower()).strip()


def _contains_terms(text: str, terms: tuple[str, ...]) -> bool:
    normalized = _plain(text)
    return any(_plain(term) in normalized for term in terms)


def _classification(note: int | None) -> str:
    if note is None:
        return "avaliação insuficiente"
    if note >= 90:
        return "confiável"
    if note >= 80:
        return "confiável com ressalvas"
    if note >= 60:
        return "parcialmente verificada"
    return "baixa reputação"


class ReputationEvaluator:
    """Aplica os pesos do TCC de forma determinística e auditável."""

    def __init__(self, interpreter: GeminiEvidenceInterpreter | None = None) -> None:
        self.interpreter = interpreter or GeminiEvidenceInterpreter()

    def evaluate(
        self,
        identity: SourceIdentity,
        collection: CollectionResult,
        *,
        origin: str,
        metadata: dict | None = None,
    ) -> SourceReputation:
        ai_review = self.interpreter.interpret(identity, collection.evidence_by_criterion)
        criteria_output: dict[str, CriterionEvaluation] = {}

        for criterion in CRITERIA:
            evidence = collection.evidence_by_criterion.get(criterion.key, [])
            ai_item = ai_review.get(criterion.key, {}) if isinstance(ai_review, dict) else {}
            criteria_output[criterion.key] = self._score_criterion(
                criterion,
                evidence,
                identity=identity,
                ai_item=ai_item if isinstance(ai_item, dict) else {},
                search_succeeded=bool(collection.providers_succeeded),
            )

        unique_urls = {
            item.url.strip().lower().rstrip("/")
            for criterion in criteria_output.values()
            for item in criterion.evidences
            if item.url
        }
        confirmed = sum(
            1 for item in criteria_output.values()
            if item.status in {"found", "partial", "clear_history", "negative_history"}
            and item.points is not None
        )

        sufficient = (
            len(unique_urls) >= MIN_EVIDENCE_URLS
            and confirmed >= MIN_CONFIRMED_CRITERIA
            and bool(collection.providers_succeeded)
        )

        now = datetime.now(timezone.utc)
        if not sufficient:
            return SourceReputation(
                identity=identity,
                status="insufficient_evidence",
                note=None,
                score=0.5,
                classification=_classification(None),
                criteria=criteria_output,
                origin=origin,
                method=METHOD_NAME,
                needs_review=True,
                requires_external_evidence_weight=True,
                reason=(
                    "A busca não recuperou evidências públicas suficientes para calcular "
                    "uma nota segura. A ausência de evidência não foi convertida em baixa reputação."
                ),
                providers_attempted=collection.providers_attempted,
                providers_succeeded=collection.providers_succeeded,
                provider_failures=collection.provider_failures,
                query_count=collection.query_count,
                evidence_count=len(unique_urls),
                expires_at=(now + timedelta(days=INSUFFICIENT_TTL_DAYS)).isoformat(),
                metadata={"confirmed_criteria": confirmed, "gemini_review_used": bool(ai_review), **(metadata or {})},
            )

        total = sum(item.points or 0 for item in criteria_output.values())
        total = max(0, min(100, int(total)))
        partial_or_inconclusive = any(
            item.status in {"partial", "inconclusive", "not_found"}
            for item in criteria_output.values()
        )

        return SourceReputation(
            identity=identity,
            status="evaluated",
            note=total,
            score=round(total / 100, 4),
            classification=_classification(total),
            criteria=criteria_output,
            origin=origin,
            method=METHOD_NAME,
            needs_review=partial_or_inconclusive or total < 80,
            requires_external_evidence_weight=total < 60,
            reason=(
                "A nota foi calculada com os pesos definidos no TCC, a partir de "
                "evidências públicas recuperadas para cada critério."
            ),
            providers_attempted=collection.providers_attempted,
            providers_succeeded=collection.providers_succeeded,
            provider_failures=collection.provider_failures,
            query_count=collection.query_count,
            evidence_count=len(unique_urls),
            expires_at=(now + timedelta(days=CACHE_TTL_DAYS)).isoformat(),
            metadata={"confirmed_criteria": confirmed, "gemini_review_used": bool(ai_review), **(metadata or {})},
        )

    def _score_criterion(
        self,
        criterion: ReputationCriterion,
        evidence,
        *,
        identity: SourceIdentity,
        ai_item: dict,
        search_succeeded: bool,
    ) -> CriterionEvaluation:
        if criterion.key == "adequacao_tecnica_sistema":
            if identity.homepage_accessible:
                return CriterionEvaluation(
                    key=criterion.key,
                    label=criterion.label,
                    max_points=criterion.weight,
                    points=criterion.weight,
                    status="found",
                    justification="O domínio respondeu e apresentou conteúdo acessível ao sistema.",
                    evidences=evidence[:5],
                )
            return CriterionEvaluation(
                key=criterion.key,
                label=criterion.label,
                max_points=criterion.weight,
                points=0,
                status="not_found",
                justification="Não foi possível confirmar a acessibilidade técnica do domínio.",
                evidences=evidence[:5],
            )

        if criterion.key == "historico_publico_desinformacao":
            return self._score_history(criterion, evidence, search_succeeded=search_succeeded)

        matching = [
            item for item in evidence
            if _contains_terms(f"{item.title} {item.snippet}", criterion.positive_terms)
        ]
        direct_matching = [item for item in matching if item.evidence_type == "direct_page" and item.is_same_domain]

        ai_status = str(ai_item.get("status", "")).lower()
        if direct_matching or len(matching) >= 2 or ai_status == "found":
            return CriterionEvaluation(
                key=criterion.key,
                label=criterion.label,
                max_points=criterion.weight,
                points=criterion.weight,
                status="found",
                justification=f"Foram encontradas evidências públicas compatíveis com {criterion.label.lower()}.",
                evidences=(direct_matching or matching or evidence)[:5],
            )

        if matching or ai_status == "partial":
            points = max(1, int(round(criterion.weight * 0.6)))
            return CriterionEvaluation(
                key=criterion.key,
                label=criterion.label,
                max_points=criterion.weight,
                points=points,
                status="partial",
                justification=f"Foi encontrada evidência parcial de {criterion.label.lower()}.",
                evidences=(matching or evidence)[:5],
            )

        return CriterionEvaluation(
            key=criterion.key,
            label=criterion.label,
            max_points=criterion.weight,
            points=0,
            status="not_found" if search_succeeded else "inconclusive",
            justification=(
                f"A busca não encontrou evidência suficiente para confirmar {criterion.label.lower()}."
                if search_succeeded
                else f"O critério {criterion.label.lower()} não pôde ser avaliado porque nenhum provedor respondeu."
            ),
            evidences=evidence[:5],
        )

    def _score_history(self, criterion: ReputationCriterion, evidence, *, search_succeeded: bool) -> CriterionEvaluation:
        joined = " ".join(f"{item.title} {item.snippet}" for item in evidence)
        strong_negative = _contains_terms(joined, STRONG_NEGATIVE_HISTORY_TERMS)

        if strong_negative:
            return CriterionEvaluation(
                key=criterion.key,
                label=criterion.label,
                max_points=criterion.weight,
                points=0,
                status="negative_history",
                justification="Foram encontrados indícios públicos de recorrência relevante de desinformação.",
                evidences=evidence[:5],
            )

        if search_succeeded:
            return CriterionEvaluation(
                key=criterion.key,
                label=criterion.label,
                max_points=criterion.weight,
                points=15,
                status="clear_history",
                justification=(
                    "Nas buscas executadas não foi encontrada recorrência significativa de desinformação. "
                    "A pontuação permanece conservadora porque ausência de resultado não é prova absoluta."
                ),
                evidences=evidence[:5],
            )

        return CriterionEvaluation(
            key=criterion.key,
            label=criterion.label,
            max_points=criterion.weight,
            points=None,
            status="inconclusive",
            justification="Não houve busca suficiente para avaliar o histórico público de desinformação.",
            evidences=evidence[:5],
        )
