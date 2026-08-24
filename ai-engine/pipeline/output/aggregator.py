# =============================================================================
# aggregator.py
# Combina os resultados parciais do pipeline em um score final de confiabilidade.
#
# Componentes considerados nesta versão:
#   - Evidências recuperadas pelo RAG / FAISS / web
#   - Similaridade entre claims e evidências
#   - Stance entre claim e evidência
#   - Reputação do domínio da notícia analisada
#   - Classificação textual auxiliar com BERTimbau
# =============================================================================

from __future__ import annotations

import os
from typing import Any


def env_float(name: str, default: float, aliases: list[str] | None = None) -> float:
    """
    Lê float do .env com fallback seguro.

    `aliases` mantém compatibilidade com nomes antigos de variáveis.
    """
    names = [name] + (aliases or [])

    for item in names:
        value = os.getenv(item)

        if value is None or value.strip() == "":
            continue

        try:
            return float(value)
        except ValueError:
            return default

    return default


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


class Aggregator:
    """
    Agrega os scores parciais da análise em um índice final.

    O resultado não representa uma verdade absoluta, mas um índice de
    confiabilidade calculado a partir dos critérios disponíveis no sistema.
    """

    MIN_VALID_EVIDENCE_SCORE: float = env_float(
        "HIBRIA_AGGREGATOR_MIN_VALID_EVIDENCE_SCORE",
        0.55,
        aliases=["HIBRIA_MIN_VALID_EVIDENCE_SCORE"],
    )
    MIN_COVERAGE_FOR_RELIABLE: float = env_float(
        "HIBRIA_AGGREGATOR_MIN_COVERAGE_FOR_RELIABLE",
        0.35,
        aliases=["HIBRIA_MIN_COVERAGE_FOR_RELIABLE"],
    )
    MIN_COVERAGE_FOR_PARTIAL: float = env_float(
        "HIBRIA_AGGREGATOR_MIN_COVERAGE_FOR_PARTIAL",
        0.20,
        aliases=["HIBRIA_MIN_COVERAGE_FOR_PARTIAL"],
    )
    MIN_SUPPORT_RATE_FOR_RELIABLE: float = env_float(
        "HIBRIA_AGGREGATOR_MIN_SUPPORT_RATE_FOR_RELIABLE",
        0.60,
        aliases=["HIBRIA_MIN_SUPPORT_RATE_FOR_RELIABLE"],
    )
    MIN_CONTRADICTION_RATE_FOR_UNRELIABLE: float = env_float(
        "HIBRIA_AGGREGATOR_MIN_CONTRADICTION_RATE_FOR_UNRELIABLE",
        0.40,
        aliases=["HIBRIA_MIN_CONTRADICTION_RATE_FOR_UNRELIABLE"],
    )

    # Wikipedia é mantida como contexto, mas não conta como evidência factual principal.
    EXCLUDED_EVIDENCE_LAYERS = {"wikipedia"}
    EXCLUDED_SOURCE_TYPES = {"encyclopedia"}

    WEIGHTS = {
        "evidence": env_float("HIBRIA_AGGREGATOR_EVIDENCE_WEIGHT", 0.60),
        "reputation": env_float("HIBRIA_AGGREGATOR_REPUTATION_WEIGHT", 0.25),
        "bertimbau": env_float("HIBRIA_AGGREGATOR_BERTIMBAU_WEIGHT", 0.15),
    }

    @classmethod
    def _is_valid_factual_evidence(cls, item: Any) -> bool:
        """
        Define se uma evidência pode entrar no cálculo factual.

        A evidência precisa:
        - existir;
        - passar do limiar mínimo de similaridade;
        - não ser Wikipedia/contexto enciclopédico;
        - ser marcada como suficiente pelo similarity.py, quando esse campo existir.
        """
        if not getattr(item, "has_evidence", False):
            return False

        if getattr(item, "score", None) is None:
            return False

        if item.score < cls.MIN_VALID_EVIDENCE_SCORE:
            return False

        if (
            hasattr(item, "has_sufficient_evidence")
            and not item.has_sufficient_evidence
        ):
            return False

        top = getattr(item, "top_evidence", None)
        if not top:
            return False

        layer = getattr(top, "evidence_layer", "")
        source_type = getattr(top, "source_type", "")
        trusted_source = bool(getattr(top, "trusted_source", False))

        if layer in cls.EXCLUDED_EVIDENCE_LAYERS:
            return False

        if source_type in cls.EXCLUDED_SOURCE_TYPES:
            return False

        if hasattr(top, "is_sufficient") and not top.is_sufficient:
            return False

        if source_type == "ai_retrieved_web" and not trusted_source:
            return False

        return True

    @classmethod
    def _calculate_evidence_score(cls, result: Any) -> float:
        """
        Calcula o score médio das claims que possuem evidência válida.
        """
        if not result.similarity_scores:
            return 0.0

        scores = [
            item.score
            for item in result.similarity_scores
            if cls._is_valid_factual_evidence(item)
        ]

        if not scores:
            return 0.0

        return round(sum(scores) / len(scores), 4)

    @classmethod
    def _calculate_coverage(cls, result: Any) -> float:
        """
        Mede a cobertura de evidências válidas.
        """
        if not result.similarity_scores:
            return 0.0

        total = len(result.similarity_scores)

        if total == 0:
            return 0.0

        with_valid_evidence = sum(
            1
            for item in result.similarity_scores
            if cls._is_valid_factual_evidence(item)
        )

        return round(with_valid_evidence / total, 4)

    @staticmethod
    def _calculate_reputation_score(result: Any) -> float | None:
        """
        Usa reputação somente quando a fonte recebeu uma avaliação completa.

        Resultado insuficiente, falho ou ainda não avaliado não é convertido em
        reputação baixa. Nesses casos o componente é retirado do peso final.
        """
        reputation = getattr(result, "reputation", None)
        if not reputation or not isinstance(reputation, dict):
            return None

        if reputation.get("status") != "evaluated" or reputation.get("note") is None:
            return None

        try:
            return clamp(float(reputation.get("score")))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _calculate_bertimbau_score(result: Any) -> float | None:
        """
        Recupera o score do BERTimbau quando a etapa foi executada com sucesso.

        O score do bertimbau_classifier.py representa probabilidade de notícia
        verdadeira. Quando o modelo falha, fica desativado ou não há texto, o
        componente não entra no peso final.
        """
        classification = getattr(result, "classification", None)

        if not classification or not isinstance(classification, dict):
            return None

        if classification.get("status") != "ok":
            return None

        score = classification.get("score")

        if score is None:
            return None

        try:
            return clamp(float(score))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _calculate_stance_stats(result: Any) -> dict:
        """
        Calcula proporções de apoio, contradição, neutralidade e insuficiência.

        Ausência de evidência não deve ser tratada como contradição.
        A notícia só deve ser marcada como não confiável quando houver
        contradição forte entre claims e evidências recuperadas.
        """
        stats = {
            "support": 0,
            "contradict": 0,
            "neutral": 0,
            "insufficient": 0,
            "total": 0,
            "support_rate": 0.0,
            "contradiction_rate": 0.0,
        }

        for item in result.stance_results or []:
            if hasattr(item, "stance"):
                stance = item.stance
            elif isinstance(item, dict):
                stance = item.get("stance")
            else:
                stance = None

            if stance not in {"support", "contradict", "neutral", "insufficient"}:
                continue

            stats[stance] += 1
            stats["total"] += 1

        # Aqui ignoramos "insufficient", porque ausência de evidência não é apoio nem contradição.
        valid_total = stats["support"] + stats["contradict"] + stats["neutral"]

        if valid_total > 0:
            stats["support_rate"] = round(stats["support"] / valid_total, 4)
            stats["contradiction_rate"] = round(stats["contradict"] / valid_total, 4)

        return stats

    @classmethod
    def _label_from_evidence(
        cls,
        final_score: float,
        evidence_score: float,
        coverage_score: float,
        reputation_score: float | None,
        stance_stats: dict,
        bertimbau_score: float | None,
    ) -> str:
        """
        Define o rótulo final sem confundir ausência de evidência com falsidade.

        Regras principais:
        - Contradição forte -> não confiável.
        - Sem evidência suficiente -> evidência insuficiente ou não verificado.
        - Boa evidência + boa cobertura -> confiável.
        - Evidência moderada -> parcialmente confiável.

        O BERTimbau entra como componente auxiliar do score, mas não transforma
        ausência de evidência em falsidade factual.
        """
        contradiction_rate = stance_stats.get("contradiction_rate", 0.0)
        support_rate = stance_stats.get("support_rate", 0.0)

        if contradiction_rate >= cls.MIN_CONTRADICTION_RATE_FOR_UNRELIABLE:
            return "não confiável"

        if coverage_score == 0 or evidence_score == 0:
            if reputation_score is not None and reputation_score >= 0.80:
                return "evidência insuficiente"
            if bertimbau_score is not None and final_score >= 60:
                return "não verificado"
            return "não verificado"

        if coverage_score < cls.MIN_COVERAGE_FOR_PARTIAL:
            return "evidência insuficiente"

        if (
            final_score >= 75
            and coverage_score >= cls.MIN_COVERAGE_FOR_RELIABLE
            and support_rate >= cls.MIN_SUPPORT_RATE_FOR_RELIABLE
        ):
            return "confiável"

        if final_score >= 50 or evidence_score >= 0.55:
            return "parcialmente confiável"

        return "evidência insuficiente"

    @classmethod
    def _normalize_component_weights(
        cls,
        bertimbau_score: float | None,
        reputation_score: float | None,
    ) -> dict[str, float]:
        weights = {
            "evidence": max(0.0, cls.WEIGHTS.get("evidence", 0.60)),
            "reputation": (
                max(0.0, cls.WEIGHTS.get("reputation", 0.25))
                if reputation_score is not None
                else 0.0
            ),
            "bertimbau": (
                max(0.0, cls.WEIGHTS.get("bertimbau", 0.15))
                if bertimbau_score is not None
                else 0.0
            ),
        }

        weight_sum = sum(weights.values())
        if weight_sum <= 0:
            weights = {"evidence": 1.0, "reputation": 0.0, "bertimbau": 0.0}
            weight_sum = 1.0

        return {key: round(value / weight_sum, 4) for key, value in weights.items()}

    @classmethod
    def aggregate(cls, result: Any) -> dict:
        evidence_score = cls._calculate_evidence_score(result)
        coverage_score = cls._calculate_coverage(result)
        reputation_score = cls._calculate_reputation_score(result)
        bertimbau_score = cls._calculate_bertimbau_score(result)
        stance_stats = cls._calculate_stance_stats(result)

        # Cobertura baixa reduz o componente de evidência, mas não transforma
        # automaticamente a notícia em "não confiável".
        evidence_component = evidence_score * coverage_score

        weights = cls._normalize_component_weights(bertimbau_score, reputation_score)

        final_score_0_1 = (
            evidence_component * weights["evidence"]
            + (reputation_score or 0.0) * weights["reputation"]
        )

        if bertimbau_score is not None:
            final_score_0_1 += bertimbau_score * weights["bertimbau"]

        final_score = round(clamp(final_score_0_1) * 100, 2)

        label = cls._label_from_evidence(
            final_score=final_score,
            evidence_score=evidence_score,
            coverage_score=coverage_score,
            reputation_score=reputation_score,
            stance_stats=stance_stats,
            bertimbau_score=bertimbau_score,
        )

        return {
            "score": final_score,
            "label": label,
            "breakdown": {
                "evidence_score": round(evidence_score * 100, 2),
                "coverage_score": round(coverage_score * 100, 2),
                "evidence_component": round(evidence_component * 100, 2),
                "reputation_score": (
                    round(reputation_score * 100, 2)
                    if reputation_score is not None
                    else None
                ),
                "reputation_status": (
                    result.reputation.get("status")
                    if getattr(result, "reputation", None)
                    else None
                ),
                "bertimbau_score": (
                    round(bertimbau_score * 100, 2)
                    if bertimbau_score is not None
                    else None
                ),
                "bertimbau_status": (
                    result.classification.get("status")
                    if getattr(result, "classification", None)
                    else None
                ),
                "stance_stats": stance_stats,
                "min_valid_evidence_score": cls.MIN_VALID_EVIDENCE_SCORE,
                "min_coverage_for_reliable": cls.MIN_COVERAGE_FOR_RELIABLE,
                "min_coverage_for_partial": cls.MIN_COVERAGE_FOR_PARTIAL,
                "min_support_rate_for_reliable": cls.MIN_SUPPORT_RATE_FOR_RELIABLE,
                "min_contradiction_rate_for_unreliable": cls.MIN_CONTRADICTION_RATE_FOR_UNRELIABLE,
                "excluded_evidence_layers": sorted(cls.EXCLUDED_EVIDENCE_LAYERS),
                "excluded_source_types": sorted(cls.EXCLUDED_SOURCE_TYPES),
                "weights": weights,
            },
        }
