# =============================================================================
# aggregator.py
# Combina os resultados parciais do pipeline em um score final de confiabilidade.
#
# Nesta versão inicial, o score considera:
#   - Evidências recuperadas pelo RAG / FAISS
#   - Similaridade entre claims e evidências
#   - Reputação do domínio da notícia analisada
#
# Futuramente, este módulo poderá incorporar:
#   - stance_model
#   - bertimbau_classifier
#   - text_features
# =============================================================================

from __future__ import annotations
import os


def env_float(name: str, default: float) -> float:
    """
    Lê float do .env com fallback seguro.
    """
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return float(value)
    except ValueError:
        return default


class Aggregator:
    """
    Agrega os scores parciais da análise em um índice final.

    O resultado não representa uma verdade absoluta, mas um índice de
    confiabilidade calculado a partir dos critérios disponíveis no sistema.
    """

    MIN_VALID_EVIDENCE_SCORE: float = env_float("HIBRIA_MIN_VALID_EVIDENCE_SCORE", 0.55)
    MIN_COVERAGE_FOR_RELIABLE: float = env_float("HIBRIA_MIN_COVERAGE_FOR_RELIABLE", 0.35)
    MIN_COVERAGE_FOR_PARTIAL: float = env_float("HIBRIA_MIN_COVERAGE_FOR_PARTIAL", 0.20)
    MIN_SUPPORT_RATE_FOR_RELIABLE: float = env_float("HIBRIA_MIN_SUPPORT_RATE_FOR_RELIABLE", 0.60)
    MIN_CONTRADICTION_RATE_FOR_UNRELIABLE: float = env_float("HIBRIA_MIN_CONTRADICTION_RATE_FOR_UNRELIABLE", 0.40)

    # Wikipedia é mantida como contexto, mas não conta como evidência factual principal.
    EXCLUDED_EVIDENCE_LAYERS = {"wikipedia"}
    EXCLUDED_SOURCE_TYPES = {"encyclopedia"}

    WEIGHTS = {
        "evidence": env_float("HIBRIA_AGGREGATOR_EVIDENCE_WEIGHT", 0.70),
        "reputation": env_float("HIBRIA_AGGREGATOR_REPUTATION_WEIGHT", 0.30),
    }

    @classmethod
    def _is_valid_factual_evidence(cls, item) -> bool:
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
    def _calculate_evidence_score(cls, result) -> float:
        """
        Calcula o score médio das claims que possuem evidência válida.

        Uma evidência só é considerada válida quando passa do limiar mínimo
        de similaridade final. Isso evita que evidências apenas superficialmente
        relacionadas aumentem o score de confiabilidade.
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
    def _calculate_coverage(cls, result) -> float:
        """
        Mede a cobertura de evidências válidas:
        quantas claims possuem evidência acima do limiar mínimo em relação
        ao total de claims analisadas.
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
    def _calculate_reputation_score(result) -> float:
        """
        Recupera o score de reputação calculado pelo reputation_engine.py.
        """
        if not result.reputation:
            return 0.0

        return float(result.reputation.get("score", 0.0))

    @staticmethod
    def _calculate_stance_stats(result) -> dict:
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

        # Aqui ignoramos "insufficient", porque ausência de evidência
        # não é apoio nem contradição.
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
        reputation_score: float,
        stance_stats: dict,
    ) -> str:
        """
        Define o rótulo final sem confundir ausência de evidência com falsidade.

        Regras:
        - Contradição forte -> não confiável.
        - Sem evidência suficiente -> evidência insuficiente.
        - Boa evidência + boa cobertura -> confiável.
        - Evidência moderada -> parcialmente confiável.
        """
        contradiction_rate = stance_stats.get("contradiction_rate", 0.0)
        support_rate = stance_stats.get("support_rate", 0.0)

        if contradiction_rate >= cls.MIN_CONTRADICTION_RATE_FOR_UNRELIABLE:
            return "não confiável"

        if coverage_score == 0 or evidence_score == 0:
            if reputation_score >= 0.80:
                return "evidência insuficiente"
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
    def aggregate(cls, result) -> dict:
        evidence_score = cls._calculate_evidence_score(result)
        coverage_score = cls._calculate_coverage(result)
        reputation_score = cls._calculate_reputation_score(result)
        stance_stats = cls._calculate_stance_stats(result)

        # Cobertura baixa reduz o componente de evidência,
        # mas não transforma automaticamente a notícia em "não confiável".
        evidence_component = evidence_score * coverage_score

        weight_evidence = cls.WEIGHTS.get("evidence", 0.70)
        weight_reputation = cls.WEIGHTS.get("reputation", 0.30)

        weight_sum = weight_evidence + weight_reputation

        if weight_sum <= 0:
            weight_evidence = 0.70
            weight_reputation = 0.30
            weight_sum = 1.0

        weight_evidence = weight_evidence / weight_sum
        weight_reputation = weight_reputation / weight_sum

        final_score_0_1 = (
            evidence_component * weight_evidence
            + reputation_score * weight_reputation
        )

        final_score = round(final_score_0_1 * 100, 2)

        label = cls._label_from_evidence(
            final_score=final_score,
            evidence_score=evidence_score,
            coverage_score=coverage_score,
            reputation_score=reputation_score,
            stance_stats=stance_stats,
        )

        return {
            "score": final_score,
            "label": label,
            "breakdown": {
                "evidence_score": round(evidence_score * 100, 2),
                "coverage_score": round(coverage_score * 100, 2),
                "evidence_component": round(evidence_component * 100, 2),
                "reputation_score": round(reputation_score * 100, 2),
                "stance_stats": stance_stats,
                "min_valid_evidence_score": cls.MIN_VALID_EVIDENCE_SCORE,
                "min_coverage_for_reliable": cls.MIN_COVERAGE_FOR_RELIABLE,
                "min_coverage_for_partial": cls.MIN_COVERAGE_FOR_PARTIAL,
                "min_support_rate_for_reliable": cls.MIN_SUPPORT_RATE_FOR_RELIABLE,
                "min_contradiction_rate_for_unreliable": cls.MIN_CONTRADICTION_RATE_FOR_UNRELIABLE,
                "excluded_evidence_layers": sorted(cls.EXCLUDED_EVIDENCE_LAYERS),
                "excluded_source_types": sorted(cls.EXCLUDED_SOURCE_TYPES),
                "weights": {
                    "evidence": round(weight_evidence, 4),
                    "reputation": round(weight_reputation, 4),
                },
            },
        }
