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


class Aggregator:
    """
    Agrega os scores parciais da análise em um índice final.

    O resultado não representa uma verdade absoluta, mas um índice de
    confiabilidade calculado a partir dos critérios disponíveis no sistema.
    """

    MIN_VALID_EVIDENCE_SCORE = 0.55

    WEIGHTS = {
        "evidence": 0.70,
        "reputation": 0.30,
    }

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
            if item.has_evidence
            and item.score is not None
            and item.score >= cls.MIN_VALID_EVIDENCE_SCORE
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
            if item.has_evidence
            and item.score is not None
            and item.score >= cls.MIN_VALID_EVIDENCE_SCORE
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
    def _label_from_score(score: float) -> str:
        """
        Converte o score numérico em rótulo textual.
        """
        if score >= 75:
            return "confiável"

        if score >= 50:
            return "parcialmente confiável"

        return "não confiável"

    @classmethod
    def aggregate(cls, result) -> dict:
        evidence_score = cls._calculate_evidence_score(result)
        coverage_score = cls._calculate_coverage(result)
        reputation_score = cls._calculate_reputation_score(result)

        # Penaliza quando poucas claims têm evidência.
        # Exemplo: evidências muito boas, mas só para 2 de 20 claims,
        # não devem gerar score final alto demais.
        evidence_component = evidence_score * coverage_score

        final_score_0_1 = (
            evidence_component * cls.WEIGHTS["evidence"]
            + reputation_score * cls.WEIGHTS["reputation"]
        )

        final_score = round(final_score_0_1 * 100, 2)
        label = cls._label_from_score(final_score)

        return {
            "score": final_score,
            "label": label,
            "breakdown": {
                "evidence_score": round(evidence_score * 100, 2),
                "coverage_score": round(coverage_score * 100, 2),
                "evidence_component": round(evidence_component * 100, 2),
                "reputation_score": round(reputation_score * 100, 2),
                "min_valid_evidence_score": cls.MIN_VALID_EVIDENCE_SCORE,
                "weights": cls.WEIGHTS,
            },
        }
