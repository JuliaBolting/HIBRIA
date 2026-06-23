# =============================================================================
# stance_model.py
# Identifica a relação entre uma claim e suas evidências recuperadas.
#
# Esta primeira versão usa regras heurísticas, sem IA generativa.
#
# Rótulos:
#   - support:      a evidência apoia a claim
#   - contradict:   a evidência contradiz a claim
#   - neutral:      a evidência fala do tema, mas não confirma/refuta
#   - insufficient: evidência ausente ou fraca
# =============================================================================

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

STANCE_SUPPORT = "support"
STANCE_CONTRADICT = "contradict"
STANCE_NEUTRAL = "neutral"
STANCE_INSUFFICIENT = "insufficient"


@dataclass
class StanceResult:
    claim_id: str
    evidence_id: str
    stance: str
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "evidence_id": self.evidence_id,
            "stance": self.stance,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }


class StanceModel:
    """
    Modelo inicial de stance detection baseado em regras.

    Ele não verifica a verdade absoluta da notícia. Apenas estima se uma
    evidência recuperada apoia, contradiz ou é neutra em relação à claim.
    """

    MIN_OVERLAP_SUPPORT = 0.45
    MIN_OVERLAP_NEUTRAL = 0.20
    MIN_EVIDENCE_SIMILARITY = 0.55

    NEGATION_TERMS = {
        "nao",
        "não",
        "nunca",
        "jamais",
        "nega",
        "negou",
        "negam",
        "negado",
        "falso",
        "falsa",
        "incorreto",
        "incorreta",
        "mentira",
        "sem",
    }

    CONTRADICTION_TERMS = {
        "diferente",
        "contrario",
        "contrário",
        "desmente",
        "desmentiu",
        "refuta",
        "refutou",
        "nega",
        "negou",
        "falso",
        "falsa",
        "inveridico",
        "inverídico",
    }

    STOPWORDS = {
        "a",
        "o",
        "os",
        "as",
        "um",
        "uma",
        "uns",
        "umas",
        "de",
        "da",
        "do",
        "das",
        "dos",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "por",
        "para",
        "com",
        "sem",
        "sobre",
        "entre",
        "que",
        "se",
        "ao",
        "aos",
        "e",
        "ou",
        "mas",
        "foi",
        "ser",
        "sao",
        "são",
        "esta",
        "está",
        "estao",
        "estão",
        "ter",
        "tem",
        "teve",
        "tambem",
        "também",
        "segundo",
        "apos",
        "após",
        "durante",
        "este",
        "esta",
        "esse",
        "essa",
        "isso",
        "aquele",
        "aquela",
    }

    @classmethod
    def analyze_retrieval_results(cls, retrieval_results: list) -> list[StanceResult]:
        """
        Analisa todos os resultados de recuperação.

        Espera receber result.retrieval_results, onde cada item possui:
          - claim
          - evidences

        Também preenche evidence.stance em cada evidência.
        """
        stance_results: list[StanceResult] = []

        if not retrieval_results:
            return stance_results

        for retrieval in retrieval_results:
            claim = retrieval.claim
            evidences = retrieval.evidences or []

            if not evidences:
                stance_results.append(
                    StanceResult(
                        claim_id=claim.claim_id,
                        evidence_id="",
                        stance=STANCE_INSUFFICIENT,
                        confidence=1.0,
                        reason="Nenhuma evidência foi recuperada para a claim.",
                    )
                )
                continue

            for evidence in evidences:
                result = cls.analyze_pair(claim, evidence)
                evidence.stance = result.stance
                stance_results.append(result)

        return stance_results

    @classmethod
    def analyze_pair(cls, claim, evidence) -> StanceResult:
        """
        Analisa uma claim contra uma evidência.
        """
        claim_text = getattr(claim, "text", "") or getattr(claim, "normalized", "")
        evidence_text = getattr(evidence, "text", "")
        evidence_id = getattr(evidence, "evidence_id", "")
        claim_id = getattr(claim, "claim_id", "")
        similarity = float(getattr(evidence, "similarity", 0.0) or 0.0)

        if not evidence_text or similarity < 0.05:
            return StanceResult(
                claim_id=claim_id,
                evidence_id=evidence_id,
                stance=STANCE_INSUFFICIENT,
                confidence=1.0,
                reason="Evidência ausente ou com similaridade muito baixa.",
            )

        claim_norm = cls._normalize_text(claim_text)
        evidence_norm = cls._normalize_text(evidence_text)

        claim_numbers = cls._extract_numbers(claim_norm)
        evidence_numbers = cls._extract_numbers(evidence_norm)

        overlap = cls._keyword_overlap(claim_norm, evidence_norm)

        has_contradiction_marker = cls._has_any_term(
            evidence_norm,
            cls.CONTRADICTION_TERMS,
        )

        claim_has_negation = cls._has_any_term(claim_norm, cls.NEGATION_TERMS)
        evidence_has_negation = cls._has_any_term(evidence_norm, cls.NEGATION_TERMS)

        numbers_conflict = cls._numbers_conflict(
            claim_numbers,
            evidence_numbers,
            overlap,
        )

        if numbers_conflict:
            return StanceResult(
                claim_id=claim_id,
                evidence_id=evidence_id,
                stance=STANCE_CONTRADICT,
                confidence=0.75,
                reason="A evidência trata de tema semelhante, mas apresenta números diferentes da claim.",
            )

        if has_contradiction_marker and overlap >= cls.MIN_OVERLAP_NEUTRAL:
            return StanceResult(
                claim_id=claim_id,
                evidence_id=evidence_id,
                stance=STANCE_CONTRADICT,
                confidence=0.65,
                reason="A evidência contém marcador textual de refutação ou negação relacionado ao tema da claim.",
            )

        if (
            claim_has_negation != evidence_has_negation
            and overlap >= cls.MIN_OVERLAP_SUPPORT
        ):
            return StanceResult(
                claim_id=claim_id,
                evidence_id=evidence_id,
                stance=STANCE_CONTRADICT,
                confidence=0.60,
                reason="Claim e evidência apresentam diferença de polaridade textual.",
            )

        if (
            similarity >= cls.MIN_EVIDENCE_SIMILARITY
            and overlap >= cls.MIN_OVERLAP_SUPPORT
        ):
            return StanceResult(
                claim_id=claim_id,
                evidence_id=evidence_id,
                stance=STANCE_SUPPORT,
                confidence=0.75,
                reason="A evidência possui alta similaridade e boa sobreposição de termos com a claim.",
            )

        if overlap >= cls.MIN_OVERLAP_NEUTRAL:
            return StanceResult(
                claim_id=claim_id,
                evidence_id=evidence_id,
                stance=STANCE_NEUTRAL,
                confidence=0.55,
                reason="A evidência menciona parte do tema, mas não confirma nem contradiz claramente a claim.",
            )

        return StanceResult(
            claim_id=claim_id,
            evidence_id=evidence_id,
            stance=STANCE_INSUFFICIENT,
            confidence=0.70,
            reason="A evidência recuperada não possui relação textual suficiente com a claim.",
        )

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        text = text or ""
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.replace("&quot;", " ")
        text = text.replace("&amp;", " ")
        text = text.lower()

        text = unicodedata.normalize("NFD", text)
        text = "".join(char for char in text if unicodedata.category(char) != "Mn")

        text = re.sub(r"[^a-z0-9%., ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        tokens = re.findall(r"\b[a-z0-9]{3,}\b", text)

        return {token for token in tokens if token not in cls.STOPWORDS}

    @classmethod
    def _keyword_overlap(cls, claim_text: str, evidence_text: str) -> float:
        claim_tokens = cls._tokens(claim_text)
        evidence_tokens = cls._tokens(evidence_text)

        if not claim_tokens or not evidence_tokens:
            return 0.0

        overlap = claim_tokens.intersection(evidence_tokens)

        return round(len(overlap) / len(claim_tokens), 4)

    @staticmethod
    def _extract_numbers(text: str) -> list[str]:
        """
        Extrai números simples, porcentagens e valores monetários aproximados.
        """
        patterns = [
            r"r\$\s?\d+(?:[.,]\d+)?",
            r"\d+(?:[.,]\d+)?\s?%",
            r"\d+(?:[.,]\d+)?",
        ]

        numbers: list[str] = []

        for pattern in patterns:
            numbers.extend(re.findall(pattern, text))

        return list(dict.fromkeys(numbers))

    @staticmethod
    def _has_any_term(text: str, terms: set[str]) -> bool:
        tokens = set(re.findall(r"\b\w+\b", text))
        normalized_terms = {term.lower() for term in terms}

        return bool(tokens.intersection(normalized_terms))

    @staticmethod
    def _numbers_conflict(
        claim_numbers: list[str],
        evidence_numbers: list[str],
        overlap: float,
    ) -> bool:
        """
        Considera conflito numérico apenas em casos fortes.

        Números diferentes só indicam possível contradição quando:
        - claim e evidência têm boa sobreposição temática;
        - ambas possuem números;
        - não há nenhum número em comum;
        - os dois lados usam porcentagens ou valores monetários.
        """
        if overlap < 0.60:
            return False

        if not claim_numbers or not evidence_numbers:
            return False

        claim_set = set(claim_numbers)
        evidence_set = set(evidence_numbers)

        if claim_set.intersection(evidence_set):
            return False

        claim_has_percent = any("%" in n for n in claim_numbers)
        evidence_has_percent = any("%" in n for n in evidence_numbers)

        claim_has_money = any("r$" in n.lower() for n in claim_numbers)
        evidence_has_money = any("r$" in n.lower() for n in evidence_numbers)

        if claim_has_percent and evidence_has_percent:
            return True

        if claim_has_money and evidence_has_money:
            return True

        return False