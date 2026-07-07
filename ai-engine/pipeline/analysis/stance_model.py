# =============================================================================
# stance_model.py
# Identifica a relação entre uma claim e as evidências recuperadas pelo RAG.
#
# Esta versão usa regras heurísticas, sem IA generativa. A função do módulo não é
# decidir a verdade absoluta da notícia, mas classificar se a evidência encontrada
# apoia, contradiz, é neutra ou insuficiente em relação à afirmação analisada.
#
# Fluxo esperado:
#   claim_detector.py → retriever.py → similarity.py → stance_model.py
#
# Rótulos usados pelo aggregator.py:
#   - support:      a evidência apoia a claim
#   - contradict:   a evidência contradiz a claim
#   - neutral:      a evidência fala do tema, mas não confirma/refuta
#   - insufficient: evidência ausente, fraca ou abaixo do corte mínimo
# =============================================================================

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

STANCE_SUPPORT = "support"
STANCE_CONTRADICT = "contradict"
STANCE_NEUTRAL = "neutral"
STANCE_INSUFFICIENT = "insufficient"


def env_float(name: str, default: float) -> float:
    """Lê float do .env com fallback seguro."""
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return float(value)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    """Lê inteiro do .env com fallback seguro."""
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class StanceResult:
    """Resultado da relação entre uma claim e uma evidência."""

    claim_id: str
    evidence_id: str
    stance: str
    confidence: float
    reason: str

    similarity: float = 0.0
    overlap: float = 0.0
    source: str = ""
    url: str = ""
    retrieval_layer: str = ""
    source_type: str = ""
    trusted_source: bool = False

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "evidence_id": self.evidence_id,
            "stance": self.stance,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "similarity": round(self.similarity, 4),
            "overlap": round(self.overlap, 4),
            "source": self.source,
            "url": self.url,
            "retrieval_layer": self.retrieval_layer,
            "source_type": self.source_type,
            "trusted_source": self.trusted_source,
        }


class StanceModel:
    """
    Modelo heurístico de stance detection da HÍBRIA.

    O módulo usa três sinais principais:
    1. similaridade final calculada em similarity.py;
    2. sobreposição de termos relevantes entre claim e evidência;
    3. marcadores linguísticos de negação, refutação e conflito numérico.

    A ausência de evidência suficiente é classificada como "insufficient", nunca
    como contradição. Isso evita penalizar automaticamente notícias sobre temas
    pouco cobertos pela base ou por APIs externas.
    """

    MIN_SIMILARITY: float = env_float("HIBRIA_STANCE_MIN_SIMILARITY", 0.55)
    MIN_OVERLAP_SUPPORT: float = env_float("HIBRIA_STANCE_MIN_OVERLAP_SUPPORT", 0.42)
    MIN_OVERLAP_NEUTRAL: float = env_float("HIBRIA_STANCE_MIN_OVERLAP_NEUTRAL", 0.18)
    HIGH_SIMILARITY: float = env_float("HIBRIA_STANCE_HIGH_SIMILARITY", 0.72)
    TOP_K: int = env_int("HIBRIA_STANCE_TOP_K", 3)

    CONTEXT_ONLY_LAYERS = {"wikipedia"}
    CONTEXT_ONLY_SOURCE_TYPES = {"encyclopedia"}

    NEGATION_TERMS = {
        "nao",
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
        "improcedente",
        "mentira",
        "sem",
    }

    CONTRADICTION_TERMS = {
        "boato",
        "desinformacao",
        "desmente",
        "desmentem",
        "desmentiu",
        "desmentido",
        "diferente",
        "enganoso",
        "enganosa",
        "falso",
        "falsa",
        "fake",
        "impreciso",
        "imprecisa",
        "incorreto",
        "incorreta",
        "inveridico",
        "inveridica",
        "mentira",
        "nega",
        "negou",
        "refuta",
        "refutou",
    }

    CONTRADICTION_PHRASES = {
        "nao e verdade",
        "nao procede",
        "e falso",
        "e falsa",
        "conteudo falso",
        "conteudo enganoso",
        "informacao falsa",
        "informacao enganosa",
        "alegacao falsa",
        "afirmacao falsa",
        "sem evidencia",
        "sem provas",
        "nao ha evidencia",
        "nao ha provas",
        "nao encontrou evidencia",
        "e boato",
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
        "ate",
        "apos",
        "durante",
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
        "esta",
        "estao",
        "estar",
        "ter",
        "tem",
        "teve",
        "tambem",
        "segundo",
        "este",
        "esse",
        "essa",
        "isso",
        "aquele",
        "aquela",
        "como",
        "mais",
        "menos",
        "muito",
        "muita",
        "muitos",
        "muitas",
        "sobre",
        "pela",
        "pelo",
        "pelos",
        "pelas",
        "sua",
        "seu",
        "suas",
        "seus",
    }

    @classmethod
    def analyze(
        cls,
        retrieval_results: list | None,
        similarity_scores: list | None = None,
    ) -> list[StanceResult]:
        """
        Analisa stance usando, preferencialmente, as evidências rerankeadas pelo
        similarity.py. Se similarity_scores não existir, usa as evidências brutas
        do retriever como fallback.
        """
        if similarity_scores:
            return cls.analyze_similarity_results(retrieval_results or [], similarity_scores)

        return cls.analyze_retrieval_results(retrieval_results or [])

    @classmethod
    def analyze_similarity_results(
        cls,
        retrieval_results: list,
        similarity_scores: list,
    ) -> list[StanceResult]:
        """
        Analisa as melhores evidências por claim após o reranking semântico.

        Para cada claim, considera somente evidências suficientes, limitadas por
        HIBRIA_STANCE_TOP_K. Quando uma claim não possui evidência suficiente, o
        resultado é "insufficient".
        """
        stance_results: list[StanceResult] = []

        claim_map = cls._build_claim_map(retrieval_results)
        evidence_lookup = cls._build_evidence_lookup(retrieval_results)

        for similarity_result in similarity_scores:
            claim_id = getattr(similarity_result, "claim_id", "")
            claim = claim_map.get(claim_id)
            claim_text = getattr(similarity_result, "claim_text", "")

            if claim is None:
                claim = cls._make_claim_adapter(claim_id=claim_id, text=claim_text)

            candidate_evidences = list(getattr(similarity_result, "evidences", []) or [])
            sufficient_evidences = [
                item
                for item in candidate_evidences
                if bool(getattr(item, "is_sufficient", False))
                and float(getattr(item, "similarity_final", 0.0) or 0.0) >= cls.MIN_SIMILARITY
            ]

            if not sufficient_evidences:
                stance_results.append(
                    StanceResult(
                        claim_id=claim_id,
                        evidence_id="",
                        stance=STANCE_INSUFFICIENT,
                        confidence=0.9,
                        reason="A claim não possui evidência externa suficiente após o cálculo de similaridade.",
                    )
                )
                continue

            for evidence_similarity in sufficient_evidences[: cls.TOP_K]:
                original_evidence = cls._find_original_evidence(
                    evidence_lookup,
                    claim_id=claim_id,
                    evidence_similarity=evidence_similarity,
                )

                evidence_for_analysis = original_evidence or evidence_similarity
                similarity = float(
                    getattr(evidence_similarity, "similarity_final", 0.0) or 0.0
                )

                result = cls.analyze_pair(
                    claim,
                    evidence_for_analysis,
                    similarity_override=similarity,
                )

                if original_evidence is not None:
                    original_evidence.stance = result.stance
                    metadata = getattr(original_evidence, "metadata", None)
                    if isinstance(metadata, dict):
                        metadata["stance_confidence"] = round(result.confidence, 4)
                        metadata["stance_reason"] = result.reason
                        metadata["stance_similarity"] = round(result.similarity, 4)
                        metadata["stance_overlap"] = round(result.overlap, 4)

                stance_results.append(result)

        return stance_results

    @classmethod
    def analyze_retrieval_results(cls, retrieval_results: list) -> list[StanceResult]:
        """
        Fallback compatível com versões anteriores: analisa as evidências brutas
        do retriever quando a etapa similarity.py ainda não foi executada.
        """
        stance_results: list[StanceResult] = []

        if not retrieval_results:
            return stance_results

        for retrieval in retrieval_results:
            claim = getattr(retrieval, "claim", None)
            evidences = list(getattr(retrieval, "evidences", []) or [])

            if claim is None:
                continue

            if not evidences:
                stance_results.append(
                    StanceResult(
                        claim_id=getattr(claim, "claim_id", ""),
                        evidence_id="",
                        stance=STANCE_INSUFFICIENT,
                        confidence=1.0,
                        reason="Nenhuma evidência foi recuperada para a claim.",
                    )
                )
                continue

            sorted_evidences = sorted(
                evidences,
                key=lambda item: float(getattr(item, "similarity", 0.0) or 0.0),
                reverse=True,
            )

            used = 0
            for evidence in sorted_evidences:
                if used >= cls.TOP_K:
                    break

                result = cls.analyze_pair(claim, evidence)
                evidence.stance = result.stance
                stance_results.append(result)
                used += 1

        return stance_results

    @classmethod
    def analyze_pair(
        cls,
        claim: Any,
        evidence: Any,
        similarity_override: float | None = None,
    ) -> StanceResult:
        """Analisa uma claim contra uma evidência."""
        claim_text = cls._first_attr(claim, "text", "claim_text", "normalized")
        claim_id = cls._first_attr(claim, "claim_id")

        evidence_text = cls._first_attr(evidence, "text", "evidence_text")
        evidence_id = cls._first_attr(evidence, "evidence_id")
        evidence_source = cls._first_attr(evidence, "source", "evidence_source")
        evidence_url = cls._first_attr(evidence, "url", "evidence_url")
        evidence_layer = cls._first_attr(evidence, "retrieval_layer", "evidence_layer")
        source_type = cls._first_attr(evidence, "source_type")
        trusted_source = bool(getattr(evidence, "trusted_source", False))

        similarity = cls._get_similarity(evidence, similarity_override)

        if not evidence_id:
            evidence_id = cls._fallback_evidence_id(
                claim_id=claim_id,
                evidence_layer=evidence_layer,
                evidence_url=evidence_url,
                evidence_text=evidence_text,
            )

        base_kwargs = {
            "claim_id": claim_id,
            "evidence_id": evidence_id,
            "similarity": similarity,
            "source": evidence_source,
            "url": evidence_url,
            "retrieval_layer": evidence_layer,
            "source_type": source_type,
            "trusted_source": trusted_source,
        }

        if not claim_text or not evidence_text:
            return StanceResult(
                stance=STANCE_INSUFFICIENT,
                confidence=1.0,
                reason="Claim ou evidência ausente para análise de stance.",
                **base_kwargs,
            )

        claim_norm = cls._normalize_text(claim_text)
        evidence_norm = cls._normalize_text(evidence_text)
        overlap = cls._keyword_overlap(claim_norm, evidence_norm)
        base_kwargs["overlap"] = overlap

        if similarity < cls.MIN_SIMILARITY:
            return StanceResult(
                stance=STANCE_INSUFFICIENT,
                confidence=0.85,
                reason="A evidência está abaixo do limiar mínimo de similaridade para stance detection.",
                **base_kwargs,
            )

        if cls._is_context_only(evidence_layer, source_type):
            return StanceResult(
                stance=STANCE_NEUTRAL,
                confidence=0.60,
                reason="A evidência vem de fonte contextual/enciclopédica; serve como contexto, não como confirmação factual principal.",
                **base_kwargs,
            )

        claim_numbers = cls._extract_numbers(claim_norm)
        evidence_numbers = cls._extract_numbers(evidence_norm)

        has_contradiction_marker = cls._has_contradiction_marker(evidence_norm)
        claim_has_negation = cls._has_any_term(claim_norm, cls.NEGATION_TERMS)
        evidence_has_negation = cls._has_any_term(evidence_norm, cls.NEGATION_TERMS)
        numbers_conflict = cls._numbers_conflict(claim_numbers, evidence_numbers, overlap)

        if numbers_conflict:
            return StanceResult(
                stance=STANCE_CONTRADICT,
                confidence=0.78,
                reason="A evidência trata de tema semelhante, mas apresenta números diferentes da claim.",
                **base_kwargs,
            )

        if has_contradiction_marker and overlap >= cls.MIN_OVERLAP_NEUTRAL:
            return StanceResult(
                stance=STANCE_CONTRADICT,
                confidence=0.70,
                reason="A evidência contém marcador textual de refutação ou checagem negativa relacionado ao tema da claim.",
                **base_kwargs,
            )

        if claim_has_negation != evidence_has_negation and overlap >= cls.MIN_OVERLAP_SUPPORT:
            return StanceResult(
                stance=STANCE_CONTRADICT,
                confidence=0.64,
                reason="Claim e evidência apresentam diferença relevante de polaridade textual.",
                **base_kwargs,
            )

        if overlap >= cls.MIN_OVERLAP_SUPPORT:
            return StanceResult(
                stance=STANCE_SUPPORT,
                confidence=cls._support_confidence(similarity, overlap),
                reason="A evidência possui similaridade suficiente e boa sobreposição de termos com a claim.",
                **base_kwargs,
            )

        if similarity >= cls.HIGH_SIMILARITY and overlap >= cls.MIN_OVERLAP_NEUTRAL:
            return StanceResult(
                stance=STANCE_SUPPORT,
                confidence=0.66,
                reason="A evidência apresenta alta similaridade semântica com a claim, apesar de menor sobreposição lexical.",
                **base_kwargs,
            )

        if overlap >= cls.MIN_OVERLAP_NEUTRAL:
            return StanceResult(
                stance=STANCE_NEUTRAL,
                confidence=0.58,
                reason="A evidência menciona parte do tema, mas não confirma nem contradiz claramente a claim.",
                **base_kwargs,
            )

        return StanceResult(
            stance=STANCE_INSUFFICIENT,
            confidence=0.75,
            reason="A evidência recuperada não possui relação textual suficiente com a claim.",
            **base_kwargs,
        )

    @classmethod
    def _build_claim_map(cls, retrieval_results: list) -> dict[str, Any]:
        claim_map: dict[str, Any] = {}
        for retrieval in retrieval_results or []:
            claim = getattr(retrieval, "claim", None)
            claim_id = getattr(claim, "claim_id", "") if claim else ""
            if claim_id:
                claim_map[claim_id] = claim
        return claim_map

    @classmethod
    def _build_evidence_lookup(cls, retrieval_results: list) -> dict[tuple[str, str, str], Any]:
        lookup: dict[tuple[str, str, str], Any] = {}

        for retrieval in retrieval_results or []:
            claim = getattr(retrieval, "claim", None)
            claim_id = getattr(claim, "claim_id", "") if claim else ""

            for evidence in getattr(retrieval, "evidences", []) or []:
                key = cls._evidence_key(
                    claim_id=claim_id,
                    url=getattr(evidence, "url", ""),
                    text=getattr(evidence, "text", ""),
                )
                lookup[key] = evidence

        return lookup

    @classmethod
    def _find_original_evidence(
        cls,
        evidence_lookup: dict[tuple[str, str, str], Any],
        claim_id: str,
        evidence_similarity: Any,
    ) -> Any | None:
        key = cls._evidence_key(
            claim_id=claim_id,
            url=getattr(evidence_similarity, "evidence_url", ""),
            text=getattr(evidence_similarity, "evidence_text", ""),
        )
        return evidence_lookup.get(key)

    @classmethod
    def _evidence_key(cls, claim_id: str, url: str, text: str) -> tuple[str, str, str]:
        return (
            claim_id or "",
            cls._canonical_url(url),
            cls._text_fingerprint(text),
        )

    @staticmethod
    def _canonical_url(url: str) -> str:
        if not url:
            return ""

        parsed = urlparse(url.strip())
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        path = parsed.path.rstrip("/").lower()
        return f"{domain}{path}"

    @classmethod
    def _text_fingerprint(cls, text: str) -> str:
        normalized = cls._normalize_text(text)[:500]
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _make_claim_adapter(claim_id: str, text: str):
        class ClaimAdapter:
            pass

        adapter = ClaimAdapter()
        adapter.claim_id = claim_id
        adapter.text = text
        adapter.normalized = text
        return adapter

    @staticmethod
    def _first_attr(obj: Any, *names: str) -> str:
        for name in names:
            value = getattr(obj, name, None)
            if value is not None and str(value).strip() != "":
                return str(value)
        return ""

    @staticmethod
    def _get_similarity(evidence: Any, override: float | None) -> float:
        if override is not None:
            try:
                return max(0.0, min(1.0, float(override)))
            except (TypeError, ValueError):
                return 0.0

        for attr in ("similarity_final", "similarity", "similarity_semantic"):
            value = getattr(evidence, attr, None)
            if value is None:
                continue
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue

        return 0.0

    @classmethod
    def _fallback_evidence_id(
        cls,
        claim_id: str,
        evidence_layer: str,
        evidence_url: str,
        evidence_text: str,
    ) -> str:
        raw = f"{claim_id}|{evidence_layer}|{evidence_url}|{evidence_text[:300]}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        text = text or ""
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.replace("&quot;", " ").replace("&amp;", " ")
        text = text.lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join(char for char in text if unicodedata.category(char) != "Mn")
        text = re.sub(r"[^a-z0-9%$., ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        tokens = re.findall(r"\b[a-z0-9][a-z0-9-]{2,}\b", text)
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
        """Extrai números, porcentagens e valores monetários simples."""
        pattern = r"(?:r\$\s*)?\d+(?:[.,]\d+)?%?"
        numbers = re.findall(pattern, text or "")
        normalized = [re.sub(r"\s+", "", item).replace(",", ".") for item in numbers]
        return list(dict.fromkeys(normalized))

    @classmethod
    def _has_any_term(cls, text: str, terms: set[str]) -> bool:
        tokens = set(re.findall(r"\b\w+\b", cls._normalize_text(text)))
        normalized_terms = {cls._normalize_text(term) for term in terms}
        return bool(tokens.intersection(normalized_terms))

    @classmethod
    def _has_contradiction_marker(cls, text: str) -> bool:
        normalized = cls._normalize_text(text)

        if cls._has_any_term(normalized, cls.CONTRADICTION_TERMS):
            return True

        return any(phrase in normalized for phrase in cls.CONTRADICTION_PHRASES)

    @classmethod
    def _numbers_conflict(
        cls,
        claim_numbers: list[str],
        evidence_numbers: list[str],
        overlap: float,
    ) -> bool:
        """
        Considera conflito numérico apenas em casos fortes.

        Números diferentes só indicam possível contradição quando claim e
        evidência têm boa sobreposição temática e usam tipos comparáveis de
        número, como porcentagens ou valores monetários.
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

    @classmethod
    def _is_context_only(cls, evidence_layer: str, source_type: str) -> bool:
        return (
            cls._normalize_text(evidence_layer) in cls.CONTEXT_ONLY_LAYERS
            or cls._normalize_text(source_type) in cls.CONTEXT_ONLY_SOURCE_TYPES
        )

    @staticmethod
    def _support_confidence(similarity: float, overlap: float) -> float:
        score = 0.55 + (similarity * 0.25) + (overlap * 0.20)
        return round(max(0.60, min(0.88, score)), 4)
