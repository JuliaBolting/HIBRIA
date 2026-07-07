# =============================================================================
# text_features.py
# Extrai características linguísticas e estruturais do texto analisado.
#
# Este módulo NÃO decide se uma notícia é verdadeira ou falsa. Ele identifica
# sinais textuais que podem influenciar a confiabilidade percebida, como:
#   - sensacionalismo;
#   - linguagem emocional;
#   - clickbait;
#   - excesso de pontuação/caixa alta;
#   - presença de números, datas, citações e atribuições a fontes.
#
# Fluxo esperado:
#   extractor/cleaner → segmentation → claim_detector → text_features
#
# Saída principal:
#   result.text_features = {
#       "status": "ok",
#       "score": 0.0-1.0,       # score linguístico: maior = menor risco textual
#       "risk_score": 0.0-1.0,  # risco textual: maior = mais sinais problemáticos
#       "label": "baixo risco linguístico" | "atenção linguística" | "alto risco linguístico",
#       "features": {...},
#       "stats": {...},
#       "flags": [...]
#   }
# =============================================================================

from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable


# =============================================================================
# Helpers
# =============================================================================


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



def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))



def normalize_text(text: str) -> str:
    """
    Normaliza texto para comparação lexical.

    Remove acentos e coloca em caixa baixa, sem alterar o texto original usado
    nas estatísticas de pontuação e caixa alta.
    """
    text = text or ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text



def safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


# =============================================================================
# TextFeatureExtractor
# =============================================================================


class TextFeatureExtractor:
    """
    Extrator heurístico de características textuais da HÍBRIA.

    A ideia é gerar sinais explicáveis e baratos computacionalmente, úteis para
    o aggregator.py e para o explanation_generator.py. Nenhuma lista abaixo é
    vinculada a uma notícia específica; são marcadores gerais de estilo textual.
    """

    MIN_TEXT_CHARS: int = env_int("HIBRIA_TEXT_FEATURES_MIN_CHARS", 120)

    # Limiar usado apenas para gerar flags interpretáveis.
    HIGH_FEATURE_THRESHOLD: float = env_float(
        "HIBRIA_TEXT_FEATURES_HIGH_THRESHOLD",
        0.60,
    )
    MEDIUM_FEATURE_THRESHOLD: float = env_float(
        "HIBRIA_TEXT_FEATURES_MEDIUM_THRESHOLD",
        0.35,
    )

    # Pesos internos do risco textual. A soma não precisa ser 1; normalizamos.
    RISK_WEIGHTS = {
        "sensationalism": env_float("HIBRIA_TEXT_FEATURES_SENSATIONALISM_WEIGHT", 0.24),
        "emotional_language": env_float("HIBRIA_TEXT_FEATURES_EMOTIONAL_WEIGHT", 0.18),
        "clickbait": env_float("HIBRIA_TEXT_FEATURES_CLICKBAIT_WEIGHT", 0.18),
        "punctuation_excess": env_float("HIBRIA_TEXT_FEATURES_PUNCTUATION_WEIGHT", 0.10),
        "uppercase_excess": env_float("HIBRIA_TEXT_FEATURES_UPPERCASE_WEIGHT", 0.08),
        "subjectivity": env_float("HIBRIA_TEXT_FEATURES_SUBJECTIVITY_WEIGHT", 0.12),
        "weak_attribution": env_float("HIBRIA_TEXT_FEATURES_WEAK_ATTRIBUTION_WEIGHT", 0.10),
    }

    # -------------------------------------------------------------------------
    # Marcadores de risco linguístico
    # -------------------------------------------------------------------------

    SENSATIONALISM_TERMS = {
        "absurdo",
        "assustador",
        "bizarro",
        "bomba",
        "bombastico",
        "caos",
        "chocante",
        "desesperador",
        "escandalo",
        "estarrecedor",
        "explosivo",
        "inacreditavel",
        "imperdivel",
        "incrivel",
        "nunca visto",
        "segredo",
        "surpreendente",
        "urgente",
        "viral",
        "viralizou",
    }

    EMOTIONAL_TERMS = {
        "agonia",
        "ameaca",
        "assusta",
        "assustou",
        "choque",
        "destruiu",
        "devastador",
        "dor",
        "furia",
        "humilhacao",
        "indignacao",
        "medo",
        "odio",
        "panico",
        "revolta",
        "sofrimento",
        "terror",
        "tragedia",
        "vergonha",
    }

    CLICKBAIT_TERMS = {
        "clique aqui",
        "compartilhe antes",
        "entenda o que aconteceu",
        "isso vai mudar",
        "nao vao te contar",
        "nao querem que voce saiba",
        "o que ninguem contou",
        "saiba agora",
        "voce nao vai acreditar",
        "veja antes que apaguem",
        "veja o video",
        "veja o que aconteceu",
        "veja por que",
    }

    SUBJECTIVE_TERMS = {
        "acho",
        "acredito",
        "aparentemente",
        "claramente",
        "evidentemente",
        "lamentavel",
        "na minha opiniao",
        "parece",
        "provavelmente",
        "sem duvida",
        "supostamente",
        "talvez",
    }

    ABSOLUTE_CLAIM_TERMS = {
        "com certeza",
        "definitivamente",
        "jamais",
        "nunca",
        "prova definitiva",
        "sempre",
        "todos sabem",
        "verdade absoluta",
    }

    WEAK_ATTRIBUTION_TERMS = {
        "circula nas redes",
        "dizem que",
        "fontes dizem",
        "informacoes preliminares",
        "internautas afirmam",
        "segundo publicacoes",
        "sem confirmacao",
        "suposto",
        "suposta",
        "teria acontecido",
        "teria sido",
    }

    # -------------------------------------------------------------------------
    # Marcadores de qualidade informativa
    # -------------------------------------------------------------------------

    SOURCE_ATTRIBUTION_TERMS = {
        "afirmou",
        "apontou",
        "comunicou",
        "confirmou",
        "de acordo com",
        "declarou",
        "disse",
        "em nota",
        "informou",
        "levantamento",
        "pesquisa",
        "relatorio",
        "segundo",
    }

    DATA_TERMS = {
        "balanco",
        "censo",
        "dados",
        "estatistica",
        "indice",
        "levantamento",
        "pesquisa",
        "relatorio",
        "serie historica",
        "taxa",
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
        "sob",
        "sobre",
        "entre",
        "que",
        "se",
        "ao",
        "aos",
        "e",
        "ou",
        "mas",
        "como",
        "mais",
        "menos",
        "muito",
        "muita",
        "muitos",
        "muitas",
        "foi",
        "ser",
        "sao",
        "esta",
        "estao",
        "ter",
        "tem",
        "teve",
        "tambem",
        "ja",
        "ainda",
        "esse",
        "essa",
        "isso",
        "este",
        "esta",
        "ele",
        "ela",
        "eles",
        "elas",
        "sua",
        "seu",
        "suas",
        "seus",
    }

    WORD_RE = re.compile(r"\b[\wÀ-ÿ]+(?:[-'][\wÀ-ÿ]+)?\b", re.UNICODE)
    SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
    NUMBER_RE = re.compile(
        r"(?:R\$|US\$)?\s*\b\d+(?:[.,]\d+)?(?:%|\s*(?:mil|milhao|milhoes|bilhao|bilhoes|anos?|dias?|meses?|km|kg))?\b",
        re.IGNORECASE,
    )
    DATE_RE = re.compile(
        r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}|"
        r"janeiro|fevereiro|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\b",
        re.IGNORECASE,
    )
    URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
    QUOTE_RE = re.compile(r"[\"“”‘’']")

    # =========================================================================
    # API pública
    # =========================================================================

    @classmethod
    def extract(
        cls,
        blocks: Iterable[str] | str | None,
        *,
        title: str = "",
        description: str = "",
        sentences: Iterable[Any] | None = None,
        claims: Iterable[Any] | None = None,
    ) -> dict:
        """
        Extrai características textuais do artigo.

        `blocks` é a entrada principal, normalmente result.blocks_clean.
        `title`, `description`, `sentences` e `claims` são opcionais e servem
        para enriquecer as métricas quando o pipeline já os calculou.
        """
        clean_blocks = cls._coerce_blocks(blocks)
        article_text = cls._join_blocks(clean_blocks)
        title = re.sub(r"\s+", " ", title or "").strip()
        description = re.sub(r"\s+", " ", description or "").strip()

        if len(article_text) < cls.MIN_TEXT_CHARS:
            return {
                "status": "empty",
                "score": 0.0,
                "risk_score": 0.0,
                "label": "texto insuficiente",
                "message": "Texto insuficiente para extrair características linguísticas.",
                "features": {},
                "stats": {
                    "block_count": len(clean_blocks),
                    "char_count": len(article_text),
                },
                "flags": [],
            }

        full_text_for_markers = "\n".join(
            part for part in [title, description, article_text] if part
        )

        sentence_texts = cls._extract_sentence_texts(article_text, sentences)
        stats = cls._calculate_stats(
            article_text=article_text,
            title=title,
            clean_blocks=clean_blocks,
            sentence_texts=sentence_texts,
            claims=claims,
        )
        features = cls._calculate_features(
            article_text=article_text,
            title=title,
            full_text_for_markers=full_text_for_markers,
            stats=stats,
        )
        risk_score = cls._calculate_risk_score(features)
        score = round(1.0 - risk_score, 4)
        label = cls._label_from_risk(risk_score)
        flags = cls._build_flags(features=features, stats=stats, risk_score=risk_score)

        return {
            "status": "ok",
            "score": score,
            "risk_score": round(risk_score, 4),
            "label": label,
            "features": features,
            "stats": stats,
            "flags": flags,
        }

    # =========================================================================
    # Preparação de texto
    # =========================================================================

    @staticmethod
    def _coerce_blocks(blocks: Iterable[str] | str | None) -> list[str]:
        if blocks is None:
            return []

        if isinstance(blocks, str):
            blocks = [blocks]

        clean_blocks: list[str] = []

        for block in blocks:
            text = re.sub(r"\s+", " ", str(block or "")).strip()
            if text:
                clean_blocks.append(text)

        return clean_blocks

    @staticmethod
    def _join_blocks(blocks: list[str]) -> str:
        return "\n\n".join(blocks).strip()

    @classmethod
    def _extract_sentence_texts(
        cls,
        article_text: str,
        sentences: Iterable[Any] | None,
    ) -> list[str]:
        if sentences:
            result: list[str] = []

            for item in sentences:
                text = getattr(item, "text", item)
                text = re.sub(r"\s+", " ", str(text or "")).strip()
                if text:
                    result.append(text)

            if result:
                return result

        return [
            s.strip()
            for s in cls.SENTENCE_SPLIT_RE.split(article_text)
            if len(s.strip()) >= 3
        ]

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        normalized = normalize_text(text)
        return cls.WORD_RE.findall(normalized)

    @classmethod
    def _content_tokens(cls, text: str) -> list[str]:
        return [
            token
            for token in cls._tokens(text)
            if len(token) >= 3 and token not in cls.STOPWORDS and not token.isdigit()
        ]

    # =========================================================================
    # Cálculo de estatísticas
    # =========================================================================

    @classmethod
    def _calculate_stats(
        cls,
        *,
        article_text: str,
        title: str,
        clean_blocks: list[str],
        sentence_texts: list[str],
        claims: Iterable[Any] | None,
    ) -> dict:
        tokens = cls._tokens(article_text)
        content_tokens = cls._content_tokens(article_text)
        unique_content_tokens = set(content_tokens)

        sentence_lengths = [len(cls._tokens(sentence)) for sentence in sentence_texts]
        sentence_lengths = [item for item in sentence_lengths if item > 0]

        uppercase_letters = sum(1 for ch in article_text if ch.isupper())
        all_letters = sum(1 for ch in article_text if ch.isalpha())
        punctuation_count = sum(1 for ch in article_text if ch in ".,;:!?…")
        exclamation_count = article_text.count("!")
        question_count = article_text.count("?")

        numbers = cls.NUMBER_RE.findall(article_text)
        dates = cls.DATE_RE.findall(normalize_text(article_text))
        quotes = cls.QUOTE_RE.findall(article_text)
        urls = cls.URL_RE.findall(article_text)

        claim_count = 0
        if claims:
            try:
                claim_count = sum(1 for _ in claims)
            except TypeError:
                claim_count = 0

        return {
            "block_count": len(clean_blocks),
            "char_count": len(article_text),
            "word_count": len(tokens),
            "content_word_count": len(content_tokens),
            "unique_content_words": len(unique_content_tokens),
            "lexical_diversity": round(safe_div(len(unique_content_tokens), len(content_tokens)), 4),
            "sentence_count": len(sentence_texts),
            "avg_sentence_words": round(safe_div(sum(sentence_lengths), len(sentence_lengths)), 2),
            "max_sentence_words": max(sentence_lengths) if sentence_lengths else 0,
            "title_word_count": len(cls._tokens(title)),
            "uppercase_ratio": round(safe_div(uppercase_letters, all_letters), 4),
            "punctuation_density": round(safe_div(punctuation_count, max(len(tokens), 1)), 4),
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "numbers_count": len(numbers),
            "dates_count": len(dates),
            "quote_marks_count": len(quotes),
            "url_count": len(urls),
            "claim_count": claim_count,
        }

    # =========================================================================
    # Cálculo das features
    # =========================================================================

    @classmethod
    def _calculate_features(
        cls,
        *,
        article_text: str,
        title: str,
        full_text_for_markers: str,
        stats: dict,
    ) -> dict:
        normalized_text = normalize_text(full_text_for_markers)
        normalized_title = normalize_text(title)
        tokens = cls._content_tokens(full_text_for_markers)
        token_count = max(len(tokens), 1)

        sensationalism_hits = cls._count_marker_hits(normalized_text, cls.SENSATIONALISM_TERMS)
        emotional_hits = cls._count_marker_hits(normalized_text, cls.EMOTIONAL_TERMS)
        clickbait_hits = cls._count_marker_hits(normalized_text, cls.CLICKBAIT_TERMS)
        subjectivity_hits = cls._count_marker_hits(normalized_text, cls.SUBJECTIVE_TERMS)
        absolute_claim_hits = cls._count_marker_hits(normalized_text, cls.ABSOLUTE_CLAIM_TERMS)
        weak_attribution_hits = cls._count_marker_hits(normalized_text, cls.WEAK_ATTRIBUTION_TERMS)
        source_attribution_hits = cls._count_marker_hits(normalized_text, cls.SOURCE_ATTRIBUTION_TERMS)
        data_terms_hits = cls._count_marker_hits(normalized_text, cls.DATA_TERMS)

        # Título tem peso maior em clickbait/sensacionalismo, porque o usuário é
        # exposto primeiro a ele na extensão.
        title_sensationalism_hits = cls._count_marker_hits(
            normalized_title,
            cls.SENSATIONALISM_TERMS | cls.CLICKBAIT_TERMS,
        )

        sensationalism_score = cls._scaled_marker_score(
            sensationalism_hits + 1.5 * title_sensationalism_hits,
            token_count,
            scale=35,
        )
        emotional_language_score = cls._scaled_marker_score(
            emotional_hits,
            token_count,
            scale=30,
        )
        clickbait_score = cls._scaled_marker_score(
            clickbait_hits + title_sensationalism_hits,
            token_count,
            scale=25,
        )
        subjectivity_score = cls._scaled_marker_score(
            subjectivity_hits + 0.7 * absolute_claim_hits,
            token_count,
            scale=35,
        )
        weak_attribution_score = cls._weak_attribution_score(
            weak_attribution_hits=weak_attribution_hits,
            source_attribution_hits=source_attribution_hits,
            word_count=stats.get("word_count", 0),
        )
        punctuation_excess_score = cls._punctuation_excess_score(stats)
        uppercase_excess_score = cls._uppercase_excess_score(stats)
        factual_density_score = cls._factual_density_score(
            stats=stats,
            source_attribution_hits=source_attribution_hits,
            data_terms_hits=data_terms_hits,
        )
        readability_score = cls._readability_score(stats)

        return {
            "sensationalism": round(sensationalism_score, 4),
            "emotional_language": round(emotional_language_score, 4),
            "clickbait": round(clickbait_score, 4),
            "punctuation_excess": round(punctuation_excess_score, 4),
            "uppercase_excess": round(uppercase_excess_score, 4),
            "subjectivity": round(subjectivity_score, 4),
            "weak_attribution": round(weak_attribution_score, 4),
            "factual_density": round(factual_density_score, 4),
            "readability": round(readability_score, 4),
            "source_attribution_count": source_attribution_hits,
            "data_terms_count": data_terms_hits,
            "absolute_claim_count": absolute_claim_hits,
            "marker_counts": {
                "sensationalism": sensationalism_hits,
                "emotional_language": emotional_hits,
                "clickbait": clickbait_hits,
                "subjectivity": subjectivity_hits,
                "weak_attribution": weak_attribution_hits,
            },
        }

    @staticmethod
    def _count_marker_hits(normalized_text: str, markers: set[str]) -> int:
        total = 0

        for marker in markers:
            marker_norm = normalize_text(marker)

            if " " in marker_norm:
                total += normalized_text.count(marker_norm)
            else:
                total += len(re.findall(rf"\b{re.escape(marker_norm)}\b", normalized_text))

        return total

    @staticmethod
    def _scaled_marker_score(hits: float, token_count: int, *, scale: float) -> float:
        """
        Converte contagens raras em score 0-1 sem exigir muitos termos.

        Fórmula logarítmica: poucos marcadores já sinalizam risco, mas o score
        satura para evitar que textos longos sejam punidos em excesso.
        """
        if hits <= 0 or token_count <= 0:
            return 0.0

        density = hits / token_count
        return clamp(math.log1p(density * scale * 100) / math.log1p(scale))

    @staticmethod
    def _punctuation_excess_score(stats: dict) -> float:
        word_count = max(stats.get("word_count", 0), 1)
        exclamation_density = stats.get("exclamation_count", 0) / word_count
        question_density = stats.get("question_count", 0) / word_count
        punctuation_density = stats.get("punctuation_density", 0.0)

        # Textos jornalísticos têm pontuação, mas excesso de !/? é sinal de tom
        # apelativo. O cálculo prioriza ! e ?, não vírgulas ou pontos comuns.
        score = (
            exclamation_density * 28
            + question_density * 16
            + max(0.0, punctuation_density - 0.16) * 2.5
        )
        return clamp(score)

    @staticmethod
    def _uppercase_excess_score(stats: dict) -> float:
        uppercase_ratio = stats.get("uppercase_ratio", 0.0)

        # Siglas elevam um pouco a razão de maiúsculas; por isso o limiar é
        # conservador. Acima de 12% começa a ficar incomum em notícia comum.
        return clamp((uppercase_ratio - 0.12) / 0.18)

    @staticmethod
    def _weak_attribution_score(
        *,
        weak_attribution_hits: int,
        source_attribution_hits: int,
        word_count: int,
    ) -> float:
        weak_score = clamp(weak_attribution_hits / 3)

        # Se há muitos verbos de atribuição ou referência a pesquisa/relatório,
        # reduzimos o risco de atribuição fraca.
        attribution_bonus = clamp(source_attribution_hits / max(word_count / 120, 1))
        return clamp(weak_score * (1 - 0.45 * attribution_bonus))

    @staticmethod
    def _factual_density_score(
        *,
        stats: dict,
        source_attribution_hits: int,
        data_terms_hits: int,
    ) -> float:
        word_count = max(stats.get("word_count", 0), 1)
        factual_items = (
            stats.get("numbers_count", 0)
            + stats.get("dates_count", 0)
            + source_attribution_hits
            + data_terms_hits
            + stats.get("quote_marks_count", 0) * 0.25
        )
        return clamp((factual_items / word_count) * 22)

    @staticmethod
    def _readability_score(stats: dict) -> float:
        """
        Score simples de legibilidade estrutural.

        Não é índice de legibilidade acadêmico; é um sinal prático para evitar
        textos extremamente fragmentados ou frases muito longas.
        """
        avg_sentence_words = stats.get("avg_sentence_words", 0.0)
        max_sentence_words = stats.get("max_sentence_words", 0)

        if avg_sentence_words <= 0:
            return 0.0

        # Faixa confortável aproximada para notícia: 12-32 palavras por sentença.
        if 12 <= avg_sentence_words <= 32:
            avg_score = 1.0
        elif avg_sentence_words < 12:
            avg_score = clamp(avg_sentence_words / 12)
        else:
            avg_score = clamp(1 - ((avg_sentence_words - 32) / 38))

        max_penalty = clamp((max_sentence_words - 65) / 55)
        return clamp(avg_score * (1 - 0.35 * max_penalty))

    # =========================================================================
    # Score e explicabilidade
    # =========================================================================

    @classmethod
    def _calculate_risk_score(cls, features: dict) -> float:
        weighted_sum = 0.0
        weight_sum = 0.0

        for key, weight in cls.RISK_WEIGHTS.items():
            if weight <= 0:
                continue
            weighted_sum += clamp(float(features.get(key, 0.0))) * weight
            weight_sum += weight

        if weight_sum <= 0:
            return 0.0

        base_risk = weighted_sum / weight_sum

        # Factual density e readability funcionam como atenuadores, não como
        # garantia de verdade. Um texto bem estruturado reduz apenas parte do
        # risco linguístico.
        factual_density = clamp(float(features.get("factual_density", 0.0)))
        readability = clamp(float(features.get("readability", 0.0)))
        mitigation = 0.18 * factual_density + 0.07 * readability

        return clamp(base_risk * (1 - mitigation))

    @classmethod
    def _label_from_risk(cls, risk_score: float) -> str:
        if risk_score >= cls.HIGH_FEATURE_THRESHOLD:
            return "alto risco linguístico"
        if risk_score >= cls.MEDIUM_FEATURE_THRESHOLD:
            return "atenção linguística"
        return "baixo risco linguístico"

    @classmethod
    def _build_flags(cls, *, features: dict, stats: dict, risk_score: float) -> list[str]:
        flags: list[str] = []

        if features.get("sensationalism", 0.0) >= cls.MEDIUM_FEATURE_THRESHOLD:
            flags.append("Foram encontrados sinais de linguagem sensacionalista.")

        if features.get("emotional_language", 0.0) >= cls.MEDIUM_FEATURE_THRESHOLD:
            flags.append("O texto apresenta vocabulário emocional acima do esperado.")

        if features.get("clickbait", 0.0) >= cls.MEDIUM_FEATURE_THRESHOLD:
            flags.append("Há indícios de chamada com tom de clickbait.")

        if features.get("punctuation_excess", 0.0) >= cls.MEDIUM_FEATURE_THRESHOLD:
            flags.append("O texto usa pontuação enfática em excesso.")

        if features.get("uppercase_excess", 0.0) >= cls.MEDIUM_FEATURE_THRESHOLD:
            flags.append("Há uso elevado de letras maiúsculas no texto.")

        if features.get("weak_attribution", 0.0) >= cls.MEDIUM_FEATURE_THRESHOLD:
            flags.append("Foram encontrados sinais de atribuição fraca ou não confirmada.")

        if features.get("factual_density", 0.0) >= 0.35:
            flags.append("O texto contém dados, datas, citações ou atribuições verificáveis.")

        if stats.get("word_count", 0) < 180:
            flags.append("O texto é curto; a análise linguística pode ser menos estável.")

        if not flags and risk_score < cls.MEDIUM_FEATURE_THRESHOLD:
            flags.append("Não foram encontrados sinais linguísticos fortes de alerta.")

        return flags

    # =========================================================================
    # Utilitários opcionais para debug/TCC
    # =========================================================================

    @classmethod
    def top_terms(cls, text: str, limit: int = 12) -> list[tuple[str, int]]:
        """Retorna termos frequentes do texto, útil para inspeção manual."""
        tokens = cls._content_tokens(text)
        counts = Counter(tokens)
        return counts.most_common(limit)
