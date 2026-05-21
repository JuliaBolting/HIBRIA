"""
claim_detector.py

Pipeline: segmentation.py → claim_detector.py → RAG / similarity / stance / agregação

Fluxo híbrido:
  1. Heurísticas descartam o que claramente NÃO é claim (rápido, sem custo de API)
  2. Heurísticas confirmam o que claramente É claim (≥2 sinais fortes OU 1 forte + verbo factual)
  3. LLM é chamado em batch para casos ambíguos (5-10 sentenças por chamada)

Mudanças desta versão:
  - Pesos separados: strong_signals / weak_signals com limiar mais rígido
  - subject extraído heuristicamente (sempre presente)
  - confidence por Claim individual
  - normalized robusto: lowercase + stopwords + normalização numérica
  - NER via spaCy (fallback regex se modelo indisponível)
  - Batching de ambíguos: N sentenças → 1 chamada LLM
  - Timeout explícito + retry com backoff
  - Rate limiting via token bucket
  - ClaimType.OPINION adicionado
  - Ambiguity thresholds efetivamente usados no roteamento
"""

from __future__ import annotations

import re
import json
import time
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ClaimType(str, Enum):
    FACTUAL      = "factual"       # fato objetivo: "A inflação foi de 5,2%"
    STATISTICAL  = "statistical"   # dado numérico: "3 mil pessoas morreram"
    CAUSAL       = "causal"        # causa-efeito: "A lei causou desemprego"
    COMPARATIVE  = "comparative"   # comparação:   "Brasil tem mais X que Y"
    DEFINITIONAL = "definitional"  # definição:    "Segundo a OMS, saúde é..."
    PREDICTIVE   = "predictive"    # previsão:     "Economia deve crescer 2%"
    QUOTED       = "quoted"        # citação:      'Ministro disse que "..."'
    OPINION      = "opinion"       # opinião identificável mas não verificável
    AMBIGUOUS    = "ambiguous"     # LLM não conseguiu classificar


class DetectionMethod(str, Enum):
    HEURISTIC_REJECT = "heuristic_reject"
    HEURISTIC_ACCEPT = "heuristic_accept"
    LLM_BATCH        = "llm_batch"          # resolvido em batch
    LLM_UNAVAILABLE  = "llm_unavailable"    # ambíguo sem LLM disponível


# =============================================================================
# Estruturas de dados
# =============================================================================

@dataclass
class Claim:
    """
    Afirmação verificável extraída de uma sentença.

    confidence: qualidade desta claim específica — permite que agregação
                pondere claims individuais de uma mesma sentença de forma
                diferente (ex: "PIB cresceu 3% e tudo melhorou" → 1ª claim
                tem confidence 0.9, 2ª tem 0.4).
    subject:    sempre presente — essencial para aggregation, clustering
                e RAG routing. Extraído heuristicamente ou pelo LLM.
    normalized: query robusta para RAG (lowercase, sem stopwords, números
                normalizados, sem adjetivos vazios).
    """
    text:        str
    normalized:  str
    claim_type:  ClaimType        = ClaimType.FACTUAL
    confidence:  float            = 0.7
    entities:    list[str]        = field(default_factory=list)
    subject:     str              = ""        # nunca None — sempre extraído
    explanation: Optional[str]    = None


@dataclass
class ClaimResult:
    """
    Resultado da análise de uma Sentence.

    confidence: confiança na classificação is_claim (não na veracidade).
    claims:     lista de Claims individuais com seus próprios scores.
    """
    sentence:     object
    is_claim:     bool
    confidence:   float
    claims:       list[Claim]       = field(default_factory=list)
    method:       DetectionMethod   = DetectionMethod.HEURISTIC_REJECT
    reject_reason: Optional[str]    = None
    llm_raw:      Optional[str]     = None


# =============================================================================
# Normalização para RAG
# =============================================================================

# Stopwords leves do português (artigos, preposições, pronomes demonstrativos)
# Intencionalmente reduzida — remover demais prejudica embeddings contextuais.
_PT_STOPWORDS = frozenset({
    "o", "a", "os", "as", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "ao", "à", "aos", "às", "pelo", "pela", "pelos", "pelas",
    "por", "para", "com", "sem", "sob", "sobre", "entre",
    "que", "se", "já", "mais", "muito", "também", "ainda",
    "isso", "este", "esta", "esse", "essa", "aquele", "aquela",
    "isto", "aquilo", "seu", "sua", "seus", "suas",
    "foi", "é", "são", "eram", "será", "serão",   # verbos cópula vazios
})

# Adjetivos vagos que poluem embeddings sem adicionar especificidade semântica
_VAGUE_ADJECTIVES = frozenset({
    "grande", "pequeno", "importante", "significativo", "relevante",
    "novo", "novo", "novos", "novas", "bom", "boa", "bons", "boas",
    "ruim", "mau", "má", "péssimo", "ótimo", "excelente",
    "interessante", "especial", "específico", "geral", "atual",
})

# Padrão para normalizar números: qualquer sequência numérica → <NUM>
# Preserva a informação de que há um número sem fixar o valor específico,
# melhorando recall em buscas semânticas ("cresceu 3%" ~ "cresceu 5%").
_RE_NUMBER = re.compile(r'\b\d+([.,]\d+)?(%|mil|bi|mi)?\b')


def normalize_for_rag(text: str) -> str:
    """
    Normalização robusta de texto para query de RAG.

    Pipeline:
      1. Lowercase
      2. Normalização numérica (3,5% → <NUM>)
      3. Remoção de pontuação (mantém hífens compostos)
      4. Tokenização simples
      5. Remoção de stopwords PT
      6. Remoção de adjetivos vagos
      7. Colapso de espaços
    """
    normalized = text.lower().strip()
    normalized = _RE_NUMBER.sub("<NUM>", normalized)
    normalized = re.sub(r"[^\w\s<>-]", " ", normalized)   # remove pontuação
    tokens = normalized.split()
    tokens = [t for t in tokens if t not in _PT_STOPWORDS]
    tokens = [t for t in tokens if t not in _VAGUE_ADJECTIVES]
    return " ".join(tokens)


# =============================================================================
# Extração de subject
# =============================================================================

# Substantivos-âncora frequentes em fact-checking — usados como fallback
# quando spaCy não está disponível ou não encontra entidade principal.
_SUBJECT_ANCHORS = re.compile(
    r'\b(governo|presidente|ministro|ministério|congresso|senado|câmara'
    r'|stf|banco central|ibge|ipca|pib|desemprego|inflação|economia'
    r'|saúde|educação|segurança|meio ambiente|orçamento|déficit|superávit'
    r'|covid|pandemia|vacina|eleição|partido|candidato|prefeito|governador)\b',
    re.IGNORECASE
)


def extract_subject(text: str, entities: list[str]) -> str:
    """
    Extrai o subject principal da sentença para aggregation / RAG routing.

    Estratégia em cascata:
      1. Primeira entidade nomeada (pessoa, org, local) — mais específica
      2. Substantivo-âncora do domínio fact-checking
      3. Primeira palavra com ≥5 chars que não seja stopword (fallback seguro)
      4. Primeiras 3 palavras normalizadas (último recurso)

    Retorna sempre uma string não-vazia.
    """
    # 1. Entidade nomeada
    if entities:
        return entities[0].lower()

    # 2. Âncora de domínio
    anchor_match = _SUBJECT_ANCHORS.search(text)
    if anchor_match:
        return anchor_match.group(0).lower()

    # 3. Primeira palavra substantiva longa
    for word in text.split():
        clean = re.sub(r'[^\w]', '', word).lower()
        if len(clean) >= 5 and clean not in _PT_STOPWORDS and clean not in _VAGUE_ADJECTIVES:
            return clean

    # 4. Fallback: primeiras 3 palavras normalizadas
    normalized = normalize_for_rag(text)
    return " ".join(normalized.split()[:3]) or text[:30].lower()


# =============================================================================
# NER — spaCy com fallback regex
# =============================================================================

def _load_spacy():
    """
    Tenta carregar o modelo spaCy pt_core_news_sm.
    Retorna o modelo ou None se não disponível.
    """
    try:
        import spacy
        return spacy.load("pt_core_news_sm")
    except Exception:
        logger.warning(
            "spaCy pt_core_news_sm não disponível — usando NER heurístico. "
            "Instale com: pip install spacy && python -m spacy download pt_core_news_sm"
        )
        return None


# Singleton: carregado uma vez por processo
_SPACY_NLP = None
_SPACY_LOADED = False


def _get_spacy():
    global _SPACY_NLP, _SPACY_LOADED
    if not _SPACY_LOADED:
        _SPACY_NLP = _load_spacy()
        _SPACY_LOADED = True
    return _SPACY_NLP


_RE_PROPER_NOUN_FALLBACK = re.compile(
    r'(?<=[.!?\s])([A-ZÁÉÍÓÚÂÊÔÃÕÜ][a-záéíóúâêôãõü]+(?:\s[A-ZÁÉÍÓÚÂÊÔÃÕÜ][a-záéíóúâêôãõü]+)+)'
)


def extract_entities(text: str) -> list[str]:
    """
    Extrai entidades nomeadas via spaCy (PER, ORG, LOC, GPE, DATE, PERCENT).
    Fallback para regex de nomes próprios se spaCy indisponível.
    """
    nlp = _get_spacy()
    if nlp is not None:
        doc = nlp(text)
        seen, entities = set(), []
        for ent in doc.ents:
            if ent.label_ in {"PER", "ORG", "LOC", "GPE", "DATE", "PERCENT", "MONEY"}:
                key = ent.text.strip()
                if key not in seen and len(key) > 2:
                    seen.add(key)
                    entities.append(key)
        return entities

    # fallback regex — menor precisão, mas evita crash
    matches = re.findall(_RE_PROPER_NOUN_FALLBACK, " " + text)
    seen, entities = set(), []
    for m in matches:
        if m not in seen:
            seen.add(m)
            entities.append(m)
    return entities


# =============================================================================
# HeuristicFilter — pesos separados, limiar mais rígido
# =============================================================================

class HeuristicFilter:
    """
    Filtros rápidos com pesos calibrados.

    Regra de aceitação (mais rígida que v1):
      ≥ 2 sinais fortes
      OU 1 sinal forte + verbo factual

    Isso reduz drasticamente falsos positivos em sentenças com apenas
    um número solto ou uma data sem contexto verificável.
    """

    MIN_CHARS = 25

    # ── Padrões de rejeição ──────────────────────────────────────────────────

    _RE_QUESTION   = re.compile(r'\?$')
    _RE_IMPERATIVE = re.compile(
        r'^(veja|leia|acesse|confira|saiba|clique|assista|ouça|baixe'
        r'|compartilhe|participe|vote|siga|curta|inscreva)[- ]',
        re.IGNORECASE
    )
    _RE_SOCIAL = re.compile(
        r'^(bom dia|boa tarde|boa noite|olá|oi |obrigado|parabéns|feliz\b)',
        re.IGNORECASE
    )
    # Opiniões sem dado embutido
    _RE_PURE_OPINION = re.compile(
        r'^\s*(acho|acredito|penso|na minha opinião|ao meu ver)\b',
        re.IGNORECASE
    )

    # ── Sinais FORTES (cada um vale 1 ponto forte) ───────────────────────────
    # Exige especificidade: número + unidade (não número solto)

    _RE_NUMERIC_STRONG = re.compile(
        r'\b\d+([.,]\d+)?\s*(%|mil|bilhões?|milhões?|kg|km|m²|ha'
        r'|anos?|meses?|dias?|pontos?|reais?|dólares?|euros?|vagas?'
        r'|mortes?|casos?|votos?|parlamentares?)\b',
        re.IGNORECASE
    )
    _RE_DATE_STRONG = re.compile(
        r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{1,2}\s+de\s+\w+\s+de\s+\d{4})\b',
        re.IGNORECASE
    )
    _RE_ATTRIBUTION = re.compile(
        r'\b(disse|afirmou|declarou|anunciou|confirmou|negou|alegou'
        r'|informou|revelou|admitiu|garantiu|prometeu|assegurou)\b',
        re.IGNORECASE
    )
    _RE_CAUSAL = re.compile(
        r'\b(causou|gerou|resultou em|levou a|provocou|em consequência'
        r'|como resultado|devido a|por causa de)\b',
        re.IGNORECASE
    )
    _RE_COMPARATIVE = re.compile(
        r'\b(maior que|menor que|mais (?:do )?que|menos (?:do )?que'
        r'|superior a|inferior a|acima de|abaixo de|supera|superam)\b',
        re.IGNORECASE
    )
    _RE_OFFICIAL_SOURCE = re.compile(
        r'\b(segundo o ibge|segundo o ipea|segundo a oms|segundo o banco central'
        r'|de acordo com|conforme dados?|pesquisa (?:do|da|de)|levantamento)\b',
        re.IGNORECASE
    )

    # ── Sinais FRACOS (peso 0.5 — sozinhos não bastam) ───────────────────────
    # Ano solto, nome próprio, verbo de estado genérico

    _RE_YEAR_WEAK = re.compile(r'\b(19|20)\d{2}\b')
    _RE_PROPER_WEAK = re.compile(
        r'\b[A-ZÁÉÍÓÚ][a-záéíóú]{3,}(?:\s[A-ZÁÉÍÓÚ][a-záéíóú]{3,})+\b'
    )

    # ── Verbo factual (bônus que combina com 1 sinal forte) ─────────────────
    _RE_FACTUAL_VERB = re.compile(
        r'\b(aumentou|diminuiu|cresceu|caiu|subiu|recuou|atingiu|alcançou'
        r'|registrou|totalizou|ultrapassou|aprovou|rejeitou|sancionou'
        r'|vetou|assinou|publicou|lançou|inaugurou|encerrou|iniciou)\b',
        re.IGNORECASE
    )

    @classmethod
    def evaluate(cls, text: str) -> tuple[str, float, str]:
        """
        Retorna (decisão, confiança, motivo).
        decisão: "reject" | "accept" | "ambiguous"
        """
        stripped = text.strip()

        # ── Rejeições duras ──────────────────────────────────────────────────
        if len(stripped) < cls.MIN_CHARS:
            return "reject", 0.95, "sentença muito curta"
        if cls._RE_QUESTION.search(stripped):
            return "reject", 0.98, "pergunta direta"
        if cls._RE_SOCIAL.match(stripped):
            return "reject", 0.97, "fórmula social"
        if cls._RE_IMPERATIVE.match(stripped):
            return "reject", 0.93, "frase imperativa"
        if cls._RE_PURE_OPINION.match(stripped):
            return "reject", 0.88, "opinião subjetiva explícita"

        # ── Contagem de sinais ───────────────────────────────────────────────
        strong = 0.0
        detected_type = ClaimType.FACTUAL

        if cls._RE_NUMERIC_STRONG.search(stripped):
            strong += 1.0
            detected_type = ClaimType.STATISTICAL
        if cls._RE_DATE_STRONG.search(stripped):
            strong += 1.0
        if cls._RE_ATTRIBUTION.search(stripped):
            strong += 1.0
            detected_type = ClaimType.QUOTED
        if cls._RE_CAUSAL.search(stripped):
            strong += 1.0
            detected_type = ClaimType.CAUSAL
        if cls._RE_COMPARATIVE.search(stripped):
            strong += 1.0
            detected_type = ClaimType.COMPARATIVE
        if cls._RE_OFFICIAL_SOURCE.search(stripped):
            strong += 1.0

        weak = 0.0
        if cls._RE_YEAR_WEAK.search(stripped):
            weak += 0.5
        if cls._RE_PROPER_WEAK.search(stripped):
            weak += 0.5

        has_factual_verb = bool(cls._RE_FACTUAL_VERB.search(stripped))

        total = strong + weak * 0.4   # weak contribui apenas parcialmente

        # ── Regra de aceitação (mais rígida) ─────────────────────────────────
        # Aceita se: ≥2 sinais fortes OU (1 sinal forte + verbo factual)
        if strong >= 2.0:
            confidence = min(0.65 + strong * 0.08, 0.93)
            return "accept", confidence, detected_type.value

        if strong >= 1.0 and has_factual_verb:
            confidence = min(0.60 + total * 0.10, 0.88)
            return "accept", confidence, detected_type.value

        # ── Zona ambígua ─────────────────────────────────────────────────────
        # 1 sinal forte sem verbo factual, ou apenas sinais fracos
        if strong >= 1.0 or total >= 0.8:
            return "ambiguous", 0.5, detected_type.value

        return "reject", 0.75, "sem sinais verificáveis suficientes"


# =============================================================================
# Rate limiter (token bucket simples)
# =============================================================================

class _TokenBucket:
    """
    Rate limiter thread-safe por token bucket.

    Parâmetros padrão conservadores para a API Anthropic:
      rate=10 req/s, capacity=20 tokens.
    """
    def __init__(self, rate: float = 10.0, capacity: float = 20.0):
        self._rate     = rate
        self._capacity = capacity
        self._tokens   = capacity
        self._last     = time.monotonic()
        self._lock     = threading.Lock()

    def acquire(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity,
                    self._tokens + (now - self._last) * self._rate
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            wait = 1.0 / self._rate
            if time.monotonic() + wait > deadline:
                return False
            time.sleep(wait)


# =============================================================================
# Prompt do LLM (batch)
# =============================================================================

LLM_SYSTEM_PROMPT = """Você é um detector de afirmações verificáveis para fact-checking jornalístico.

Receberá uma lista de sentenças numeradas. Analise cada uma e retorne SOMENTE um array JSON válido, sem markdown, sem texto fora do JSON.

Definição de afirmação verificável: declaração que pode ser confirmada ou refutada com evidências externas. Opiniões puras, perguntas e saudações NÃO são verificáveis.

Esquema — array com um objeto por sentença, na mesma ordem:
[
  {
    "index": 0,
    "is_claim": true | false,
    "confidence": 0.0 a 1.0,
    "reject_reason": "motivo se is_claim=false, senão null",
    "claims": [
      {
        "text": "trecho exato",
        "normalized": "query para RAG: lowercase, sem stopwords, números como <NUM>",
        "claim_type": "factual|statistical|causal|comparative|definitional|predictive|quoted|opinion|ambiguous",
        "confidence": 0.0 a 1.0,
        "entities": ["entidade1", "entidade2"],
        "subject": "tópico central em 1-3 palavras (OBRIGATÓRIO)",
        "explanation": "por que é verificável e como classificou"
      }
    ]
  }
]

Regras:
- Retorne exatamente um objeto por sentença de entrada, na mesma ordem
- subject é OBRIGATÓRIO em toda Claim — use o tema principal (ex: "inflação", "Lula", "orçamento federal")
- Uma sentença pode gerar múltiplas claims
- Se is_claim=false, claims deve ser []
- normalized deve ser útil como query semântica: sem artigos, preposições, demonstrativos; números → <NUM>"""


# =============================================================================
# ClaimDetector
# =============================================================================

class ClaimDetector:
    """
    Detector híbrido com batching de ambíguos, retry e rate limiting.

    Uso típico:
        detector = ClaimDetector(api_key="sk-ant-...")
        results  = detector.detect_batch(segmentation_output["sentences"])

    Parâmetros de batching:
        batch_size   — sentenças ambíguas por chamada LLM (padrão: 8)
        max_retries  — tentativas em caso de erro de rede (padrão: 3)
        timeout      — timeout por chamada em segundos (padrão: 30)
    """

    # Limites da zona ambígua — usados efetivamente no roteamento
    AMBIGUITY_LOW  = 0.40   # abaixo → reject direto
    AMBIGUITY_HIGH = 0.65   # acima → accept direto

    def __init__(
        self,
        api_key:     Optional[str] = None,
        batch_size:  int           = 8,
        max_retries: int           = 3,
        timeout:     float         = 30.0,
    ):
        self._api_key    = api_key
        self._available  = api_key is not None
        self._batch_size = batch_size
        self._max_retries= max_retries
        self._timeout    = timeout
        self._rate_limiter = _TokenBucket()

        if not self._available:
            logger.warning(
                "ClaimDetector sem api_key — ambíguos marcados como LLM_UNAVAILABLE."
            )

    # =========================================================================
    # API pública
    # =========================================================================

    def detect(self, sentence) -> ClaimResult:
        """Analisa uma única Sentence. Para batches, prefira detect_batch."""
        return self.detect_batch([sentence])[0]

    def detect_batch(self, sentences: list) -> list[ClaimResult]:
        """
        Processa lista de Sentences com batching inteligente de ambíguos.

        Fluxo:
          1. Heurística classifica cada sentença
          2. Ambíguos são acumulados em lotes de batch_size
          3. Cada lote → 1 chamada LLM
          4. Resultados são remontados na ordem original
        """
        results: list[Optional[ClaimResult]] = [None] * len(sentences)
        ambiguous_indices: list[int]         = []

        # ── Passo 1: heurística ──────────────────────────────────────────────
        for i, sentence in enumerate(sentences):
            decision, confidence, reason = HeuristicFilter.evaluate(sentence.text)

            if decision == "reject":
                results[i] = self._make_reject(sentence, confidence, reason,
                                               DetectionMethod.HEURISTIC_REJECT)

            elif decision == "accept":
                results[i] = self._make_accept_heuristic(sentence, confidence, reason)

            else:  # ambiguous
                # aplica os thresholds: confiança extrema resolve sem LLM
                if confidence < self.AMBIGUITY_LOW:
                    results[i] = self._make_reject(sentence, confidence,
                                                   "abaixo do limiar de ambiguidade",
                                                   DetectionMethod.HEURISTIC_REJECT)
                elif confidence > self.AMBIGUITY_HIGH:
                    results[i] = self._make_accept_heuristic(sentence, confidence, reason)
                else:
                    ambiguous_indices.append(i)

        # ── Passo 2: resolver ambíguos em batch ──────────────────────────────
        for batch_start in range(0, len(ambiguous_indices), self._batch_size):
            batch_idx = ambiguous_indices[batch_start: batch_start + self._batch_size]
            batch_sentences = [sentences[i] for i in batch_idx]
            batch_results   = self._resolve_batch_llm(batch_sentences)
            for i, result in zip(batch_idx, batch_results):
                results[i] = result

        # ── Log ──────────────────────────────────────────────────────────────
        n_claims = sum(1 for r in results if r and r.is_claim)
        n_llm    = sum(1 for r in results if r and r.method == DetectionMethod.LLM_BATCH)
        n_calls  = (len(ambiguous_indices) + self._batch_size - 1) // self._batch_size if ambiguous_indices else 0
        logger.info(
            f"detect_batch: {len(sentences)} sentenças | "
            f"{n_claims} claims | "
            f"{len(ambiguous_indices)} ambíguos → {n_calls} chamadas LLM"
        )

        return [r for r in results if r is not None]

    # =========================================================================
    # Construção de resultados heurísticos
    # =========================================================================

    def _make_reject(
        self, sentence, confidence: float, reason: str, method: DetectionMethod
    ) -> ClaimResult:
        return ClaimResult(
            sentence      = sentence,
            is_claim      = False,
            confidence    = confidence,
            method        = method,
            reject_reason = reason,
        )

    def _make_accept_heuristic(
        self, sentence, confidence: float, type_str: str
    ) -> ClaimResult:
        claim_type = (
            ClaimType(type_str)
            if type_str in ClaimType._value2member_map_
            else ClaimType.FACTUAL
        )
        entities = extract_entities(sentence.text)
        subject  = extract_subject(sentence.text, entities)
        claim = Claim(
            text        = sentence.text,
            normalized  = normalize_for_rag(sentence.text),
            claim_type  = claim_type,
            confidence  = confidence,
            entities    = entities,
            subject     = subject,
            explanation = f"Detectado por heurística: {type_str}",
        )
        return ClaimResult(
            sentence   = sentence,
            is_claim   = True,
            confidence = confidence,
            claims     = [claim],
            method     = DetectionMethod.HEURISTIC_ACCEPT,
        )

    # =========================================================================
    # Batch LLM com retry e timeout
    # =========================================================================

    def _resolve_batch_llm(self, sentences: list) -> list[ClaimResult]:
        """
        Envia um lote de sentenças ambíguas para o LLM em uma única chamada.
        Retry com backoff exponencial. Timeout explícito.
        """
        if not self._available:
            return [
                self._make_reject(s, 0.5, "LLM indisponível",
                                  DetectionMethod.LLM_UNAVAILABLE)
                for s in sentences
            ]

        # monta prompt com sentenças numeradas
        numbered = "\n".join(
            f"[{i}] {s.text}" for i, s in enumerate(sentences)
        )

        raw = self._call_llm_with_retry(numbered)
        if raw is None:
            # fallback conservador: todas viram rejeição com confiança baixa
            return [
                self._make_reject(s, 0.35, "falha persistente na API LLM",
                                  DetectionMethod.LLM_BATCH)
                for s in sentences
            ]

        return self._parse_batch_response(sentences, raw)

    def _call_llm_with_retry(self, prompt: str) -> Optional[str]:
        """
        Chama a API com retry exponencial e timeout.
        Retorna o texto da resposta ou None após esgotar tentativas.
        """
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        delay  = 1.0

        for attempt in range(1, self._max_retries + 1):
            if not self._rate_limiter.acquire(timeout=self._timeout):
                logger.warning("Rate limiter: timeout aguardando token")
                return None
            try:
                message = client.messages.create(
                    model      = "claude-sonnet-4-20250514",
                    max_tokens = 2048,
                    timeout    = self._timeout,
                    system     = LLM_SYSTEM_PROMPT,
                    messages   = [{"role": "user", "content": prompt}],
                )
                return message.content[0].text

            except anthropic.APITimeoutError:
                logger.warning(f"LLM timeout (tentativa {attempt}/{self._max_retries})")
            except anthropic.RateLimitError:
                logger.warning(f"Rate limit da API (tentativa {attempt}/{self._max_retries})")
                delay *= 2   # backoff mais agressivo para rate limit
            except anthropic.APIError as exc:
                logger.warning(f"Erro de API (tentativa {attempt}/{self._max_retries}): {exc}")

            if attempt < self._max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 30.0)   # backoff exponencial com teto

        return None

    def _parse_batch_response(self, sentences: list, raw: str) -> list[ClaimResult]:
        """
        Parseia array JSON retornado pelo LLM e constrói ClaimResults.
        Fallback por sentença em caso de índice ausente ou parse error.
        """
        try:
            clean = re.sub(r'```(?:json)?', '', raw).strip()
            data  = json.loads(clean)
            if not isinstance(data, list):
                raise ValueError("Esperado array JSON no topo")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(f"Parse LLM batch falhou: {exc} | raw={raw[:300]}")
            return [
                self._make_reject(s, 0.35, f"JSON inválido: {exc}",
                                  DetectionMethod.LLM_BATCH)
                for s in sentences
            ]

        # indexa por "index" do LLM para tolerar reordenação
        by_index = {item.get("index", i): item for i, item in enumerate(data)}
        results  = []

        for i, sentence in enumerate(sentences):
            item = by_index.get(i)
            if item is None:
                logger.warning(f"LLM não retornou resultado para sentença index={i}")
                results.append(
                    self._make_reject(sentence, 0.4, "sem resultado do LLM",
                                      DetectionMethod.LLM_BATCH)
                )
                continue

            try:
                claims = []
                for c in item.get("claims", []):
                    ct = c.get("claim_type", "factual")
                    entities = c.get("entities", [])
                    claims.append(Claim(
                        text        = c.get("text", sentence.text),
                        normalized  = c.get("normalized") or normalize_for_rag(sentence.text),
                        claim_type  = ClaimType(ct) if ct in ClaimType._value2member_map_ else ClaimType.AMBIGUOUS,
                        confidence  = float(c.get("confidence", 0.6)),
                        entities    = entities,
                        subject     = c.get("subject") or extract_subject(sentence.text, entities),
                        explanation = c.get("explanation"),
                    ))

                results.append(ClaimResult(
                    sentence      = sentence,
                    is_claim      = bool(item.get("is_claim", False)),
                    confidence    = float(item.get("confidence", 0.5)),
                    claims        = claims,
                    method        = DetectionMethod.LLM_BATCH,
                    reject_reason = item.get("reject_reason"),
                    llm_raw       = raw,
                ))

            except Exception as exc:
                logger.warning(f"Erro ao parsear item LLM index={i}: {exc}")
                results.append(
                    self._make_reject(sentence, 0.35, f"erro de parse: {exc}",
                                      DetectionMethod.LLM_BATCH)
                )

        return results


# =============================================================================
# Serialização e helpers para pipeline.py
# =============================================================================

def claim_result_to_dict(result: ClaimResult) -> dict:
    """Serializa ClaimResult para dict JSON-compatível."""
    return {
        "sentence": {
            "text":        result.sentence.text,
            "block_index": result.sentence.block_index,
            "sent_index":  result.sentence.sent_index,
            "char_start":  getattr(result.sentence, "char_start", 0),
            "char_end":    getattr(result.sentence, "char_end", 0),
        },
        "is_claim":     result.is_claim,
        "confidence":   round(result.confidence, 4),
        "method":       result.method.value,
        "reject_reason": result.reject_reason,
        "claims": [
            {
                "text":        c.text,
                "normalized":  c.normalized,
                "claim_type":  c.claim_type.value,
                "confidence":  round(c.confidence, 4),
                "entities":    c.entities,
                "subject":     c.subject,
                "explanation": c.explanation,
            }
            for c in result.claims
        ],
    }


def filter_verified_claims(results: list[ClaimResult]) -> list[ClaimResult]:
    """Filtra apenas resultados com claims verificáveis."""
    return [r for r in results if r.is_claim and r.claims]


def claims_by_subject(results: list[ClaimResult]) -> dict[str, list[Claim]]:
    """
    Agrupa Claims por subject — útil para aggregation e RAG routing.

    Retorna dict {subject: [Claim, ...]} ordenado por número de claims.
    """
    groups: dict[str, list[Claim]] = {}
    for result in filter_verified_claims(results):
        for claim in result.claims:
            groups.setdefault(claim.subject, []).append(claim)
    return dict(sorted(groups.items(), key=lambda x: -len(x[1])))