"""
claim_detector.py

Pipeline: segmentation.py → claim_detector.py → RAG / similarity / stance / agregação

Fluxo híbrido:
  1. Heurísticas descartam o que claramente NÃO é claim
  2. Heurísticas confirmam o que claramente É claim (requer 2 sinais fortes OU 1 forte + verbo factual)
  3. LLM é chamado em batch (5-10 sentenças ambíguas por chamada)

Retorna ClaimResult por sentença, com List[Claim] embutida.
Cada Claim tem confidence individual.

Consome: Sentence (de segmentation.py)
Alimenta: RAG / similarity.py / stance_model.py / agregação / clustering
"""

from __future__ import annotations

import re
import json
import math
import time
import logging
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
    CAUSAL       = "causal"        # causa-efeito: "A lei causou aumento do desemprego"
    COMPARATIVE  = "comparative"   # comparação: "O Brasil tem mais X do que Y"
    DEFINITIONAL = "definitional"  # definição: "Segundo a OMS, saúde é..."
    PREDICTIVE   = "predictive"    # previsão: "A economia deve crescer 2%"
    QUOTED       = "quoted"        # citação atribuída: 'O ministro disse que "..."'
    OPINION      = "opinion"       # opinião atribuída verificável: "X disse que Y é ruim"
    AMBIGUOUS    = "ambiguous"     # LLM não classificou com certeza


class DetectionMethod(str, Enum):
    HEURISTIC_REJECT = "heuristic_reject"
    HEURISTIC_ACCEPT = "heuristic_accept"
    LLM_AMBIGUOUS    = "llm_ambiguous"


# =============================================================================
# Estruturas de dados
# =============================================================================

@dataclass
class Claim:
    """
    Uma afirmação verificável extraída de uma sentença.

    Uma sentença pode conter múltiplas claims com qualidades diferentes:
    "O PIB cresceu 3% e o ministro disse que a reforma é necessária."
    → Claim 1 (statistical, conf=0.92) + Claim 2 (quoted/opinion, conf=0.71)

    Atributos:
        confidence  — qualidade desta claim específica (independente do ClaimResult)
        subject     — SEMPRE preenchido; usado por agregação, clustering e RAG routing
        normalized  — query limpa para RAG (lowercase, sem stopwords, números normalizados)
        entities    — via spaCy NER quando disponível, fallback para regex
    """
    text:        str
    normalized:  str
    confidence:  float                    # confiança desta claim individual
    claim_type:  ClaimType = ClaimType.FACTUAL
    entities:    list[str] = field(default_factory=list)
    subject:     str       = ""           # SEMPRE preenchido — nunca None
    explanation: Optional[str] = None


@dataclass
class ClaimResult:
    """
    Resultado da análise de uma Sentence.

    confidence = confiança da decisão is_claim (booleana).
    Cada Claim dentro de claims tem sua própria confidence (qualidade da claim).
    """
    sentence:      object
    is_claim:      bool
    confidence:    float
    claims:        list[Claim]            = field(default_factory=list)
    method:        DetectionMethod        = DetectionMethod.HEURISTIC_REJECT
    reject_reason: Optional[str]          = None
    llm_raw:       Optional[str]          = None


# =============================================================================
# Sinais heurísticos com pesos separados
# =============================================================================

class HeuristicFilter:
    """
    Filtro baseado em sinais fortes e fracos com limiares explícitos.

    Regra de aceitação:
        score >= STRONG_THRESHOLD  →  accept
        score >= WEAK_THRESHOLD    →  ambiguous
        score <  WEAK_THRESHOLD    →  reject (se não tiver sinal de rejeição explícito)

    Onde score é a soma dos pesos dos sinais detectados.
    Sinais fortes valem 0.5, sinais fracos valem 0.15.
    Aceitação requer: 2 sinais fortes (≥1.0) OU 1 forte + verbo factual (≥0.7).
    """

    # Limiares
    STRONG_THRESHOLD = 0.70   # accept com confiança alta
    WEAK_THRESHOLD   = 0.30   # zona ambígua → LLM

    # ── Pesos ────────────────────────────────────────────────────────────────

    STRONG_SIGNALS = {
        "numeric":      0.50,  # número + unidade mensurável
        "attribution":  0.50,  # verbo de atribuição ("disse", "afirmou")
        "causal":       0.45,  # conectivo causal explícito
        "comparative":  0.45,  # estrutura comparativa
        "definition":   0.40,  # verbo definitório ("é definido como", "consiste em")
        "prediction":   0.40,  # verbo de previsão com dado ("deve crescer X%")
    }

    WEAK_SIGNALS = {
        "date":         0.20,  # data sem dado associado
        "proper_noun":  0.15,  # entidade nomeada (heurística)
        "factual_verb": 0.20,  # verbo factual sem atribuição ("ocorreu", "registrou")
        "negation":     0.15,  # negação de fato ("não houve", "nunca foi")
        "quantity":     0.15,  # quantidade sem unidade clara
    }

    # ── Padrões de rejeição explícita (alta confiança, sem score) ────────────

    _RE_QUESTION    = re.compile(r'\?$')
    _RE_IMPERATIVE  = re.compile(
        r'^(veja|leia|acesse|confira|saiba|clique|assista|ouça|baixe|'
        r'compartilhe|participe|vote|siga|curta|inscreva)[- ]',
        re.IGNORECASE
    )
    _RE_SOCIAL      = re.compile(
        r'^(bom dia|boa tarde|boa noite|olá|oi |obrigado|parabéns|feliz\b)',
        re.IGNORECASE
    )
    _RE_PURE_SUBJ   = re.compile(
        r'^(acho|acredito|penso|na minha opinião|ao meu ver)\b',
        re.IGNORECASE
    )
    MIN_CHARS = 25

    # ── Padrões de sinais fortes ─────────────────────────────────────────────

    _RE_NUMERIC = re.compile(
        r'\b\d+([.,]\d+)?\s*(%|mil|bilhões?|milhões?|kg|km|m²|ha|anos?|meses?'
        r'|dias?|pontos?|reais?|dólares?|euros?|habitantes?|mortos?|casos?|vagas?)\b',
        re.IGNORECASE
    )
    _RE_ATTRIBUTION = re.compile(
        r'\b(disse|afirmou|declarou|anunciou|confirmou|negou|alegou|'
        r'informou|comunicou|revelou|admitiu|garantiu|prometeu|assegurou)\b',
        re.IGNORECASE
    )
    _RE_CAUSAL = re.compile(
        r'\b(causou|gerou|resultou em|levou a|provocou|devido a|'
        r'por causa de|em consequência|como resultado de)\b',
        re.IGNORECASE
    )
    _RE_COMPARATIVE = re.compile(
        r'\b(maior|menor|mais|menos|superior|inferior|dobro|metade|'
        r'cresceu|caiu|aumentou|reduziu|superou|ultrapassou)\b.{0,30}'
        r'\b(que|do que|em relação|comparado)\b',
        re.IGNORECASE
        # FALSO POSITIVO CONHECIDO: "A empresa cresceu depois que..." dispara.
        # O lookahead .{0,30} é largo demais para distinguir comparação de
        # subordinação temporal. Soluções futuras (por custo crescente):
        #   1. Exigir objeto numérico: "cresceu X% (?:a mais )?(?:que|do que)"
        #   2. Dependency parse: checar se há arco "nsubj + advmod comparativo"
        # Hoje o falso positivo soma apenas 0.45 ao score — sozinho não aceita.
        # Aceitável enquanto o LLM filtra os ambíguos que esse sinal gera.
    )
    _RE_DEFINITION = re.compile(
        r'\b(é definido como|consiste em|refere-se a|significa|'
        r'corresponde a|trata-se de|é considerado)\b',
        re.IGNORECASE
    )
    _RE_PREDICTION = re.compile(
        r'\b(deve|deverá|vai|irá|prevê|projeta|estima).{0,40}\d',
        re.IGNORECASE
    )

    # ── Padrões de sinais fracos ─────────────────────────────────────────────

    _RE_DATE = re.compile(
        r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}|'
        r'janeiro|fevereiro|março|abril|maio|junho|'
        r'julho|agosto|setembro|outubro|novembro|dezembro)\b',
        re.IGNORECASE
    )
    _RE_PROPER_NOUN = re.compile(
        r'(?<=[.\s])([A-ZÁÉÍÓÚÂÊÔÃÕÜ][a-záéíóúâêôãõü]+'
        r'(?:\s[A-ZÁÉÍÓÚÂÊÔÃÕÜ][a-záéíóúâêôãõü]+)+)'
    )
    _RE_FACTUAL_VERB = re.compile(
        r'\b(ocorreu|aconteceu|registrou|atingiu|alcançou|'
        r'completou|iniciou|terminou|aprovou|rejeitou|sancionou)\b',
        re.IGNORECASE
    )
    _RE_NEGATION = re.compile(
        r'\b(não houve|nunca foi|jamais|não existe|não há|'
        r'não ocorreu|não registrou)\b',
        re.IGNORECASE
    )
    _RE_QUANTITY = re.compile(r'\b\d+\b')

    @classmethod
    def evaluate(cls, text: str) -> tuple[str, float, str, dict]:
        """
        Retorna (decisão, confiança, motivo, sinais_detectados).

        decisão: "reject" | "accept" | "ambiguous"
        sinais_detectados: dict com os sinais encontrados (para debug e explicabilidade)
        """
        stripped = text.strip()

        # ── Rejeições explícitas ─────────────────────────────────────────────
        if len(stripped) < cls.MIN_CHARS:
            return "reject", 0.95, "sentença muito curta", {}
        if cls._RE_QUESTION.search(stripped):
            return "reject", 0.98, "pergunta direta", {}
        if cls._RE_SOCIAL.match(stripped):
            return "reject", 0.97, "fórmula social", {}
        if cls._RE_IMPERATIVE.match(stripped):
            return "reject", 0.93, "frase imperativa", {}
        if cls._RE_PURE_SUBJ.match(stripped):
            return "reject", 0.90, "opinião subjetiva explícita (primeira pessoa)", {}

        # ── Coleta de sinais com pesos ───────────────────────────────────────
        signals: dict[str, float] = {}

        # sinais fortes
        if cls._RE_NUMERIC.search(stripped):
            signals["numeric"] = cls.STRONG_SIGNALS["numeric"]
        if cls._RE_ATTRIBUTION.search(stripped):
            signals["attribution"] = cls.STRONG_SIGNALS["attribution"]
        if cls._RE_CAUSAL.search(stripped):
            signals["causal"] = cls.STRONG_SIGNALS["causal"]
        if cls._RE_COMPARATIVE.search(stripped):
            signals["comparative"] = cls.STRONG_SIGNALS["comparative"]
        if cls._RE_DEFINITION.search(stripped):
            signals["definition"] = cls.STRONG_SIGNALS["definition"]
        if cls._RE_PREDICTION.search(stripped):
            signals["prediction"] = cls.STRONG_SIGNALS["prediction"]

        # sinais fracos
        if cls._RE_DATE.search(stripped):
            signals["date"] = cls.WEAK_SIGNALS["date"]
        if cls._RE_PROPER_NOUN.search(" " + stripped):
            signals["proper_noun"] = cls.WEAK_SIGNALS["proper_noun"]
        if cls._RE_FACTUAL_VERB.search(stripped):
            signals["factual_verb"] = cls.WEAK_SIGNALS["factual_verb"]
        if cls._RE_NEGATION.search(stripped):
            signals["negation"] = cls.WEAK_SIGNALS["negation"]
        if cls._RE_QUANTITY.search(stripped) and "numeric" not in signals:
            signals["quantity"] = cls.WEAK_SIGNALS["quantity"]

        score = sum(signals.values())

        if score >= cls.STRONG_THRESHOLD:
            # confiança escala com o score, teto em 0.93
            confidence = min(0.65 + score * 0.18, 0.93)
            reason = ", ".join(signals.keys())
            return "accept", confidence, reason, signals

        if score >= cls.WEAK_THRESHOLD:
            return "ambiguous", 0.5, f"score={score:.2f} — sinais insuficientes", signals

        return "reject", max(0.60, 0.85 - score), "sem sinais suficientes", signals


# =============================================================================
# Normalização para RAG
# =============================================================================

# Stopwords leves para português — palavras que poluem embeddings sem sentido.
#
# ATENÇÃO: manter esta lista MINIMALISTA.
# Stopwords demais destroem semântica importante:
#   "O governo não aprovou a lei" → sem "não" vira consulta oposta no RAG.
# Regra: só remover se a palavra NUNCA carrega carga semântica relevante.
# Negações ("não", "nunca", "jamais") são intencionalmente excluídas.
_PT_STOPWORDS = frozenset({
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos", "em", "na", "no", "nas", "nos",
    "por", "para", "com", "que", "se", "ao", "à", "às", "aos",
    "e", "ou", "mas", "também", "já", "ainda", "muito", "mais",
    "isso", "este", "esta", "esse", "essa", "aquele", "aquela",
    "sobre", "entre", "quando", "onde", "como", "ser", "ter",
    "foi", "são", "está", "estão", "era", "eram",
})

# Adjetivos vazios que não contribuem para busca semântica
_EMPTY_ADJECTIVES = re.compile(
    r'\b(grande|pequeno|importante|relevante|significativo|notável|'
    r'especial|específico|determinado|certo|possível|necessário)\b',
    re.IGNORECASE
)

# Normalização numérica: "3,5 milhões" → "3500000", "R$ 1.200" → "1200"
_RE_NUM_MILHOES = re.compile(r'(\d+(?:[.,]\d+)?)\s*milhões?', re.IGNORECASE)
_RE_NUM_BILHOES = re.compile(r'(\d+(?:[.,]\d+)?)\s*bilhões?', re.IGNORECASE)
_RE_NUM_MIL     = re.compile(r'(\d+(?:[.,]\d+)?)\s*mil\b',    re.IGNORECASE)
_RE_CURRENCY    = re.compile(r'R\$\s*')
_RE_PERCENT     = re.compile(r'(\d+(?:[.,]\d+)?)\s*%')


def normalize_for_rag(text: str) -> str:
    """
    Normalização de texto para query de RAG.

    Operações (em ordem):
      1. Lowercase
      2. Remove demonstrativos no início (Isso, Este, etc.)
      3. Normaliza números com unidade ("3 milhões" → "3000000")
      4. Remove símbolo de moeda
      5. Normaliza percentual ("5%" → "5 por cento")
      6. Remove adjetivos vazios
      7. Remove stopwords leves
      8. Colapsa espaços
    """
    t = text.strip()

    # 1. lowercase
    t = t.lower()

    # 2. remove demonstrativos no início
    t = re.sub(r'^(isso|este|esta|esse|essa|aquilo|aquele|aquela)\s+', '', t)

    # 3. normalização numérica
    t = _RE_NUM_BILHOES.sub(lambda m: str(int(float(m.group(1).replace(',', '.')) * 1_000_000_000)), t)
    t = _RE_NUM_MILHOES.sub(lambda m: str(int(float(m.group(1).replace(',', '.')) * 1_000_000)), t)
    t = _RE_NUM_MIL.sub(    lambda m: str(int(float(m.group(1).replace(',', '.')) * 1_000)), t)

    # 4. remove símbolo de moeda
    t = _RE_CURRENCY.sub('', t)

    # 5. percentual legível
    t = _RE_PERCENT.sub(r'\1 por cento', t)

    # 6. remove adjetivos vazios
    t = _EMPTY_ADJECTIVES.sub('', t)

    # 7. remove stopwords (tokenização simples por espaço)
    tokens = [tok for tok in t.split() if tok not in _PT_STOPWORDS and len(tok) > 1]
    t = ' '.join(tokens)

    # 8. colapsa espaços
    t = re.sub(r'\s{2,}', ' ', t).strip()

    return t


# =============================================================================
# Extração de subject heurística
# =============================================================================

# Substantivos-chave de domínio — usados para extrair subject
_DOMAIN_KEYWORDS = re.compile(
    r'\b(inflação|pib|desemprego|economia|saúde|educação|segurança|'
    r'crime|violência|eleição|governo|presidente|ministro|congresso|'
    r'senado|câmara|lei|decreto|reforma|orçamento|dívida|juros|'
    r'pandemia|vacina|covid|clima|ambiente|energia|petróleo|'
    r'exportação|importação|comércio|dólar|real|bolsa|mercado|'
    r'emprego|salário|renda|pobreza|desigualdade|mortalidade|'
    r'natalidade|população|censo|pesquisa|estudo|relatório)\b',
    re.IGNORECASE
)


def extract_subject(text: str, entities: list[str]) -> str:
    """
    Extrai o tópico central da sentença.

    Estratégia em cascata:
      1. Procura keyword de domínio no texto
      2. Se não encontrar, usa a primeira entidade nomeada
      3. Fallback: primeiras 3 palavras significativas do normalized

    Retorna sempre uma string não-vazia (nunca None).

    VIÉS DE CLUSTERING CONHECIDO:
      "O presidente falou sobre inflação" → subject="presidente" (step 2)
      quando o tópico real é "inflação" (step 1 deveria pegar, mas depende
      da keyword estar em _DOMAIN_KEYWORDS).
    Solução futura: noun chunk ranking por posição na frase + dependency parse
    (sujeito gramatical ≠ tópico semântico). TF-IDF local também ajudaria
    em corpora grandes. Por ora, ampliar _DOMAIN_KEYWORDS é o atalho mais
    barato para reduzir esse viés.
    """
    # 1. keyword de domínio
    match = _DOMAIN_KEYWORDS.search(text)
    if match:
        return match.group(1).lower()

    # 2. primeira entidade nomeada
    if entities:
        return entities[0].lower()

    # 3. fallback: primeiras palavras do normalized
    normalized = normalize_for_rag(text)
    words = normalized.split()
    subject_words = [w for w in words[:6] if len(w) > 3][:3]
    return ' '.join(subject_words) if subject_words else "geral"


# =============================================================================
# Extração de entidades
# =============================================================================

def extract_entities(text: str, nlp=None) -> list[str]:
    """
    Extrai entidades nomeadas do texto.

    Se spaCy (nlp) for fornecido, usa NER real (pt_core_news_lg ou sm).
    Caso contrário, fallback para regex de nomes próprios.

    O nlp deve ser carregado externamente e passado para evitar
    múltiplos carregamentos de modelo no pipeline:

        import spacy
        nlp = spacy.load("pt_core_news_lg")
        detector = ClaimDetector(api_key=..., nlp=nlp)
    """
    if nlp is not None:
        doc = nlp(text)
        entities = list({ent.text for ent in doc.ents
                         if ent.label_ in ("PER", "ORG", "GPE", "LOC", "DATE", "MONEY")})
        return entities

    # Fallback regex — mais falso positivo, mas melhor que nada.
    #
    # FALSO POSITIVO CONHECIDO: "Na Segunda Feira", "Em Janeiro" viram entidades.
    # TODO: adicionar blacklist de dias/meses e conectivos temporais:
    #   _NOUN_BLACKLIST = {"Segunda", "Terça", ..., "Janeiro", ..., "Na", "Em"}
    # Solução definitiva: POS tagging com spaCy (PROPN vs NOUN/ADV).
    # Enquanto nlp=None, aceitar essa imprecisão é intencional — o custo de
    # falso positivo em entidades é baixo (filtra mal no RAG, não inventa fato).
    matches = re.findall(
        r'(?<=[.\s])([A-ZÁÉÍÓÚÂÊÔÃÕÜ][a-záéíóúâêôãõü]+'
        r'(?:\s[A-ZÁÉÍÓÚÂÊÔÃÕÜ][a-záéíóúâêôãõü]+)+)',
        " " + text
    )
    seen, entities = set(), []
    for m in matches:
        if m not in seen:
            seen.add(m)
            entities.append(m)
    return entities


# =============================================================================
# Prompt LLM (batch)
# =============================================================================

LLM_SYSTEM_PROMPT = """Você é um detector de afirmações verificáveis para fact-checking jornalístico.

Receberá uma lista de sentenças numeradas. Para cada uma, retorne um objeto JSON.
Retorne SOMENTE um array JSON válido, sem markdown, sem texto fora do array.

Definição: afirmação verificável = declaração que pode ser confirmada ou refutada com evidências externas.
Opiniões puras, perguntas e saudações NÃO são verificáveis.

Esquema de cada elemento do array:
{
  "index": <número da sentença>,
  "is_claim": true | false,
  "confidence": 0.0 a 1.0,
  "reject_reason": "motivo se is_claim=false, senão null",
  "claims": [
    {
      "text": "trecho exato da claim",
      "normalized": "versão em lowercase sem stopwords para busca semântica",
      "confidence": 0.0 a 1.0,
      "claim_type": "factual|statistical|causal|comparative|definitional|predictive|quoted|opinion|ambiguous",
      "entities": ["entidade1", "entidade2"],
      "subject": "tópico central em 1-3 palavras (OBRIGATÓRIO, nunca null)",
      "explanation": "por que é verificável e como classificou"
    }
  ]
}

Regras:
- Uma sentença pode gerar múltiplas claims com confidences diferentes
- Se is_claim=false, claims deve ser []
- subject é OBRIGATÓRIO em toda Claim — nunca omita
- confidence em Claim reflete qualidade/clareza desta claim específica
- normalized deve ser útil para busca semântica (lowercase, sem artigos, números explícitos)"""


# =============================================================================
# ClaimDetector
# =============================================================================

class ClaimDetector:
    """
    Detector híbrido com batching de LLM.

    Arquitetura:
      - Heurísticas processam todas as sentenças (sem custo)
      - Ambíguas são acumuladas em buffer
      - Quando buffer atinge BATCH_SIZE (ou no flush final), 1 chamada LLM
        resolve N sentenças simultaneamente

    Uso:
        import spacy
        nlp = spacy.load("pt_core_news_lg")  # opcional mas recomendado

        detector = ClaimDetector(api_key="sk-ant-...", nlp=nlp)
        results  = detector.detect_batch(segmentation_output["sentences"])

    TODO — CACHE DE LLM (alta prioridade de custo):
        Adicionar cache por hash(sentence.text) antes de _resolve_batch_with_llm.
        Backend sugerido: sqlite (zero dependência) ou redis (produção).
        Reduz custo absurdamente em re-runs e artigos com sentenças repetidas.
        Esquema: {sha256(text): ClaimResult serializado, timestamp, model_version}

    TODO — RATE LIMITING:
        A API Anthropic tem limites de tokens/min e requests/min por tier.
        Adicionar token bucket ou semáforo antes de _call_llm_with_retry.
        Biblioteca sugerida: `limits` (PyPI) ou implementação manual com
        time.sleep proporcional ao tamanho do batch.
        Essencial antes de processar corpora grandes em produção.

    TODO — CLAIM SPLITTING (próximo grande passo):
        "O PIB cresceu 3% e o desemprego caiu 2%" → hoje gera 1 Claim.
        A estrutura List[Claim] já está preparada para múltiplas por sentença.
        Estratégias (por complexidade crescente):
          1. Regex de coordenação: split em " e ", " mas ", " porém " com heurística
          2. Dependency parse: detectar coordenação de VPs (spaCy conj arcs)
          3. LLM decomposition: pedir ao LLM para decompor no mesmo batch
        O LLM já faz isso parcialmente quando classifica — só falta expor no prompt.
    """

    BATCH_SIZE      = 8     # sentenças ambíguas por chamada LLM
    LLM_TIMEOUT     = 30.0  # segundos
    MAX_RETRIES     = 3
    RETRY_DELAY     = 2.0   # segundos (backoff simples)

    def __init__(
        self,
        api_key:  Optional[str] = None,
        nlp       = None,
        batch_size: int = BATCH_SIZE,
    ):
        self._api_key    = api_key
        self._nlp        = nlp
        self._batch_size = batch_size
        self._llm_available = api_key is not None

        if not self._llm_available:
            logger.warning(
                "ClaimDetector sem api_key — casos ambíguos marcados como AMBIGUOUS."
            )

    # =========================================================================
    # API pública
    # =========================================================================

    def detect(self, sentence) -> ClaimResult:
        """
        Analisa uma única Sentence sem overhead de pipeline batch.

        Para lotes, use detect_batch() — processa N sentenças com 1 chamada LLM
        para todos os ambíguos, enquanto detect() faria 1 chamada por ambíguo.
        """
        return self._detect_single(sentence)

    def _detect_single(self, sentence) -> ClaimResult:
        """
        Núcleo de detecção para uma sentença.

        Usado por detect() diretamente e por detect_batch() no loop de heurísticas,
        eliminando o overhead de criação de lista e indexação desnecessários.
        """
        decision, confidence, reason, signals = HeuristicFilter.evaluate(sentence.text)

        if decision == "reject":
            return ClaimResult(
                sentence      = sentence,
                is_claim      = False,
                confidence    = confidence,
                method        = DetectionMethod.HEURISTIC_REJECT,
                reject_reason = reason,
            )

        if decision == "accept":
            claim_type = self._claim_type_from_signals(signals)
            entities   = extract_entities(sentence.text, self._nlp)
            subject    = extract_subject(sentence.text, entities)
            claim = Claim(
                text        = sentence.text,
                normalized  = normalize_for_rag(sentence.text),
                confidence  = confidence,
                claim_type  = claim_type,
                entities    = entities,
                subject     = subject,
                explanation = f"Heurística: {reason}",
            )
            return ClaimResult(
                sentence   = sentence,
                is_claim   = True,
                confidence = confidence,
                claims     = [claim],
                method     = DetectionMethod.HEURISTIC_ACCEPT,
            )

        # ambiguous — resolve via LLM se disponível, senão placeholder
        if self._llm_available:
            batch = self._resolve_batch_with_llm([sentence])
            return batch[0]

        return ClaimResult(
            sentence      = sentence,
            is_claim      = False,
            confidence    = 0.5,
            method        = DetectionMethod.LLM_AMBIGUOUS,
            reject_reason = "LLM indisponível — caso ambíguo não resolvido",
        )

    def detect_batch(self, sentences: list) -> list[ClaimResult]:
        """
        Processa lista de Sentences com batching de LLM.

        Fluxo:
          1. Heurísticas classificam todas as sentenças
          2. Ambíguas são agrupadas em batches de BATCH_SIZE
          3. Cada batch → 1 chamada LLM
          4. Resultados LLM substituem os placeholders ambíguos
        """
        # Passo 1: heurísticas via _detect_single (sem overhead de pipeline batch).
        # Ambíguos ficam como None — resolvidos em batch no passo 2.
        results: list[ClaimResult | None] = [None] * len(sentences)
        ambiguous_indices: list[int] = []

        for i, sentence in enumerate(sentences):
            decision, _, _, _ = HeuristicFilter.evaluate(sentence.text)

            if decision == "ambiguous":
                ambiguous_indices.append(i)
                # placeholder — será preenchido no passo 2
            else:
                # reject ou accept: _detect_single resolve sem LLM
                results[i] = self._detect_single(sentence)

        # Passo 2: resolve ambíguos em batches
        if ambiguous_indices and self._llm_available:
            for batch_start in range(0, len(ambiguous_indices), self._batch_size):
                batch_idx = ambiguous_indices[batch_start : batch_start + self._batch_size]
                batch_sentences = [sentences[i] for i in batch_idx]

                llm_results = self._resolve_batch_with_llm(batch_sentences)

                for i, llm_result in zip(batch_idx, llm_results):
                    results[i] = llm_result

        # Passo 3: preenche ambíguos sem LLM (api_key ausente ou falha)
        for i in ambiguous_indices:
            if results[i] is None:
                results[i] = ClaimResult(
                    sentence      = sentences[i],
                    is_claim      = False,
                    confidence    = 0.5,
                    method        = DetectionMethod.LLM_AMBIGUOUS,
                    reject_reason = "LLM indisponível — caso ambíguo não resolvido",
                )

        heuristic_accepts  = sum(1 for r in results if r.method == DetectionMethod.HEURISTIC_ACCEPT)
        heuristic_rejects  = sum(1 for r in results if r.method == DetectionMethod.HEURISTIC_REJECT)
        llm_resolved       = sum(1 for r in results if r.method == DetectionMethod.LLM_AMBIGUOUS)
        llm_calls          = math.ceil(len(ambiguous_indices) / self._batch_size) if ambiguous_indices else 0

        logger.info(
            f"detect_batch: {len(sentences)} sentenças | "
            f"{heuristic_accepts} aceitas | {heuristic_rejects} rejeitadas | "
            f"{llm_resolved} via LLM ({llm_calls} chamadas, batch={self._batch_size})"
        )

        return results

    # =========================================================================
    # Resolução LLM em batch
    # =========================================================================

    def _resolve_batch_with_llm(self, sentences: list) -> list[ClaimResult]:
        """
        Resolve N sentenças ambíguas em 1 chamada LLM.

        Monta prompt numerado, parseia array JSON de volta,
        reconstrói ClaimResults na ordem original.
        """
        # monta input numerado
        numbered = "\n".join(
            f"{i+1}. {s.text}"
            for i, s in enumerate(sentences)
        )
        user_message = f"Analise estas {len(sentences)} sentenças:\n\n{numbered}"

        raw = self._call_llm_with_retry(user_message)
        if raw is None:
            # falha total → retorna ambíguos não resolvidos
            return [
                ClaimResult(
                    sentence      = s,
                    is_claim      = False,
                    confidence    = 0.4,
                    method        = DetectionMethod.LLM_AMBIGUOUS,
                    reject_reason = "Falha na chamada LLM (todos os retries esgotados)",
                )
                for s in sentences
            ]

        return self._parse_batch_response(sentences, raw)

    def _call_llm_with_retry(self, user_message: str) -> Optional[str]:
        """
        Chama a API com retry exponencial simples e timeout explícito.

        Retorna o texto da resposta ou None em caso de falha total.
        """
        try:
            import anthropic
        except ImportError:
            logger.error("anthropic não instalado — pip install anthropic")
            return None

        client = anthropic.Anthropic(api_key=self._api_key)

        for attempt in range(self.MAX_RETRIES):
            try:
                message = client.messages.create(
                    model      = "claude-sonnet-4-20250514",
                    max_tokens = 2048,
                    timeout    = self.LLM_TIMEOUT,
                    system     = LLM_SYSTEM_PROMPT,
                    messages   = [{"role": "user", "content": user_message}],
                )
                return message.content[0].text

            except Exception as exc:
                wait = self.RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    f"LLM tentativa {attempt+1}/{self.MAX_RETRIES} falhou: "
                    f"{type(exc).__name__}: {exc}. Aguardando {wait}s."
                )
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(wait)

        return None

    def _parse_batch_response(self, sentences: list, raw: str) -> list[ClaimResult]:
        """
        Parseia array JSON do LLM e mapeia de volta às sentenças originais.

        Robusto a: fences de markdown, índices faltando, JSON parcialmente inválido.
        """
        try:
            clean = re.sub(r'```(?:json)?', '', raw).strip()
            data  = json.loads(clean)
            if not isinstance(data, list):
                raise ValueError("Resposta LLM não é um array")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(f"Falha ao parsear batch LLM: {exc} | raw[:300]={raw[:300]}")
            return [
                ClaimResult(
                    sentence      = s,
                    is_claim      = False,
                    confidence    = 0.35,
                    method        = DetectionMethod.LLM_AMBIGUOUS,
                    reject_reason = f"JSON inválido: {exc}",
                    llm_raw       = raw,
                )
                for s in sentences
            ]

        # indexa por "index" (1-based) para tolerância a ordenação diferente
        by_index: dict[int, dict] = {item.get("index", i+1): item for i, item in enumerate(data)}

        results = []
        for i, sentence in enumerate(sentences):
            item = by_index.get(i + 1, {})

            claims = []
            for c in item.get("claims", []):
                entities = c.get("entities", [])
                subject  = c.get("subject") or extract_subject(sentence.text, entities)
                claims.append(Claim(
                    text        = c.get("text", sentence.text),
                    normalized  = c.get("normalized") or normalize_for_rag(sentence.text),
                    confidence  = float(c.get("confidence", 0.5)),
                    claim_type  = ClaimType(c["claim_type"])
                                  if c.get("claim_type") in ClaimType._value2member_map_
                                  else ClaimType.AMBIGUOUS,
                    entities    = entities,
                    subject     = subject,
                    explanation = c.get("explanation"),
                ))

            results.append(ClaimResult(
                sentence      = sentence,
                is_claim      = bool(item.get("is_claim", False)),
                confidence    = float(item.get("confidence", 0.5)),
                claims        = claims,
                method        = DetectionMethod.LLM_AMBIGUOUS,
                reject_reason = item.get("reject_reason"),
                llm_raw       = raw,
            ))

        return results

    # =========================================================================
    # Utilitário interno
    # =========================================================================

    @staticmethod
    def _claim_type_from_signals(signals: dict) -> ClaimType:
        """Infere o ClaimType pelo sinal de maior peso detectado."""
        priority = [
            ("attribution", ClaimType.QUOTED),
            ("causal",      ClaimType.CAUSAL),
            ("comparative", ClaimType.COMPARATIVE),
            ("numeric",     ClaimType.STATISTICAL),
            ("prediction",  ClaimType.PREDICTIVE),
            ("definition",  ClaimType.DEFINITIONAL),
        ]
        for signal_name, claim_type in priority:
            if signal_name in signals:
                return claim_type
        return ClaimType.FACTUAL


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
        "is_claim":      result.is_claim,
        "confidence":    round(result.confidence, 4),
        "method":        result.method.value,
        "reject_reason": result.reject_reason,
        "claims": [
            {
                "text":        c.text,
                "normalized":  c.normalized,
                "confidence":  round(c.confidence, 4),
                "claim_type":  c.claim_type.value,
                "entities":    c.entities,
                "subject":     c.subject,
                "explanation": c.explanation,
            }
            for c in result.claims
        ],
    }


def filter_verified_claims(results: list[ClaimResult]) -> list[ClaimResult]:
    """Retorna apenas ClaimResults com claims verificáveis."""
    return [r for r in results if r.is_claim and r.claims]