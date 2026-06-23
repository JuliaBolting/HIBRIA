# =============================================================================
# claim_detector.py
# Etapa de identificação de afirmações verificáveis (claims).
#
# Recebe sentenças do segmentation.py e classifica cada uma em:
#   - CHECKABLE  : afirmação factual verificável → passa para o retriever
#   - OPINION    : julgamento subjetivo          → descartada
#   - RHETORICAL : pergunta ou retórica          → descartada
#   - NOISE      : fragmento sem conteúdo        → descartada
#
# Pipeline interno por sentença:
#   1. Filtros rápidos (regex)        — descarta ruído óbvio sem custo
#   2. Análise linguística (spaCy)    — POS, NER, estrutura verbal
#   3. Heurísticas de checkabilidade  — regras do domínio jornalístico
#   4. Score de confiança             — 0.0 a 1.0
#
# Entrada:  list[Sentence]  — saída do segmentation.py
# Saída:    list[Claim]     — entrada do retriever.py
# =============================================================================

from __future__ import annotations

import re
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enums e estruturas
# =============================================================================

class ClaimType(str, Enum):
    CHECKABLE   = "checkable"    # afirmação factual verificável
    OPINION     = "opinion"      # julgamento subjetivo
    RHETORICAL  = "rhetorical"   # pergunta ou retórica
    NOISE       = "noise"        # fragmento sem conteúdo útil


@dataclass
class Claim:
    """
    Afirmação verificável extraída de uma sentença.

    Campos consumidos pelo retriever.py:
        normalized  — texto sem stopwords e lematizado (query para busca)
        entities    — entidades nomeadas (queries específicas para Wikipedia)
        subject     — sujeito principal (foco da verificação)

    Campos para rastreabilidade:
        sentence_text  — sentença original
        block_index    — bloco de origem
        sent_index     — posição na lista de sentenças
        confidence     — confiança da classificação [0.0, 1.0]
    """
    text:          str                        # texto original da sentença
    normalized:    str                        # normalizado para busca
    entities:      list[str] = field(default_factory=list)
    entity_types:  dict[str, str] = field(default_factory=dict)  # entidade → tipo (PER, ORG, LOC...)
    subject:       str = ""                   # sujeito principal
    predicate:     str = ""                   # predicado principal
    claim_id:      str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    claim_type:    ClaimType = ClaimType.CHECKABLE
    confidence:    float = 0.0                # confiança da classificação
    block_index:   int = 0
    sent_index:    int = 0
    has_numbers:   bool = False               # contém dados numéricos verificáveis
    has_entities:  bool = False               # contém entidades nomeadas
    keywords:      list[str] = field(default_factory=list)  # termos relevantes para busca


class ClassificationResult(NamedTuple):
    """Resultado intermediário da classificação de uma sentença."""
    claim_type: ClaimType
    confidence: float
    reason:     str           # motivo — útil para debug e TCC


# =============================================================================
# ClaimDetector
# =============================================================================

class ClaimDetector:

    # -------------------------------------------------------------------------
    # Padrões de descarte rápido (filtros pré-NLP)
    #
    # Ordenados do mais barato ao mais caro computacionalmente.
    # Evitam passar sentenças triviais pelo pipeline do spaCy.
    # -------------------------------------------------------------------------

    # perguntas retóricas e diretas
    _QUESTION_RE = re.compile(
        r"^\s*[\"«]?.{0,10}(quem|o que|quando|onde|como|por que|qual|será|poderia|"
        r"deveria|pode|existe|há|tem)[^.!?]{0,80}[?]\s*$",
        re.IGNORECASE,
    )

    # fragmentos curtos que não formam proposição
    _FRAGMENT_RE = re.compile(
        r"^\s*(veja|leia|saiba|clique|acesse|confira|assine|siga|ouça|"
        r"compartilhe|comente|curta|baixe|instale|atualiz)[a-z]*\b",
        re.IGNORECASE,
    )

    # marcadores de opinião explícita
    _OPINION_MARKERS_RE = re.compile(
        r"\b(acho|acha|acredito|acredita|penso|pensa|na minha|na minha opinião|"
        r"em minha|parece|ao meu ver|ao nosso ver|é possível que|talvez|"
        r"provavelmente|aparentemente|supostamente|seria|poderia ser|"
        r"pode ser que|é provável|improvável)\b",
        re.IGNORECASE,
    )

    # verbos de citação — sentença é relato, não afirmação direta
    # o claim_detector mantém relatos pois eles podem ser verificáveis
    # ("X afirmou que Y" → Y pode ser checado), mas marca para o retriever
    _CITATION_VERBS_RE = re.compile(
        r"\b(afirmou|disse|declarou|informou|relatou|explicou|destacou|"
        r"ressaltou|apontou|alegou|garante|nega|confirmou|desmentiu|"
        r"revelou|admitiu|reconheceu|anunciou|prometeu|defendeu)\b",
        re.IGNORECASE,
    )

    # indicadores de conteúdo factual verificável
    _FACTUAL_INDICATORS_RE = re.compile(
        r"\b(\d+[%]|\d+[\.,]\d+|R\$\s*\d|US\$\s*\d|"          # números, percentuais, valores
        r"segundo\s+\w+|de\s+acordo\s+com|com\s+base\s+em|"     # atribuições factuais
        r"estudo|pesquisa|levantamento|relatório|dado|índice|"   # fontes de dados
        r"lei\s+n[°º]|decreto|portaria|resolução|medida|"        # documentos oficiais
        r"aprovado|sancionado|publicado|registrado|comprovado|"  # verbos de fato consumado
        r"inaugurado|lançado|eleito|nomeado|demitido|preso|"     # eventos verificáveis
        r"morreu|nasceu|fundado|criado|extinto)\b",
        re.IGNORECASE,
    )

    # padrões numéricos que aumentam checkabilidade
    _NUMBER_RE = re.compile(
        r"\b\d+([.,]\d+)?(%|mil|milhão|milhões|bilhão|bilhões|"
        r"metros?|km|kg|ton|anos?|meses?|dias?|horas?)?\b"
    )

    # -------------------------------------------------------------------------
    # Stopwords para normalização da query de busca
    # Complementam as do spaCy com termos jornalísticos sem valor semântico
    # -------------------------------------------------------------------------
    _QUERY_STOPWORDS = {
        "o", "a", "os", "as", "um", "uma", "uns", "umas",
        "de", "do", "da", "dos", "das", "em", "no", "na",
        "nos", "nas", "por", "para", "com", "sem", "sob",
        "que", "se", "não", "mas", "e", "ou", "já", "ainda",
        "mais", "menos", "muito", "pouco", "todo", "toda",
        "segundo", "afirmou", "disse", "declarou", "informou",
        "também", "além", "disso", "isso", "aquilo", "este",
        "esta", "esse", "essa", "esses", "essas", "eles", "elas",
    }

    # -------------------------------------------------------------------------
    # Tipos de entidade spaCy que indicam conteúdo verificável
    # -------------------------------------------------------------------------
    _CHECKABLE_ENTITY_TYPES = {
        "PER",   # pessoa
        "ORG",   # organização
        "LOC",   # localização
        "GPE",   # entidade geopolítica (país, cidade, estado)
        "DATE",  # data
        "MONEY", # valor monetário
        "PERCENT", # percentual
        "CARDINAL", # número cardinal
    }

    # =========================================================================
    # Carregamento do spaCy (compartilhado com segmentation e normalization)
    # =========================================================================

    _nlp = None

    @classmethod
    def _get_nlp(cls):
        """
        Carrega o modelo spaCy com NER ativo.
        Diferente do segmentation.py, aqui o NER é necessário para
        extrair entidades verificáveis (pessoas, organizações, locais).
        O parser continua desabilitado — não precisamos de dependências sintáticas.
        """
        if cls._nlp is None:
            try:
                import spacy
                # habilita NER — essencial para extrair entidades verificáveis
                cls._nlp = spacy.load(
                    "pt_core_news_sm",
                    disable=["parser"],
                )
                cls._nlp.max_length = 2_000_000

                # sentencizer necessário pois desabilitamos o parser
                if "sentencizer" not in cls._nlp.pipe_names:
                    cls._nlp.add_pipe("sentencizer")

            except OSError:
                raise ImportError(
                    "Modelo spaCy não encontrado. Execute:\n"
                    "  python -m spacy download pt_core_news_sm"
                )
        return cls._nlp

    # =========================================================================
    # Filtros rápidos pré-NLP
    # =========================================================================

    @classmethod
    def _quick_filter(cls, text: str) -> ClassificationResult | None:
        """
        Aplica filtros baratos antes do pipeline NLP.
        Retorna ClassificationResult se a sentença pode ser descartada
        imediatamente, ou None se precisa de análise completa.
        """
        stripped = text.strip()

        # muito curta para ser uma proposição
        if len(stripped) < 25:
            return ClassificationResult(
                ClaimType.NOISE, 1.0, "sentença muito curta"
            )

        # pergunta direta ou retórica
        if stripped.endswith("?") or cls._QUESTION_RE.search(stripped):
            return ClassificationResult(
                ClaimType.RHETORICAL, 0.95, "padrão de pergunta detectado"
            )

        # call-to-action / fragmento de UI
        if cls._FRAGMENT_RE.match(stripped):
            return ClassificationResult(
                ClaimType.NOISE, 0.95, "fragmento de UI ou call-to-action"
            )

        # opinião explícita forte
        opinion_match = cls._OPINION_MARKERS_RE.search(stripped)
        if opinion_match:
            # ainda pode ser verificável se tem indicadores factuais fortes
            factual = cls._FACTUAL_INDICATORS_RE.search(stripped)
            if not factual:
                return ClassificationResult(
                    ClaimType.OPINION,
                    0.85,
                    f"marcador de opinião: '{opinion_match.group()}'",
                )

        return None  # precisa de análise NLP completa

    # =========================================================================
    # Análise linguística via spaCy
    # =========================================================================

    @classmethod
    def _analyze_with_spacy(cls, text: str) -> dict:
        """
        Extrai informações linguísticas relevantes para checkabilidade:
          - Entidades nomeadas (PER, ORG, LOC, DATE, MONEY...)
          - Sujeito e predicado principal (via POS tagging)
          - Presença de verbos no indicativo (proposição factual)
          - Presença de verbos modais (opinião/hipótese)

        Retorna um dicionário com os sinais extraídos.
        """
        nlp = cls._get_nlp()
        doc = nlp(text)

        # ── entidades nomeadas ────────────────────────────────────────────────
        entities     = []
        entity_types = {}
        for ent in doc.ents:
            if ent.label_ in cls._CHECKABLE_ENTITY_TYPES:
                entities.append(ent.text)
                entity_types[ent.text] = ent.label_

        # ── análise de tokens ─────────────────────────────────────────────────
        verbs_indicative = []   # verbos no indicativo — proposição factual
        verbs_modal      = []   # verbos modais — hipótese/opinião
        nouns            = []   # substantivos — candidatos a sujeito/predicado
        subject          = ""
        predicate        = ""

        for token in doc:
            # verbos no indicativo (POS=VERB, modo indicativo)
            if token.pos_ == "VERB":
                morph = token.morph.to_dict()
                mood  = morph.get("Mood", "")
                if mood == "Ind":
                    verbs_indicative.append(token.lemma_)
                elif mood == "Cnd" or token.lemma_ in (
                    "poder", "dever", "querer", "precisar", "parecer"
                ):
                    verbs_modal.append(token.lemma_)

            # substantivos para identificar sujeito/predicado
            elif token.pos_ in ("NOUN", "PROPN"):
                nouns.append(token.text)

        # heurística simples de sujeito: primeiro PROPN ou NER PER/ORG/GPE
        for ent in doc.ents:
            if ent.label_ in ("PER", "ORG", "GPE", "LOC"):
                subject = ent.text
                break
        if not subject and nouns:
            subject = nouns[0]

        # predicado: primeiro verbo no indicativo ou o principal
        predicate = verbs_indicative[0] if verbs_indicative else ""

        return {
            "entities":          entities,
            "entity_types":      entity_types,
            "verbs_indicative":  verbs_indicative,
            "verbs_modal":       verbs_modal,
            "has_entities":      len(entities) > 0,
            "has_numbers":       bool(cls._NUMBER_RE.search(text)),
            "has_citation":      bool(cls._CITATION_VERBS_RE.search(text)),
            "subject":           subject,
            "predicate":         predicate,
            "token_count":       sum(1 for t in doc if not t.is_punct and not t.is_space),
        }

    # =========================================================================
    # Score de checkabilidade
    # =========================================================================

    @classmethod
    def _checkability_score(cls, text: str, nlp_data: dict) -> float:
        """
        Calcula um score de checkabilidade [0.0, 1.0] combinando sinais.

        Sinais positivos (aumentam score):
          + entidades nomeadas verificáveis
          + números, percentuais, valores monetários
          + verbos no indicativo (proposição factual)
          + indicadores factuais explícitos (pesquisa, estudo, lei)
          + verbos de citação (relato atribuível)

        Sinais negativos (diminuem score):
          - verbos modais (poder, dever, parecer)
          - marcadores de opinião
          - ausência de sujeito identificável
        """
        score = 0.0

        # sinais positivos
        if nlp_data["has_entities"]:
            # mais entidades = mais específico = mais verificável
            score += min(0.30, len(nlp_data["entities"]) * 0.10)

        if nlp_data["has_numbers"]:
            score += 0.20

        if nlp_data["verbs_indicative"]:
            score += 0.20

        if cls._FACTUAL_INDICATORS_RE.search(text):
            score += 0.20

        if nlp_data["has_citation"]:
            # relato atribuível: "X disse que Y" — Y é verificável
            score += 0.15

        if nlp_data["subject"]:
            score += 0.10

        # sinais negativos
        if nlp_data["verbs_modal"]:
            score -= 0.15

        if cls._OPINION_MARKERS_RE.search(text):
            score -= 0.20

        return round(max(0.0, min(1.0, score)), 3)

    # =========================================================================
    # Normalização para query de busca
    # =========================================================================

    @classmethod
    def _normalize_for_query(cls, text: str, nlp_data: dict) -> str:
        """
        Produz versão normalizada do claim para uso como query no retriever.

        Estratégia: prioriza entidades + termos factuais + números.
        Remove stopwords mas MANTÉM entidades nomeadas intactas
        (remover partes de "Banco Central do Brasil" destrói a query).
        """
        # protege entidades nomeadas da remoção de stopwords
        protected = set()
        for ent in nlp_data["entities"]:
            for word in ent.lower().split():
                protected.add(word)

        tokens = re.findall(r"\b\w{2,}\b", text.lower())
        filtered = [
            t for t in tokens
            if t not in cls._QUERY_STOPWORDS or t in protected
        ]

        # se ficou muito curto, usa o texto completo sem stopwords
        if len(filtered) < 3:
            filtered = [t for t in tokens if len(t) > 2]

        return " ".join(filtered)

    # =========================================================================
    # Classificação completa de uma sentença
    # =========================================================================

    @classmethod
    def _classify(cls, text: str) -> tuple[ClassificationResult, dict]:
        """
        Classificação completa em dois estágios:
          1. Filtro rápido (regex)   — descarta casos óbvios
          2. Análise NLP (spaCy)     — classifica os casos ambíguos

        Retorna (ClassificationResult, nlp_data).
        nlp_data é {} quando o filtro rápido já descartou a sentença.
        """
        # estágio 1: filtro rápido
        quick = cls._quick_filter(text)
        if quick is not None:
            return quick, {}

        # estágio 2: análise NLP completa
        nlp_data = cls._analyze_with_spacy(text)
        score    = cls._checkability_score(text, nlp_data)

        # threshold: score >= 0.25 → checkable
        # abaixo disso mas com entidades → checkable com baixa confiança
        # sem entidades e score baixo → opinião ou ruído
        if score >= 0.25:
            return ClassificationResult(
                ClaimType.CHECKABLE,
                score,
                f"score={score} — entidades={nlp_data['entities'][:3]}",
            ), nlp_data

        elif score >= 0.10 and nlp_data["has_entities"]:
            return ClassificationResult(
                ClaimType.CHECKABLE,
                score,
                f"score baixo mas tem entidades: {nlp_data['entities'][:2]}",
            ), nlp_data

        elif nlp_data["verbs_modal"] and not nlp_data["verbs_indicative"]:
            return ClassificationResult(
                ClaimType.OPINION,
                1.0 - score,
                "só verbos modais, sem indicativo",
            ), nlp_data

        else:
            return ClassificationResult(
                ClaimType.NOISE,
                0.8,
                f"score insuficiente ({score}) sem entidades",
            ), nlp_data

    # =========================================================================
    # Interface pública
    # =========================================================================

    @classmethod
    def detect(
        cls,
        sentences: list,            # list[Sentence] do segmentation.py
        min_confidence: float = 0.15,
        max_claims: int = 50,
    ) -> dict:
        """
        Processa lista de sentenças e extrai claims verificáveis.

        Args:
            sentences:       list[Sentence] do segmentation.py
            min_confidence:  threshold mínimo para aceitar um claim
            max_claims:      limite máximo de claims por artigo
                             evita sobrecarga no retriever em artigos longos

        Retorno:
            claims         — list[Claim] ordenados por confiança decrescente
            discarded      — contagem por tipo descartado (para debug/TCC)
            stats          — métricas do processo
        """
        claims:    list[Claim] = []
        discarded: dict[str, int] = {
            ClaimType.OPINION:    0,
            ClaimType.RHETORICAL: 0,
            ClaimType.NOISE:      0,
        }

        for sentence in sentences:
            text = getattr(sentence, "text", str(sentence))

            classification, nlp_data = cls._classify(text)

            # descarta não-checkables
            if classification.claim_type != ClaimType.CHECKABLE:
                discarded[classification.claim_type] += 1
                logger.debug(
                    f"[claim_detector] descartado ({classification.claim_type.value}): "
                    f"{text[:60]}... — {classification.reason}"
                )
                continue

            # abaixo do threshold de confiança
            if classification.confidence < min_confidence:
                discarded[ClaimType.NOISE] += 1
                continue

            # extrai keywords para busca (top termos informativos)
            keywords = [
                w for w in re.findall(r"\b\w{4,}\b", text.lower())
                if w not in cls._QUERY_STOPWORDS
            ][:8]

            claim = Claim(
                text         = text,
                normalized   = cls._normalize_for_query(text, nlp_data),
                entities     = nlp_data.get("entities", []),
                entity_types = nlp_data.get("entity_types", {}),
                subject      = nlp_data.get("subject", ""),
                predicate    = nlp_data.get("predicate", ""),
                claim_type   = ClaimType.CHECKABLE,
                confidence   = classification.confidence,
                block_index  = getattr(sentence, "block_index", 0),
                sent_index   = getattr(sentence, "sent_index", 0),
                has_numbers  = nlp_data.get("has_numbers", False),
                has_entities = nlp_data.get("has_entities", False),
                keywords     = keywords,
            )
            claims.append(claim)

            if len(claims) >= max_claims:
                logger.info(
                    f"[claim_detector] limite de {max_claims} claims atingido"
                )
                break

        # ordena por confiança decrescente — retriever processa os mais fortes primeiro
        claims.sort(key=lambda c: c.confidence, reverse=True)

        total = len(claims) + sum(discarded.values())
        stats = {
            "sentences_processed": total,
            "claims_found":        len(claims),
            "discarded_opinion":   discarded[ClaimType.OPINION],
            "discarded_rhetorical": discarded[ClaimType.RHETORICAL],
            "discarded_noise":     discarded[ClaimType.NOISE],
            "checkability_rate":   round(len(claims) / total, 3) if total else 0.0,
            "avg_confidence":      (
                round(sum(c.confidence for c in claims) / len(claims), 3)
                if claims else 0.0
            ),
            "claims_with_entities":  sum(1 for c in claims if c.has_entities),
            "claims_with_numbers":   sum(1 for c in claims if c.has_numbers),
        }

        return {
            "claims":    claims,
            "discarded": discarded,
            "stats":     stats,
        }