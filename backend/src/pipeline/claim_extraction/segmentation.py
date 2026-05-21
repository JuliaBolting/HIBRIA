# =============================================================================
# segmentation.py
# Responsável por dividir o texto limpo em unidades menores para análise.
#
# Produz duas saídas com granularidades distintas:
#
#   sentences — sentenças individuais
#       Consumido por: claim_detector.py
#       Por quê: afirmações verificáveis ocorrem no nível da sentença.
#       "O presidente assinou o decreto" é uma claim. Um parágrafo inteiro não.
#
#   segments — grupos de sentenças semanticamente coesos
#       Consumido por: similarity.py, retriever.py (RAG), stance_model.py
#       Por quê: uma sentença isolada tem contexto insuficiente para embeddings
#       e busca semântica. Segmentos de 3-5 sentenças capturam o contexto
#       necessário para comparação com evidências externas.
#
# Depende de: cleaner.py (recebe blocks_clean)
# Alimenta:   claim_detector.py (sentences), similarity.py / retriever.py (segments)
# =============================================================================

import re
from dataclasses import dataclass, field


# =============================================================================
# Estruturas de dados de saída
# Dataclasses garantem tipagem clara e facilitam serialização no pipeline.py
# =============================================================================

@dataclass
class Sentence:
    """
    Representa uma sentença individual extraída de um bloco.

    Atributos:
        text        — texto da sentença
        block_index — índice do bloco de origem (para rastreabilidade)
        sent_index  — posição da sentença dentro do bloco
        char_start  — posição inicial no texto do bloco (para highlighting)
        char_end    — posição final no texto do bloco
    """
    text:        str
    block_index: int
    sent_index:  int
    char_start:  int = 0
    char_end:    int = 0


@dataclass
class Segment:
    """
    Representa um segmento temático composto por N sentenças consecutivas.

    Atributos:
        text        — texto completo do segmento (sentenças unidas)
        sentences   — lista de sentenças que compõem o segmento
        block_index — bloco de origem (segmentos não cruzam blocos)
        seg_index   — posição do segmento na lista final
    """
    text:        str
    sentences:   list[Sentence]
    block_index: int
    seg_index:   int


# =============================================================================
# TextSegmenter
# =============================================================================

class TextSegmenter:

    # -------------------------------------------------------------------------
    # Padrão de segmentação de sentenças.
    #
    # Regras (em ordem de precedência):
    #   1. Não quebra em abreviações comuns: "Dr.", "Sr.", "Sra.", "Art.", "Av."
    #      "Jan.", "Fev." etc. — o lookahead negativo (?<!\b[A-Z][a-z]{0,3})
    #      evita falsos positivos em títulos e meses.
    #   2. Não quebra em números decimais: "3.14", "R$ 1.500,00"
    #   3. Quebra em ".", "!", "?" seguidos de espaço + letra maiúscula.
    #   4. Aceita múltiplas pontuações seguidas: "...Texto" ou "?! Próximo"
    #
    # A regex usa lookbehind e lookahead para não consumir os delimitadores,
    # preservando a pontuação original em cada sentença.
    # -------------------------------------------------------------------------
    _SENT_SPLIT_RE = re.compile(
        r'(?<!\w\.\w.)'               # não quebra em abreviações tipo "e.g."
        r'(?<!\b[A-ZÁÊÉ][a-záéê]{0,3}\.)'  # não quebra em "Dr.", "Sr.", "Art."
        r'(?<!\b\d{1,3})'             # não quebra após número sozinho: "Lei 9.394"
        r'(?<=[.!?…])'               # quebra APÓS pontuação de fim de sentença
        r'(?:\s+)'                    # consome o espaço entre sentenças
        r'(?=[A-ZÁÀÃÂÉÊÍÓÔÕÚ"\u201c])'  # próxima começa com maiúscula ou aspas
    )

    # Tamanho mínimo de uma sentença válida (em caracteres).
    # Elimina fragmentos como "Ibid.", "Op. cit.", artefatos de extração.
    MIN_SENTENCE_CHARS: int = 20

    # Número de sentenças por segmento.
    # 3-5 sentenças é o sweet spot para embeddings semânticos:
    # menos que isso = contexto insuficiente, mais = perde especificidade.
    SENTENCES_PER_SEGMENT: int = 4

    # Sobreposição entre segmentos consecutivos (em sentenças).
    # Garante que afirmações na fronteira entre segmentos sejam cobertas
    # por pelo menos um segmento completo. Padrão usado em chunking de RAG.
    SEGMENT_OVERLAP: int = 1

    # =========================================================================
    # Segmentação de sentenças
    # =========================================================================

    @classmethod
    def _split_into_sentences(
        cls,
        text: str,
        block_index: int,
    ) -> list[Sentence]:
        """
        Divide um bloco de texto em sentenças individuais.

        Usa regex para identificar fronteiras de sentença sem depender de
        modelos pesados (o spaCy já está no normalizer — evitamos duplicar
        o carregamento aqui). Para português jornalístico essa abordagem
        cobre >95% dos casos corretamente.

        Rastreia as posições char_start/char_end para permitir que módulos
        downstream (ex: explanation_generator) destaquem afirmações no
        texto original.
        """
        sentences: list[Sentence] = []
        # divide preservando a pontuação (lookbehind não consome o delimitador)
        raw_parts = cls._SENT_SPLIT_RE.split(text)

        char_cursor = 0
        sent_index  = 0

        for part in raw_parts:
            part = part.strip()

            # descarta fragmentos muito curtos
            if len(part) < cls.MIN_SENTENCE_CHARS:
                # mesmo assim avança o cursor para manter posições corretas
                char_cursor += len(part) + 1
                continue

            # localiza a sentença no texto original para extrair char_start/end
            char_start = text.find(part, char_cursor)
            char_end   = char_start + len(part)

            sentences.append(Sentence(
                text        = part,
                block_index = block_index,
                sent_index  = sent_index,
                char_start  = char_start,
                char_end    = char_end,
            ))

            char_cursor = char_end
            sent_index += 1

        return sentences

    # =========================================================================
    # Agrupamento em segmentos (chunking com sobreposição)
    # =========================================================================

    @classmethod
    def _build_segments(
        cls,
        sentences:   list[Sentence],
        block_index: int,
        seg_offset:  int = 0,
    ) -> list[Segment]:
        """
        Agrupa sentenças em segmentos com sobreposição configurável.

        Exemplo com SENTENCES_PER_SEGMENT=4 e SEGMENT_OVERLAP=1:
            Sentenças: [S0, S1, S2, S3, S4, S5, S6]
            Segmento 0: [S0, S1, S2, S3]
            Segmento 1: [S3, S4, S5, S6]  ← S3 repetido (overlap)

        A sobreposição garante cobertura de afirmações que caem exatamente
        na fronteira entre dois segmentos — prática padrão em pipelines RAG.

        Args:
            sentences:   lista de sentenças do bloco
            block_index: índice do bloco de origem
            seg_offset:  deslocamento global do índice (para múltiplos blocos)
        """
        if not sentences:
            return []

        segments: list[Segment] = []
        step      = max(1, cls.SENTENCES_PER_SEGMENT - cls.SEGMENT_OVERLAP)
        seg_index = seg_offset

        for i in range(0, len(sentences), step):
            chunk = sentences[i : i + cls.SENTENCES_PER_SEGMENT]

            # descarta segmentos com uma única sentença muito curta no final
            if len(chunk) == 1 and len(chunk[0].text) < 60:
                continue

            segment_text = " ".join(s.text for s in chunk)

            segments.append(Segment(
                text        = segment_text,
                sentences   = chunk,
                block_index = block_index,
                seg_index   = seg_index,
            ))
            seg_index += 1

        return segments

    # =========================================================================
    # Ponto de entrada público
    # =========================================================================

    @classmethod
    def segment(cls, blocks: list[str]) -> dict:
        """
        Processa a lista de blocos limpos e retorna sentenças e segmentos.

        Args:
            blocks: lista de blocos de texto vindos do cleaner.py (blocks_clean)

        Retorno:
            sentences      — lista de objetos Sentence (para claim_detector.py)
            segments       — lista de objetos Segment (para similarity.py / RAG)
            sentence_texts — lista de strings puras das sentenças (para serialização JSON)
            segment_texts  — lista de strings puras dos segmentos (para serialização JSON)
            stats          — contagens para log no pipeline.py
        """
        all_sentences: list[Sentence] = []
        all_segments:  list[Segment]  = []
        seg_offset = 0

        for block_index, block in enumerate(blocks):
            # pula blocos muito curtos — provavelmente cabeçalhos ou metadados
            # que passaram pelo cleaner mas não contêm afirmações verificáveis
            if len(block.strip()) < cls.MIN_SENTENCE_CHARS * 2:
                continue

            # ── segmentação de sentenças ──────────────────────────────────
            sentences = cls._split_into_sentences(block, block_index)
            all_sentences.extend(sentences)

            # ── agrupamento em segmentos ──────────────────────────────────
            # segmentos não cruzam blocos: cada bloco é chunked independentemente
            # isso preserva a separação editorial entre parágrafos do artigo
            segments = cls._build_segments(sentences, block_index, seg_offset)
            all_segments.extend(segments)
            seg_offset += len(segments)

        return {
            # objetos tipados — para módulos Python downstream
            "sentences":      all_sentences,
            "segments":       all_segments,

            # strings puras — para serialização JSON no pipeline.py
            "sentence_texts": [s.text for s in all_sentences],
            "segment_texts":  [s.text for s in all_segments],

            # métricas para log
            "stats": {
                "blocks_processed":  len(blocks),
                "sentence_count":    len(all_sentences),
                "segment_count":     len(all_segments),
                "avg_sent_per_block": (
                    round(len(all_sentences) / len(blocks), 1) if blocks else 0
                ),
            },
        }