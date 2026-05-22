# =============================================================================
# segmentation.py
# Divide o texto limpo em unidades menores para os módulos downstream.
#
# Produz duas saídas com granularidades distintas:
#
#   sentences — sentenças individuais
#       → claim_detector.py
#       Afirmações verificáveis ocorrem no nível da sentença.
#
#   segments — grupos de sentenças semanticamente coesos
#       → similarity.py, retriever.py (RAG), stance_model.py
#       Uma sentença isolada tem contexto insuficiente para embeddings.
#       Segmentos de 3-5 sentenças capturam o contexto necessário.
#
# Depende de: cleaner.py (blocks_clean) e do modelo spaCy já carregado
#             pelo normalization.py via lazy loading compartilhado.
# =============================================================================

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator


# =============================================================================
# Estruturas de dados
# =============================================================================

@dataclass
class Sentence:
    """
    Sentença individual extraída de um bloco.

    char_start / char_end referenciam posições no texto do bloco de origem,
    permitindo que explanation_generator.py destaque a sentença no original.

    sentence.segment_id é preenchido depois da segmentação, ao montar os
    Segments — garante rastreabilidade bidirecional sentença ↔ segmento.
    """
    text:        str
    block_index: int          # bloco de origem (blocks_clean[block_index])
    sent_index:  int          # posição dentro do bloco
    char_start:  int = 0      # posição inicial no texto do bloco
    char_end:    int = 0      # posição final no texto do bloco
    segment_id:  int | None = None  # preenchido após _build_segments

    # estatísticas de token — usadas para adaptive chunking e LLM routing
    token_count: int = 0      # número de tokens spaCy (excluindo punct/space)


@dataclass
class Segment:
    """
    Grupo de sentenças consecutivas sobre o mesmo contexto temático.

    sentences guarda referências aos objetos Sentence originais —
    não duplica o texto, apenas os ponteiros. text é gerado sob demanda
    pela property para evitar duplicação de memória em artigos grandes.
    """
    sentences:   list[Sentence]
    block_index: int
    seg_index:   int          # id global único entre todos os segmentos

    # estatísticas agregadas — geradas uma vez em _build_segments
    token_count: int = 0      # soma dos token_count das sentenças

    @property
    def text(self) -> str:
        """Gera o texto do segmento sob demanda — não armazena cópia separada."""
        return " ".join(s.text for s in self.sentences)

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)


# =============================================================================
# TextSegmenter
# =============================================================================

class TextSegmenter:

    # -------------------------------------------------------------------------
    # Configuração de chunking
    #
    # SENTENCES_PER_SEGMENT: sweet spot para embeddings semânticos.
    #   Menos que 3 = contexto insuficiente. Mais que 6 = perde especificidade.
    #
    # SEGMENT_OVERLAP: sobreposição em sentenças entre segmentos consecutivos.
    #   Garante que afirmações na fronteira entre segmentos sejam cobertas
    #   por pelo menos um segmento completo. Padrão em pipelines RAG.
    #
    # MIN_SEGMENT_TOKENS: segmentos abaixo desse limiar são descartados.
    #   Evita segmentos de 1 sentença curtíssima no fim de um bloco.
    # -------------------------------------------------------------------------
    SENTENCES_PER_SEGMENT: int = 4
    SEGMENT_OVERLAP:       int = 1
    MIN_SENTENCE_CHARS:    int = 20
    MIN_SEGMENT_TOKENS:    int = 15

    # -------------------------------------------------------------------------
    # Marcadores discursivos de mudança de tópico em português jornalístico.
    # Usados pela segmentação semântica futura para identificar fronteiras
    # de tópico independentemente do tamanho fixo do chunk.
    #
    # IMPLEMENTAÇÃO FUTURA — não usado ainda, apenas documentado aqui
    # para facilitar a extensão sem refatoração da estrutura.
    # -------------------------------------------------------------------------
    _DISCOURSE_MARKERS = {
        "contraste":   ["no entanto", "por outro lado", "contudo", "todavia",
                        "porém", "entretanto", "apesar disso", "em contrapartida"],
        "adição":      ["além disso", "ademais", "outrossim", "igualmente",
                        "do mesmo modo", "da mesma forma"],
        "conclusão":   ["portanto", "assim", "dessa forma", "desse modo",
                        "logo", "por conseguinte", "em suma", "concluindo"],
        "localização": ["já em", "no", "em brasília", "em são paulo",
                        "no rio", "no exterior"],
        "tempo":       ["enquanto isso", "ao mesmo tempo", "nesse meio tempo",
                        "posteriormente", "anteriormente"],
    }

    # =========================================================================
    # Carregamento do modelo spaCy
    #
    # spaCy é a solução correta para sentence splitting em português.
    # Regex SEMPRE quebra em casos edge: "EUA.", "U.S.", "etc.", "p.ex.",
    # "R$ 1.500.000", abreviações de meses, nomes próprios com ponto etc.
    #
    # O modelo é o mesmo que normalization.py usa — lazy loading garante
    # que seja carregado apenas uma vez por processo, não por chamada.
    # =========================================================================

    _nlp = None

    @classmethod
    def _get_nlp(cls):
        """
        Retorna o modelo spaCy, carregando uma única vez.

        Desabilita ner e parser para velocidade — só precisamos do
        sentencizer (que depende do tagger, mantido ativo).

        O sentencizer do spaCy lida corretamente com:
          - "EUA.", "U.S.", "etc.", "p.ex."
          - "R$ 1.500.000"
          - Abreviações de títulos: "Dr.", "Sr.", "Sra.", "Art."
          - Abreviações de meses: "jan.", "fev.", "mar."
          - Sentenças sem ponto final (último parágrafo de notícia)
        """
        if cls._nlp is None:
            try:
                import spacy
                cls._nlp = spacy.load(
                    "pt_core_news_sm",
                    disable=["ner", "parser"],  # parser seria redundante com sentencizer
                )
                cls._nlp.max_length = 2_000_000

                # adiciona sentencizer explicitamente após desabilitar o parser
                # o parser do spaCy também faz sentence splitting, mas é muito
                # mais lento — o sentencizer de regras é suficiente e rápido
                if "sentencizer" not in cls._nlp.pipe_names:
                    cls._nlp.add_pipe("sentencizer")

            except OSError:
                raise ImportError(
                    "Modelo spaCy não encontrado. Execute:\n"
                    "  python -m spacy download pt_core_news_sm"
                )
        return cls._nlp

    # =========================================================================
    # Segmentação de sentenças via spaCy
    # =========================================================================

    @classmethod
    def _split_into_sentences(
        cls,
        text: str,
        block_index: int,
    ) -> list[Sentence]:
        """
        Divide um bloco em sentenças usando o sentencizer do spaCy.

        Por que spaCy e não regex:
          - Lida corretamente com abreviações ("Dr.", "EUA.", "R$ 1.500")
          - Usa o modelo de linguagem para decidir quando "." encerra sentença
          - Não precisa de lookahead/lookbehind complexos frágeis de manter
          - Fornece token_count sem trabalho extra (doc.sents já tokeniza)

        Rastreamento de posições (char_start / char_end):
          Usamos sent.start_char e sent.end_char do spaCy, que são posições
          exatas no texto original. Isso é mais confiável que str.find() +
          cursor, que falha com sentenças repetidas:
            "A inflação caiu. A inflação caiu novamente."
          find() com cursor pode pegar a ocorrência errada se o cursor não
          avançar corretamente — spaCy resolve isso com spans exatos.
        """
        nlp = cls._get_nlp()
        doc = nlp(text)

        sentences: list[Sentence] = []

        for sent_index, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()

            if len(sent_text) < cls.MIN_SENTENCE_CHARS:
                continue

            # conta tokens reais (exclui pontuação e espaços)
            # usado para adaptive chunking e estatísticas de LLM routing
            token_count = sum(
                1 for token in sent
                if not token.is_punct and not token.is_space
            )

            sentences.append(Sentence(
                text        = sent_text,
                block_index = block_index,
                sent_index  = sent_index,
                char_start  = sent.start_char,   # posição exata no texto original
                char_end    = sent.end_char,
                token_count = token_count,
            ))

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
            Sentenças: [S0, S1, S2, S3, S4, S5, S6, S7, S8]
            Segmento 0: [S0, S1, S2, S3]   tokens: soma de S0-S3
            Segmento 1: [S3, S4, S5, S6]   ← S3 repetido (overlap)
            Segmento 2: [S6, S7, S8]        ← chunk menor no fim

        Tratamento de chunks pequenos no fim:
            O último chunk pode ter menos de SENTENCES_PER_SEGMENT sentenças.
            Em vez de descartar, verificamos se tem tokens suficientes
            (MIN_SEGMENT_TOKENS). Chunks pequenos com conteúdo são mantidos;
            fragmentos triviais (1 sentença curtíssima) são descartados.

        Rastreabilidade bidirecional:
            Após criar o segmento, preenche segment_id em cada Sentence
            do chunk. Isso permite ir de sentença → segmento e de
            segmento → sentenças sem estrutura adicional.

        IMPLEMENTAÇÃO FUTURA — segmentação semântica:
            O próximo nível é substituir o chunking fixo por detecção de
            fronteiras semânticas usando similaridade entre sentenças e
            marcadores discursivos (_DISCOURSE_MARKERS). A estrutura de
            dados (Sentence, Segment) já suporta isso sem mudança.
        """
        if not sentences:
            return []

        segments: list[Segment] = []
        step      = max(1, cls.SENTENCES_PER_SEGMENT - cls.SEGMENT_OVERLAP)
        seg_index = seg_offset

        for i in range(0, len(sentences), step):
            chunk = sentences[i : i + cls.SENTENCES_PER_SEGMENT]

            # calcula tokens do chunk — usado para filtrar chunks triviais
            chunk_tokens = sum(s.token_count for s in chunk)

            # descarta chunks com tokens insuficientes (fragmentos triviais)
            if chunk_tokens < cls.MIN_SEGMENT_TOKENS:
                continue

            segment = Segment(
                sentences   = chunk,         # referências, não cópias
                block_index = block_index,
                seg_index   = seg_index,
                token_count = chunk_tokens,
            )

            # rastreabilidade bidirecional: sentença sabe a qual segmento pertence
            # se uma sentença estiver em múltiplos segmentos (overlap), o último
            # seg_index prevalece — isso é aceitável para rastreabilidade
            for sent in chunk:
                sent.segment_id = seg_index

            segments.append(segment)
            seg_index += 1

        return segments

    # =========================================================================
    # Estatísticas do bloco
    # =========================================================================

    @staticmethod
    def _block_stats(sentences: list[Sentence]) -> dict:
        """
        Calcula estatísticas de token do bloco para adaptive chunking e routing.

        token_count e avg_tokens_per_sent são usados futuramente para:
          - Adaptive chunking: blocos longos → chunks maiores; curtos → menores
          - LLM routing: artigos longos podem precisar de truncation antes do BERT
          - Truncation: sinalizar quando o artigo excede context window do modelo
        """
        if not sentences:
            return {
                "sentence_count":     0,
                "token_count":        0,
                "avg_tokens_per_sent": 0.0,
                "min_sent_tokens":    0,
                "max_sent_tokens":    0,
            }

        token_counts = [s.token_count for s in sentences]
        return {
            "sentence_count":      len(sentences),
            "token_count":         sum(token_counts),
            "avg_tokens_per_sent": round(sum(token_counts) / len(sentences), 1),
            "min_sent_tokens":     min(token_counts),
            "max_sent_tokens":     max(token_counts),
        }

    # =========================================================================
    # Ponto de entrada público
    # =========================================================================

    @classmethod
    def segment(cls, blocks: list[str]) -> dict:
        """
        Processa blocos limpos e retorna sentenças e segmentos.

        Args:
            blocks: lista de blocos vindos do cleaner.py (blocks_clean)

        Retorno:
            sentences     — lista de objetos Sentence (para claim_detector.py)
            segments      — lista de objetos Segment  (para similarity / RAG)
            stats         — métricas globais e por bloco para log no pipeline.py

        Sobre sentence_texts / segment_texts:
            NÃO são retornados diretamente para evitar duplicação de memória.
            O pipeline.py acessa via properties:
                [s.text for s in result["sentences"]]
                [s.text for s in result["segments"]]   ← Segment.text é property
        """
        all_sentences: list[Sentence] = []
        all_segments:  list[Segment]  = []
        block_stats:   list[dict]     = []
        seg_offset = 0

        for block_index, block in enumerate(blocks):
            # pula blocos muito curtos — cabeçalhos, metadados, labels
            if len(block.strip()) < cls.MIN_SENTENCE_CHARS * 2:
                block_stats.append({"block_index": block_index, "skipped": True})
                continue

            # ── segmentação de sentenças via spaCy ───────────────────────────
            sentences = cls._split_into_sentences(block, block_index)
            all_sentences.extend(sentences)

            # ── estatísticas do bloco ─────────────────────────────────────────
            stats = cls._block_stats(sentences)
            stats["block_index"] = block_index
            stats["skipped"]     = False
            block_stats.append(stats)

            # ── agrupamento em segmentos ──────────────────────────────────────
            # segmentos não cruzam blocos: cada bloco é chunked de forma
            # independente para preservar a separação editorial entre parágrafos
            segments = cls._build_segments(sentences, block_index, seg_offset)
            all_segments.extend(segments)
            seg_offset += len(segments)

        # estatísticas globais para log no pipeline.py
        total_tokens = sum(s.token_count for s in all_sentences)
        global_stats = {
            "blocks_processed":    len(blocks),
            "blocks_skipped":      sum(1 for b in block_stats if b.get("skipped")),
            "sentence_count":      len(all_sentences),
            "segment_count":       len(all_segments),
            "total_token_count":   total_tokens,
            "avg_tokens_per_sent": (
                round(total_tokens / len(all_sentences), 1)
                if all_sentences else 0.0
            ),
            "avg_sent_per_block":  (
                round(len(all_sentences) / len(blocks), 1)
                if blocks else 0.0
            ),
            "by_block": block_stats,
        }

        return {
            # objetos tipados — para módulos Python downstream
            "sentences": all_sentences,
            "segments":  all_segments,

            # textos gerados sob demanda pelo pipeline.py:
            # [s.text for s in result["sentences"]]
            # [s.text for s in result["segments"]]

            # métricas
            "stats": global_stats,
        }