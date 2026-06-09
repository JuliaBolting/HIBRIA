# =============================================================================
# pipeline.py
# Orquestrador central do sistema HÍBRIA.
#
# Conecta todos os módulos implementados em sequência, passando o
# PipelineResult entre as etapas. Módulos ainda não implementados
# estão documentados como pendentes — o pipeline continua funcionando
# parcialmente sem eles.
#
# Fluxo atual (implementado):
#   extractor → cleaner → normalizer → segmentation →
#   claim_detector → retriever (Wikipedia + VectorStore)
#
# Fluxo completo (TCC):
#   + similarity → stance_model → bertimbau_classifier →
#   + text_features → reputation_engine → aggregator →
#   + explanation_generator → response_formatter
# =============================================================================

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# PipelineResult
# Contrato de dados entre todas as etapas do pipeline.
#
# Cada módulo lê o que precisa e escreve no seu campo.
# Campos ainda não implementados ficam como None — o pipeline
# continua funcionando parcialmente enquanto o projeto cresce.
# =============================================================================


@dataclass
class PipelineResult:

    # ── extractor ─────────────────────────────────────────────────────────────
    url: str = ""
    title: str = ""
    description: str = ""
    render_method: str = ""  # "static" | "playwright"
    paywall_detected: bool = False
    warnings: list[str] = field(default_factory=list)

    # ── cleaner ───────────────────────────────────────────────────────────────
    blocks_clean: list[str] = field(default_factory=list)

    # ── normalizer ────────────────────────────────────────────────────────────
    blocks_bert: list[str] = field(default_factory=list)
    blocks_tfidf: list[str] = field(default_factory=list)
    blocks_similarity: list[str] = field(default_factory=list)

    # ── segmentation ──────────────────────────────────────────────────────────
    sentences: list | None = None  # list[Sentence]
    segments: list | None = None  # list[Segment]
    sentence_texts: list[str] | None = None
    segment_texts: list[str] | None = None

    # ── claim_detector ────────────────────────────────────────────────────────
    claims: list | None = None  # list[Claim]

    # ── retriever ─────────────────────────────────────────────────────────────
    retrieval_results: list | None = None  # list[RetrievalResult]

    # ── similarity (pendente) ─────────────────────────────────────────────────
    similarity_scores: list | None = None

    # ── stance_model (pendente) ───────────────────────────────────────────────
    stance_results: list | None = None

    # ── bertimbau_classifier (pendente) ───────────────────────────────────────
    classification: dict | None = None
    # ex: {"label": "fake", "score": 0.87, "probabilities": {...}}

    # ── text_features (pendente) ──────────────────────────────────────────────
    text_features: dict | None = None
    # ex: {"sensationalism": 0.4, "emotional_language": 0.6, ...}

    # ── reputation_engine (pendente) ──────────────────────────────────────────
    reputation: dict | None = None
    # ex: {"domain": "g1.globo.com", "score": 0.95, "category": "confiável"}

    # ── aggregator (pendente) ─────────────────────────────────────────────────
    score_final: float | None = None  # 0.0 a 100.0
    label_final: str | None = (
        None  # "confiável" | "parcialmente confiável" | "não confiável"
    )
    score_breakdown: dict | None = None  # scores individuais por componente

    # ── explanation_generator (pendente) ──────────────────────────────────────
    explanation: str | None = None

    # ── response_formatter (pendente) ─────────────────────────────────────────
    response: dict | None = None  # JSON final para a extensão

    # ── métricas internas ─────────────────────────────────────────────────────
    _segmentation_stats: dict = field(default_factory=dict)
    _claim_stats: dict = field(default_factory=dict)
    _processing_time: dict = field(default_factory=dict)  # tempo por etapa

    # ── properties de conveniência ────────────────────────────────────────────

    @property
    def block_count(self) -> int:
        return len(self.blocks_clean)

    @property
    def char_count(self) -> int:
        return sum(len(b) for b in self.blocks_clean)

    @property
    def claim_count(self) -> int:
        return len(self.claims) if self.claims else 0

    @property
    def evidence_count(self) -> int:
        if not self.retrieval_results:
            return 0
        return sum(len(r.evidences) for r in self.retrieval_results)

    def to_dict(self) -> dict:
        """
        Serializa o resultado para JSON.
        Usado pelo response_formatter.py e para debug no main.py.
        Exclui campos internos (_) e objetos não serializáveis.
        """
        return {
            # identificação
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "render_method": self.render_method,
            "paywall_detected": self.paywall_detected,
            "warnings": self.warnings,
            # métricas de extração
            "block_count": self.block_count,
            "char_count": self.char_count,
            # textos processados
            "blocks_clean": self.blocks_clean,
            "blocks_bert": self.blocks_bert,
            "blocks_tfidf": self.blocks_tfidf,
            "blocks_similarity": self.blocks_similarity,
            # segmentação
            "sentence_texts": self.sentence_texts,
            "segment_texts": self.segment_texts,
            "segmentation_stats": self._segmentation_stats,
            # claims
            "claim_count": self.claim_count,
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "text": c.text,
                    "normalized": c.normalized,
                    "entities": c.entities,
                    "subject": c.subject,
                    "confidence": c.confidence,
                    "has_numbers": c.has_numbers,
                    "keywords": c.keywords,
                }
                for c in (self.claims or [])
            ],
            "claim_stats": self._claim_stats,
            # evidências
            "evidence_count": self.evidence_count,
            "retrieval_results": [
                {
                    "claim_id": r.claim.claim_id,
                    "claim_text": r.claim.text,
                    "layers_used": r.layers_used,
                    "layers_failed": r.layers_failed,
                    "retrieval_time": r.retrieval_time,
                    "evidences": [
                        {
                            "text": e.text,
                            "source": e.source,
                            "url": e.url,
                            "similarity": e.similarity,
                            "published_at": e.published_at,
                            "retrieval_layer": e.retrieval_layer,
                        }
                        for e in r.evidences
                    ],
                }
                for r in (self.retrieval_results or [])
            ],
            # similaridade semântica
            "similarity_scores": (
                [
                    {
                        "claim_id": r.claim_id,
                        "score": r.score,
                        "has_evidence": r.has_evidence,
                        "top_evidence": (
                            {
                                "source": r.top_evidence.evidence_source,
                                "similarity_final": r.top_evidence.similarity_final,
                                "similarity_semantic": r.top_evidence.similarity_semantic,
                            }
                            if r.top_evidence
                            else None
                        ),
                    }
                    for r in self.similarity_scores
                ]if self.similarity_scores else None,
            ),
            # resultados de análise (pendentes — None até implementar)
            "stance_results": self.stance_results,
            "classification": self.classification,
            "text_features": self.text_features,
            "reputation": self.reputation,
            # resultado final (pendente)
            "score_final": self.score_final,
            "label_final": self.label_final,
            "score_breakdown": self.score_breakdown,
            "explanation": self.explanation,
            # tempo de processamento por etapa
            "processing_time": self._processing_time,
        }


# =============================================================================
# HibriaPipeline
# Orquestra todas as etapas em sequência.
# Cada step recebe e devolve o PipelineResult.
# =============================================================================


class HibriaPipeline:

    # VectorStore compartilhado entre análises — carregado uma vez na RAM
    _vector_store = None

    @classmethod
    def _get_vector_store(cls):
        """
        Lazy loading do VectorStore.
        Carrega o índice FAISS do disco na primeira chamada
        e mantém na RAM para análises subsequentes.
        """
        if cls._vector_store is None:
            try:
                from pipeline.retrieval.vector_store import VectorStore

                cls._vector_store = VectorStore()
                logger.info(f"[pipeline] VectorStore: {cls._vector_store}")
            except Exception as e:
                logger.warning(f"[pipeline] VectorStore não disponível: {e}")
                cls._vector_store = None
        return cls._vector_store

    # =========================================================================
    # Steps individuais
    # =========================================================================

    @staticmethod
    def _step_extract(url: str, result: PipelineResult) -> PipelineResult:
        """
        Etapa 1: extração do conteúdo da página.
        extractor.py → HTML → blocks_raw
        """
        from pipeline.preprocessing.extractor import TextExtractor

        raw = TextExtractor.extract(url)

        result.url = raw["url"]
        result.title = raw["title"]
        result.description = raw["description"]
        result.render_method = raw["render_method"]
        result.paywall_detected = raw["paywall_detected"]
        result.warnings.extend(raw["warnings"])

        # blocos brutos armazenados temporariamente para o cleaner
        result._raw_blocks = raw["content_blocks"]

        logger.info(
            f"[extractor] {len(raw['content_blocks'])} blocos · "
            f"{raw['char_count']} chars · via {raw['render_method']}"
        )
        return result

    @staticmethod
    def _step_clean(result: PipelineResult) -> PipelineResult:
        """
        Etapa 2: limpeza do texto bruto.
        cleaner.py → remove ruído, emojis, artefatos
        """
        from pipeline.preprocessing.cleaner import TextCleaner

        raw_blocks = getattr(result, "_raw_blocks", [])

        result.blocks_clean = TextCleaner.clean_blocks(
            raw_blocks,
            source=result.render_method,
        )

        # libera memória — blocos brutos não são mais necessários
        if hasattr(result, "_raw_blocks"):
            del result._raw_blocks

        logger.info(f"[cleaner] {len(result.blocks_clean)} blocos após limpeza")
        return result

    @staticmethod
    def _step_normalize(result: PipelineResult) -> PipelineResult:
        """
        Etapa 3: normalização em três perfis.
        normalization.py → bert / tfidf / similarity
        """
        from pipeline.preprocessing.normalization import TextNormalizer

        result.blocks_bert = TextNormalizer.normalize_blocks(
            result.blocks_clean, profile="bert"
        )
        result.blocks_tfidf = TextNormalizer.normalize_blocks(
            result.blocks_clean, profile="tfidf"
        )
        result.blocks_similarity = TextNormalizer.normalize_blocks(
            result.blocks_clean, profile="similarity"
        )

        logger.info(
            f"[normalizer] {len(result.blocks_bert)} blocos normalizados "
            f"(bert={len(result.blocks_bert)}, "
            f"tfidf={len(result.blocks_tfidf)}, "
            f"similarity={len(result.blocks_similarity)})"
        )
        return result

    @staticmethod
    def _step_segment(result: PipelineResult) -> PipelineResult:
        """
        Etapa 4: segmentação em sentenças e segmentos.
        segmentation.py →
            sentences → claim_detector
            segments  → similarity / RAG
        """
        from pipeline.preprocessing.segmentation import TextSegmenter

        output = TextSegmenter.segment(result.blocks_clean)

        result.sentences = output["sentences"]
        result.segments = output["segments"]
        result._segmentation_stats = output["stats"]

        logger.info(
            f"[segmentation] "
            f"{output['stats']['sentence_count']} sentenças · "
            f"{output['stats']['segment_count']} segmentos"
        )
        return result

    @staticmethod
    def _step_detect_claims(result: PipelineResult) -> PipelineResult:
        """
        Etapa 5: detecção de afirmações verificáveis.
        claim_detector.py → sentences → list[Claim]
        """
        from pipeline.analysis.claim_detector import ClaimDetector

        if not result.sentences:
            result.warnings.append("[claim_detector] pulado — sem sentenças")
            return result

        output = ClaimDetector.detect(result.sentences)

        result.claims = output["claims"]
        result._claim_stats = output["stats"]

        logger.info(
            f"[claim_detector] "
            f"{output['stats']['claims_found']} claims · "
            f"descartados: "
            f"{output['stats']['discarded_opinion']} opinião · "
            f"{output['stats']['discarded_noise']} ruído · "
            f"{output['stats']['discarded_rhetorical']} retórica"
        )
        return result

    @staticmethod
    def _step_retrieve(result: PipelineResult) -> PipelineResult:
        """
        Etapa 6: busca de evidências externas (RAG).
        retriever.py → list[Claim] → list[RetrievalResult]

        Camadas:
          1. VectorStore local  (FAISS)     ⭐⭐⭐⭐⭐
          2. Wikipedia API                  ⭐⭐⭐⭐
          3. Google Fact Check (requer key) ⭐⭐⭐⭐
          4. Bing Search       (requer key) ⭐⭐
        """
        from pipeline.retrieval.retriever import EvidenceRetriever

        if not result.claims:
            result.warnings.append("[retriever] pulado — sem claims")
            return result

        vector_store = HibriaPipeline._get_vector_store()
        retriever = EvidenceRetriever(vector_store=vector_store)

        result.retrieval_results = retriever.retrieve_batch(
            result.claims,
            top_k=10,
        )

        logger.info(
            f"[retriever] {result.evidence_count} evidências "
            f"para {result.claim_count} claims"
        )
        return result

    @staticmethod
    def _step_similarity(result: PipelineResult) -> PipelineResult:
        """
        Etapa 7: similaridade semântica entre claims e evidências.
        similarity.py → cosine similarity via embeddings + score do retriever
        Rerankeai evidências por similarity_final para stance_model e aggregator.
        """
        from pipeline.analysis.similarity import SimilarityCalculator

        if not result.claims or not result.retrieval_results:
            result.warnings.append("[similarity] pulado — sem claims ou evidências")
            return result

        result.similarity_scores = SimilarityCalculator.calculate(
            result.claims,
            result.retrieval_results,
        )

        has = sum(1 for r in result.similarity_scores if r.has_evidence)
        logger.info(
            f"[similarity] {has}/{len(result.similarity_scores)} claims com evidência"
        )
        return result

    # =========================================================================
    # Steps pendentes — documentados para implementação futura
    # =========================================================================

    @staticmethod
    def _step_stance(result: PipelineResult) -> PipelineResult:
        """
        🔲 PENDENTE: stance_model.py
        Classifica cada evidência como supports / refutes / neutral
        em relação ao claim correspondente.
        """
        # TODO: implementar
        # from pipeline.analysis.stance_model import StanceModel
        # result.stance_results = StanceModel.classify(
        #     result.claims,
        #     result.retrieval_results,
        # )
        return result

    @staticmethod
    def _step_bertimbau(result: PipelineResult) -> PipelineResult:
        """
        🔲 PENDENTE: bertimbau_classifier.py
        Classifica o texto com BERTimbau fine-tuned no Fake.Br Corpus.
        Entrada: blocks_bert (perfil sem alterações para Transformers)
        """
        # TODO: implementar
        # from pipeline.analysis.bertimbau_classifier import BERTimbauClassifier
        # result.classification = BERTimbauClassifier().classify(
        #     result.blocks_bert,
        # )
        return result

    @staticmethod
    def _step_text_features(result: PipelineResult) -> PipelineResult:
        """
        🔲 PENDENTE: text_features.py
        Extrai características linguísticas: sensacionalismo,
        linguagem emocional, vocabulário, estrutura sintática.
        """
        # TODO: implementar
        # from pipeline.analysis.text_features import TextFeatureExtractor
        # result.text_features = TextFeatureExtractor.extract(
        #     result.blocks_clean,
        # )
        return result

    @staticmethod
    def _step_reputation(result: PipelineResult) -> PipelineResult:
        """
        🔲 PENDENTE: reputation_engine.py
        Avalia a reputação do domínio da notícia.
        Consulta a tabela fontes_confiaveis do PostgreSQL.
        """
        # TODO: implementar
        # from pipeline.analysis.reputation_engine import ReputationEngine
        # result.reputation = ReputationEngine.evaluate(result.url)
        return result

    @staticmethod
    def _step_aggregate(result: PipelineResult) -> PipelineResult:
        """
        🔲 PENDENTE: aggregator.py
        Combina todos os scores em um índice final 0-100.
        Fórmula definida na seção 3.5.6 do TCC (a ser especificada).
        """
        # TODO: implementar
        # from pipeline.output.aggregator import Aggregator
        # agg = Aggregator.aggregate(result)
        # result.score_final     = agg["score"]
        # result.label_final     = agg["label"]
        # result.score_breakdown = agg["breakdown"]
        return result

    @staticmethod
    def _step_explain(result: PipelineResult) -> PipelineResult:
        """
        🔲 PENDENTE: explanation_generator.py
        LLM gera explicação textual interpretável para o usuário.
        Entrada: score_final + claims + evidências + stance
        """
        # TODO: implementar
        # from pipeline.output.explanation_generator import ExplanationGenerator
        # result.explanation = ExplanationGenerator.generate(result)
        return result

    @staticmethod
    def _step_format(result: PipelineResult) -> PipelineResult:
        """
        🔲 PENDENTE: response_formatter.py
        Organiza o resultado final em JSON para a extensão.
        """
        # TODO: implementar
        # from pipeline.output.response_formatter import ResponseFormatter
        # result.response = ResponseFormatter.format(result)
        return result

    # =========================================================================
    # Ponto de entrada público
    # =========================================================================

    @classmethod
    def run(cls, url: str) -> PipelineResult:
        """
        Executa o pipeline completo para uma URL.

        Tratamento de erros por etapa:
          - ExtractionError → erro fatal, interrompe o pipeline
          - Outros erros    → registra em warnings, continua com o que tem

        Isso garante que o pipeline nunca retorna vazio por falha
        em um módulo intermediário — retorna o resultado parcial
        com o warning explicando o que falhou.
        """
        from pipeline.preprocessing.extractor import ExtractionError

        result = PipelineResult()
        start_time = time.time()

        # ── steps implementados ───────────────────────────────────────────────
        steps_implemented = [
            ("extractor", lambda r: cls._step_extract(url, r)),
            ("cleaner", cls._step_clean),
            ("normalizer", cls._step_normalize),
            ("segmentation", cls._step_segment),
            ("claim_detector", cls._step_detect_claims),
            ("retriever", cls._step_retrieve),
            ("similarity", cls._step_similarity),
        ]

        # ── steps pendentes ───────────────────────────────────────────────────
        steps_pending = [
            ("stance", cls._step_stance),
            ("bertimbau", cls._step_bertimbau),
            ("text_features", cls._step_text_features),
            ("reputation", cls._step_reputation),
            ("aggregator", cls._step_aggregate),
            ("explanation", cls._step_explain),
            ("formatter", cls._step_format),
        ]

        all_steps = steps_implemented + steps_pending

        for name, step in all_steps:
            step_start = time.time()
            try:
                result = step(result)
            except ExtractionError:
                # erro fatal — interrompe o pipeline
                raise
            except Exception as e:
                # erro não-fatal — registra e continua
                msg = f"[{name}] falhou: {type(e).__name__}: {e}"
                result.warnings.append(msg)
                logger.error(msg, exc_info=True)

            result._processing_time[name] = round(time.time() - step_start, 3)

        result._processing_time["total"] = round(time.time() - start_time, 3)

        logger.info(
            f"[pipeline] concluído em "
            f"{result._processing_time['total']}s · "
            f"{result.claim_count} claims · "
            f"{result.evidence_count} evidências"
        )
        return result
