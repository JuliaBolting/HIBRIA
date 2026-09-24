# =============================================================================
# Orquestrador central do sistema HÍBRIA.
#
# Conecta os módulos em sequência por meio de PipelineResult.
#
# Fluxo principal:
#   extensão Chrome / extractor
#        ↓
#   cleaner → normalizer → segmentation → claim_detector
#        ↓
#   retriever/RAG → similarity → stance_model
#        ↓
#   bertimbau_classifier → text_features → reputation
#        ↓
#   aggregator → explanation → formatter
#        ↓
#   explanation_generator → response_formatter
#
# A entrada pode vir de dois fluxos:
#
# 1. Extensão Chrome:
#       url + title + content
#       O conteúdo já foi capturado da página pelo navegador.
#
# 2. Execução direta:
#       apenas url
#       O TextExtractor acessa a página e realiza a extração.
# =============================================================================

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable

from pipeline.analysis.stance_model import StanceModel

logger = logging.getLogger(__name__)


# =============================================================================
# Funções auxiliares para variáveis de ambiente
# =============================================================================

def env_int(name: str, default: int) -> int:
    """
    Lê inteiro do .env com fallback seguro.
    """
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return int(value)
    except ValueError:
        logger.warning(
            f"[env] valor inválido para {name}={value!r}; "
            f"usando padrão {default}"
        )
        return default


def env_flag(name: str, default: bool = False) -> bool:
    """
    Lê booleano do .env.

    Aceita:
        true, 1, yes, y, sim, s, on
    """
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    return value.strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "sim",
        "s",
        "on",
    }


# =============================================================================
# PipelineResult
# Contrato de dados entre todas as etapas do pipeline.
# =============================================================================

@dataclass
class PipelineResult:

    # ── extractor ─────────────────────────────────────────────────────────────

    url: str = ""
    title: str = ""
    description: str = ""

    # "static" | "playwright" | "browser"
    render_method: str = ""

    paywall_detected: bool = False

    warnings: list[str] = field(default_factory=list)

    # ── cleaner ───────────────────────────────────────────────────────────────

    blocks_clean: list[str] = field(default_factory=list)

    # ── normalizer ────────────────────────────────────────────────────────────

    blocks_bert: list[str] = field(default_factory=list)
    blocks_tfidf: list[str] = field(default_factory=list)
    blocks_similarity: list[str] = field(default_factory=list)

    # ── segmentation ──────────────────────────────────────────────────────────

    sentences: list | None = None
    segments: list | None = None

    sentence_texts: list[str] | None = None
    segment_texts: list[str] | None = None

    # ── claim_detector ────────────────────────────────────────────────────────

    claims: list | None = None

    # ── retriever ─────────────────────────────────────────────────────────────

    retrieval_results: list | None = None

    # ── similarity ────────────────────────────────────────────────────────────

    similarity_scores: list | None = None

    # ── stance_model ──────────────────────────────────────────────────────────

    stance_results: list | None = None

    # ── bertimbau_classifier ──────────────────────────────────────────────────

    classification: dict | None = None

    # ── text_features ─────────────────────────────────────────────────────────

    text_features: dict | None = None

    # ── reputação da fonte ────────────────────────────────────────────────────

    reputation: dict | None = None

    # ── aggregator ────────────────────────────────────────────────────────────

    score_final: float | None = None

    label_final: str | None = None

    score_breakdown: dict | None = None

    # ── explanation_generator ────────────────────────────────────────────────

    explanation: str | None = None
    details: list[str] = field(default_factory=list)
    explanation_source: str | None = None

    # ── response_formatter ───────────────────────────────────────────────────

    response: dict | None = None

    # ── métricas internas ─────────────────────────────────────────────────────

    _segmentation_stats: dict = field(default_factory=dict)
    _claim_stats: dict = field(default_factory=dict)

    _processing_time: dict = field(default_factory=dict)

    # ── propriedades de conveniência ──────────────────────────────────────────

    @property
    def block_count(self) -> int:
        return len(self.blocks_clean)

    @property
    def char_count(self) -> int:
        return sum(len(block) for block in self.blocks_clean)

    @property
    def claim_count(self) -> int:
        return len(self.claims) if self.claims else 0

    @property
    def evidence_count(self) -> int:
        if not self.retrieval_results:
            return 0

        return sum(
            len(result.evidences)
            for result in self.retrieval_results
        )

    # =========================================================================
    # Serialização
    # =========================================================================

    def to_dict(self) -> dict:
        """
        Serializa o resultado para JSON.

        Usado pelo FastAPI, response_formatter.py e main.py.
        """

        return {
            # identificação
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "render_method": self.render_method,
            "paywall_detected": self.paywall_detected,
            "warnings": self.warnings,

            # métricas
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
                    "claim_id": claim.claim_id,
                    "text": claim.text,
                    "normalized": claim.normalized,
                    "entities": claim.entities,
                    "subject": claim.subject,
                    "confidence": claim.confidence,
                    "has_numbers": claim.has_numbers,
                    "keywords": claim.keywords,
                }
                for claim in (self.claims or [])
            ],

            "claim_stats": self._claim_stats,

            # evidências
            "evidence_count": self.evidence_count,

            "retrieval_results": [
                {
                    "claim_id": retrieval.claim.claim_id,
                    "claim_text": retrieval.claim.text,
                    "rag_score": retrieval.rag_score,
                    "layers_used": retrieval.layers_used,
                    "layers_failed": retrieval.layers_failed,
                    "retrieval_time": retrieval.retrieval_time,

                    "evidences": [
                        {
                            "evidence_id": evidence.evidence_id,
                            "text": evidence.text,
                            "source": evidence.source,
                            "title": evidence.title,
                            "url": evidence.url,
                            "domain": evidence.domain,
                            "similarity": evidence.similarity,
                            "published_at": evidence.published_at,
                            "retrieval_layer": evidence.retrieval_layer,
                            "source_type": evidence.source_type,
                            "trusted_source": evidence.trusted_source,
                            "stance": evidence.stance,
                            "metadata": evidence.metadata,
                        }
                        for evidence in retrieval.evidences
                    ],
                }
                for retrieval in (self.retrieval_results or [])
            ],

            # similaridade semântica
            "similarity_scores": (
                [
                    {
                        "claim_id": result.claim_id,
                        "claim_text": result.claim_text,
                        "score": result.score,
                        "has_evidence": result.has_evidence,

                        "has_sufficient_evidence": getattr(
                            result,
                            "has_sufficient_evidence",
                            False,
                        ),

                        "top_evidence": (
                            {
                                "text": result.top_evidence.evidence_text[:200],
                                "source": result.top_evidence.evidence_source,
                                "url": result.top_evidence.evidence_url,
                                "layer": result.top_evidence.evidence_layer,
                                "source_type": result.top_evidence.source_type,
                                "trusted_source": result.top_evidence.trusted_source,
                                "similarity_final": result.top_evidence.similarity_final,
                                "similarity_semantic": result.top_evidence.similarity_semantic,
                                "similarity_retriever": result.top_evidence.similarity_retriever,

                                "is_sufficient": getattr(
                                    result.top_evidence,
                                    "is_sufficient",
                                    False,
                                ),
                                "metadata": getattr(
                                    result.top_evidence,
                                    "metadata",
                                    {},
                                ),
                            }
                            if result.top_evidence
                            else None
                        ),

                        "evidences": [
                            {
                                "rank": evidence.rank,
                                "text": evidence.evidence_text[:200],
                                "source": evidence.evidence_source,
                                "url": evidence.evidence_url,
                                "layer": evidence.evidence_layer,
                                "source_type": evidence.source_type,
                                "trusted_source": evidence.trusted_source,
                                "published_at": evidence.published_at,
                                "similarity_final": evidence.similarity_final,
                                "similarity_semantic": evidence.similarity_semantic,
                                "similarity_retriever": evidence.similarity_retriever,

                                "is_sufficient": getattr(
                                    evidence,
                                    "is_sufficient",
                                    False,
                                ),
                                "metadata": getattr(evidence, "metadata", {}),
                            }
                            for evidence in result.evidences
                        ],
                    }
                    for result in self.similarity_scores
                ]
                if self.similarity_scores
                else None
            ),

            # resultados das análises
            "stance_results": [
                item.to_dict() if hasattr(item, "to_dict") else item
                for item in (self.stance_results or [])
            ],

            "classification": self.classification,

            "text_features": self.text_features,

            "reputation": self.reputation,

            # resultado agregado
            "score_final": self.score_final,
            "label_final": self.label_final,
            "score_breakdown": self.score_breakdown,

            # saída
            "explanation": self.explanation,
            "details": self.details,
            "explanation_source": self.explanation_source,
            "response": self.response,
            # processamento
            "processing_time": self._processing_time,
        }


# =============================================================================
# HibriaPipeline
# =============================================================================

class HibriaPipeline:

    # VectorStore compartilhado entre análises.
    _vector_store = None

    # =========================================================================
    # VectorStore
    # =========================================================================

    @classmethod
    def _get_vector_store(cls):

        if cls._vector_store is None:

            try:

                from pipeline.retrieval.vector_store import VectorStore

                cls._vector_store = VectorStore()

                logger.info(
                    f"[pipeline] VectorStore: {cls._vector_store}"
                )

            except Exception as e:

                logger.warning(
                    f"[pipeline] VectorStore não disponível: {e}"
                )

                cls._vector_store = None

        return cls._vector_store

    # =========================================================================
    # STEP 1 — EXTRACTOR
    # =========================================================================

    @staticmethod
    def _step_extract(
        url: str,
        result: PipelineResult,
        title: str = "",
        content: str = "",
    ) -> PipelineResult:
        """
        Etapa 1: obtenção do conteúdo da página.

        Existem dois modos:

        1. Extensão Chrome:
           recebe URL + título + conteúdo já capturado.

        2. Execução direta:
           recebe apenas URL e utiliza o TextExtractor.

        Quando o conteúdo já veio da extensão, NÃO acessamos
        novamente a página pela URL.
        """

        from pipeline.preprocessing.extractor import (
            TextExtractor,
            ExtractionError,
        )

        # ---------------------------------------------------------------------
        # FLUXO DA EXTENSÃO CHROME
        # ---------------------------------------------------------------------

        if content and content.strip():

            normalized_content = content.strip()

            # A extensão envia os parágrafos separados por linha.
            blocks = [
                block.strip()
                for block in normalized_content.split("\n")
                if block.strip()
            ]

            # Remove blocos muito pequenos.
            # Normalmente correspondem a:
            # menus, botões, labels, navegação etc.
            blocks = [
                block
                for block in blocks
                if len(block) >= 40
            ]

            if not blocks:

                raise ExtractionError(
                    "Não foi possível obter conteúdo suficiente da página."
                )

            result.url = url

            result.title = (
                title.strip()
                if title
                else ""
            )

            result.description = ""

            result.render_method = "browser"

            result.paywall_detected = False

            result._raw_blocks = blocks

            result.warnings.append(
                "[extractor] conteúdo recebido diretamente "
                "da extensão Chrome"
            )

            logger.info(
                f"[extractor] conteúdo recebido da extensão: "
                f"{len(blocks)} blocos"
            )

            return result

        # ---------------------------------------------------------------------
        # FLUXO TRADICIONAL
        # ---------------------------------------------------------------------

        raw = TextExtractor.extract(url)

        result.url = raw["url"]

        result.title = raw["title"]

        result.description = raw["description"]

        result.render_method = raw["render_method"]

        result.paywall_detected = raw["paywall_detected"]

        result.warnings.extend(
            raw["warnings"]
        )

        result._raw_blocks = raw["content_blocks"]

        logger.info(
            f"[extractor] conteúdo extraído via "
            f"{result.render_method}: "
            f"{len(result._raw_blocks)} blocos"
        )

        return result

    # =========================================================================
    # STEP 2 — CLEANER
    # =========================================================================

    @staticmethod
    def _step_clean(
        result: PipelineResult,
    ) -> PipelineResult:

        from pipeline.preprocessing.cleaner import TextCleaner

        raw_blocks = getattr(
            result,
            "_raw_blocks",
            [],
        )

        result.blocks_clean = TextCleaner.clean_blocks(
            raw_blocks,
            source=result.render_method,
        )

        if hasattr(result, "_raw_blocks"):
            del result._raw_blocks

        logger.info(
            f"[cleaner] "
            f"{len(result.blocks_clean)} blocos após limpeza"
        )

        return result

    # =========================================================================
    # STEP 3 — NORMALIZER
    # =========================================================================

    @staticmethod
    def _step_normalize(
        result: PipelineResult,
    ) -> PipelineResult:

        from pipeline.preprocessing.normalization import TextNormalizer

        result.blocks_bert = TextNormalizer.normalize_blocks(
            result.blocks_clean,
            profile="bert",
        )

        result.blocks_tfidf = TextNormalizer.normalize_blocks(
            result.blocks_clean,
            profile="tfidf",
        )

        result.blocks_similarity = TextNormalizer.normalize_blocks(
            result.blocks_clean,
            profile="similarity",
        )

        logger.info(
            f"[normalizer] "
            f"{len(result.blocks_bert)} blocos normalizados "
            f"(bert={len(result.blocks_bert)}, "
            f"tfidf={len(result.blocks_tfidf)}, "
            f"similarity={len(result.blocks_similarity)})"
        )

        return result

    # =========================================================================
    # STEP 4 — SEGMENTATION
    # =========================================================================

    @staticmethod
    def _step_segment(
        result: PipelineResult,
    ) -> PipelineResult:

        from pipeline.preprocessing.segmentation import TextSegmenter

        output = TextSegmenter.segment(
            result.blocks_clean
        )

        result.sentences = output["sentences"]

        result.segments = output["segments"]

        result.sentence_texts = [
            getattr(sentence, "text", str(sentence))
            for sentence in result.sentences
        ]
        result.segment_texts = [
            getattr(segment, "text", str(segment))
            for segment in result.segments
        ]

        result._segmentation_stats = output["stats"]

        logger.info(
            f"[segmentation] "
            f"{output['stats']['sentence_count']} sentenças · "
            f"{output['stats']['segment_count']} segmentos"
        )

        return result

    # =========================================================================
    # STEP 5 — CLAIM DETECTOR
    # =========================================================================

    @staticmethod
    def _step_detect_claims(
        result: PipelineResult,
    ) -> PipelineResult:

        from pipeline.analysis.claim_detector import ClaimDetector

        if not result.sentences:

            result.warnings.append(
                "[claim_detector] pulado — sem sentenças"
            )

            return result

        output = ClaimDetector.detect(
            result.sentences,
            max_claims=max(1, env_int("HIBRIA_MAX_CLAIMS_PER_ANALYSIS", 10)),
        )

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

    # =========================================================================
    # STEP 6 — RETRIEVER / RAG
    # =========================================================================

    @staticmethod
    def _step_retrieve(
        result: PipelineResult,
    ) -> PipelineResult:

        from pipeline.retrieval.retriever import EvidenceRetriever

        if not result.claims:

            result.warnings.append(
                "[retriever] pulado — sem claims"
            )

            return result

        vector_store = HibriaPipeline._get_vector_store()

        document_context = " ".join(
            part
            for part in [
                result.title,
                result.description,
            ]
            if part
        )

        document_context = document_context.strip()

        retriever = EvidenceRetriever(
            vector_store=vector_store,
            current_url=result.url,
            document_context=document_context,
        )

        result.retrieval_results = (
            retriever.retrieve_batch(
                result.claims,
                top_k=env_int(
                    "HIBRIA_RETRIEVAL_TOP_K",
                    10,
                ),
            )
        )

        logger.info(
            f"[retriever] "
            f"{result.evidence_count} evidências "
            f"para {result.claim_count} claims"
        )

        return result

    # =========================================================================
    # STEP 7 — SIMILARITY
    # =========================================================================

    @staticmethod
    def _step_similarity(
        result: PipelineResult,
    ) -> PipelineResult:

        from pipeline.analysis.similarity import SimilarityCalculator

        if (
            not result.claims
            or not result.retrieval_results
        ):

            result.warnings.append(
                "[similarity] pulado — "
                "sem claims ou evidências"
            )

            return result

        result.similarity_scores = (
            SimilarityCalculator.calculate(
                result.claims,
                result.retrieval_results,
            )
        )

        has = sum(
            1
            for item in result.similarity_scores
            if item.has_evidence
        )

        logger.info(
            f"[similarity] "
            f"{has}/{len(result.similarity_scores)} "
            f"claims com evidência"
        )

        return result

    # =========================================================================
    # STEP 8 — STANCE
    # =========================================================================

    @staticmethod
    def _step_stance(
        result: PipelineResult,
    ) -> PipelineResult:

        result.stance_results = StanceModel.analyze(
            retrieval_results=result.retrieval_results or [],
            similarity_scores=result.similarity_scores or [],
        )

        logger.info(
            f"[stance] "
            f"{len(result.stance_results or [])} "
            f"relações claim-evidência analisadas"
        )

        return result

    # =========================================================================
    # STEP 9 — BERTIMBAU
    # =========================================================================

    @staticmethod
    def _step_bertimbau(
        result: PipelineResult,
    ) -> PipelineResult:

        from pipeline.analysis.bertimbau_classifier import (
            BERTimbauClassifier,
        )

        result.classification = (
            BERTimbauClassifier().classify(
                result.blocks_bert
            )
        )

        status = result.classification.get(
            "status"
        )

        label = result.classification.get(
            "label",
            "indefinido",
        )

        score = result.classification.get(
            "score"
        )

        if status == "ok":

            logger.info(
                f"[bertimbau] "
                f"label={label} · "
                f"score={score}"
            )

        else:

            message = result.classification.get(
                "message",
                "sem detalhes",
            )

            result.warnings.append(
                f"[bertimbau] {status}: {message}"
            )

            logger.warning(
                f"[bertimbau] {status}: {message}"
            )

        return result

    # =========================================================================
    # STEP 10 — TEXT FEATURES
    # =========================================================================

    @staticmethod
    def _step_text_features(
        result: PipelineResult,
    ) -> PipelineResult:

        from pipeline.analysis.text_features import (
            TextFeatureExtractor,
        )

        result.text_features = (
            TextFeatureExtractor.extract(
                result.blocks_clean,
                title=result.title,
                description=result.description,
                sentences=result.sentences or [],
                claims=result.claims or [],
            )
        )

        status = result.text_features.get(
            "status"
        )

        label = result.text_features.get(
            "label",
            "indefinido",
        )

        score = result.text_features.get(
            "score"
        )

        risk = result.text_features.get(
            "risk_score"
        )

        if status == "ok":

            logger.info(
                f"[text_features] "
                f"label={label} · "
                f"score={score} · "
                f"risk={risk}"
            )

        else:

            message = result.text_features.get(
                "message",
                "sem detalhes",
            )

            result.warnings.append(
                f"[text_features] "
                f"{status}: {message}"
            )

            logger.warning(
                f"[text_features] "
                f"{status}: {message}"
            )

        return result

    # =========================================================================
    # STEP 11 — REPUTATION
    # =========================================================================

    @staticmethod
    def _step_reputation(
        result: PipelineResult,
    ) -> PipelineResult:

        from pipeline.analysis.reputation.service import (
            SourceReputationService,
        )

        reputation = (
            SourceReputationService().get_or_evaluate(
                result.url,
                trigger="pipeline",
            )
        )

        result.reputation = reputation.to_dict()

        logger.info(
            f"[reputation] "
            f"domínio={result.reputation.get('canonical_domain') or result.reputation.get('domain')} · "
            f"status={result.reputation.get('status')} · "
            f"nota={result.reputation.get('note')}"
        )

        return result

    # =========================================================================
    # STEP 12 — AGGREGATOR
    # =========================================================================

    @staticmethod
    def _step_aggregate(
        result: PipelineResult,
    ) -> PipelineResult:

        from pipeline.output.aggregator import Aggregator

        aggregated = Aggregator.aggregate(
            result
        )

        result.score_final = aggregated[
            "score"
        ]

        result.label_final = aggregated[
            "label"
        ]

        result.score_breakdown = aggregated[
            "breakdown"
        ]

        logger.info(
            f"[aggregator] "
            f"score={result.score_final} · "
            f"label={result.label_final}"
        )

        return result

    # =========================================================================
    # STEP 13 — EXPLANATION
    # =========================================================================

    @staticmethod
    def _step_explain(result: PipelineResult) -> PipelineResult:
        """
        Gera a explicação textual para o usuário.

        Esta etapa é auxiliar:
        se o LLM falhar, a análise híbrida permanece válida.
        """

        from pipeline.output.explanation_generator import ExplanationGenerator

        report = ExplanationGenerator.generate(result)

        if report:
            result.explanation = report.get("explanation")
            result.details = list(report.get("details") or [])[:4]
            result.explanation_source = "qwen"
            logger.info(
                "[explanation_generator] explicação e detalhes gerados com sucesso"
            )
        else:
            fallback = ExplanationGenerator.fallback_report(result)
            result.explanation = fallback["explanation"]
            result.details = fallback["details"]
            result.explanation_source = "fallback"
            result.warnings.append(
                "[explanation_generator] Qwen local indisponível — "
                "usada explicação determinística de contingência"
            )

        return result

    # =========================================================================
    # STEP 15 — FORMATTER
    # =========================================================================

    @staticmethod
    def _step_format(
        result: PipelineResult,
    ) -> PipelineResult:
        """
        Etapa final: formatação da resposta para a API/extensão.
        """

        from pipeline.output.response_formatter import ResponseFormatter

        result.response = ResponseFormatter.format(result)

        logger.info(
            "[formatter] resposta final preparada"
        )

        return result

    # =========================================================================
    # PONTO DE ENTRADA PÚBLICO
    # =========================================================================

    @classmethod
    def run(
        cls,
        url: str,
        title: str = "",
        content: str = "",
        progress_callback: Callable[[str], None] | None = None,
    ) -> PipelineResult:
        """
        Executa o pipeline completo.

        Parâmetros
        ----------
        url:
            URL da página analisada.

        title:
            Título capturado pela extensão.
            Opcional.

        content:
            Conteúdo textual já capturado pela extensão.
            Opcional.

            Quando informado, o TextExtractor NÃO é chamado.

        Fluxos suportados
        -----------------

        Extensão Chrome:

            HibriaPipeline.run(
                url=url,
                title=title,
                content=content,
            )

        Execução direta:

            HibriaPipeline.run(
                url=url,
            )
        """

        from pipeline.preprocessing.extractor import (
            ExtractionError,
        )

        result = PipelineResult()

        start_time = time.time()

        # =====================================================================
        # STEPS
        # =====================================================================

        steps_implemented = [

            (
                "extractor",
                lambda r: cls._step_extract(
                    url,
                    r,
                    title=title,
                    content=content,
                ),
            ),

            (
                "cleaner",
                cls._step_clean,
            ),

            (
                "normalizer",
                cls._step_normalize,
            ),

            (
                "segmentation",
                cls._step_segment,
            ),

            (
                "claim_detector",
                cls._step_detect_claims,
            ),

            (
                "retriever",
                cls._step_retrieve,
            ),

            (
                "similarity",
                cls._step_similarity,
            ),

            (
                "stance",
                cls._step_stance,
            ),

            (
                "bertimbau",
                cls._step_bertimbau,
            ),

            (
                "text_features",
                cls._step_text_features,
            ),

            (
                "reputation",
                cls._step_reputation,
            ),

            (
                "aggregator",
                cls._step_aggregate,
            ),

        ]

        steps_pending = [

            (
                "explanation",
                cls._step_explain,
            ),

            (
                "formatter",
                cls._step_format,
            ),
        ]

        all_steps = (
            steps_implemented
            + steps_pending
        )

        show_progress = env_flag(
            "HIBRIA_SHOW_PIPELINE_PROGRESS",
            True,
        )

        # =====================================================================
        # EXECUÇÃO
        # =====================================================================

        for name, step in all_steps:

            step_start = time.time()

            # Usado pela API assíncrona para que a extensão consiga recuperar
            # a etapa atual mesmo depois que o popup for fechado. A callback é
            # executada fora do bloco de tolerância a falhas: cancelamentos e
            # erros de controle não devem ser convertidos em simples avisos.
            if progress_callback is not None:
                progress_callback(name)

            if show_progress:

                print(
                    f"[pipeline] "
                    f"Iniciando etapa: {name}...",
                    flush=True,
                )

            try:

                result = step(result)

            except ExtractionError:

                # Erro de extração é fatal.
                raise

            except Exception as e:

                msg = (
                    f"[{name}] falhou: "
                    f"{type(e).__name__}: {e}"
                )

                result.warnings.append(
                    msg
                )

                logger.error(
                    msg,
                    exc_info=True,
                )

            result._processing_time[name] = round(
                time.time() - step_start,
                3,
            )

            if show_progress:

                print(
                    f"[pipeline] "
                    f"Etapa concluída: {name} "
                    f"({result._processing_time[name]}s)",
                    flush=True,
                )

        # =====================================================================
        # TEMPO TOTAL
        # =====================================================================

        result._processing_time["total"] = round(
            time.time() - start_time,
            3,
        )

        logger.info(
            f"[pipeline] concluído em "
            f"{result._processing_time['total']}s · "
            f"{result.claim_count} claims · "
            f"{result.evidence_count} evidências"
        )

        return result
