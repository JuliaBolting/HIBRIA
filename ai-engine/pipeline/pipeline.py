# =============================================================================
# Orquestrador central do sistema HÍBRIA.
#
# Conecta os módulos em sequência por meio de PipelineResult.
#
# Fluxo principal:
#   extractor → cleaner → normalizer → segmentation → claim_detector →
#   retriever/RAG → similarity → stance_model → bertimbau_classifier →
#   text_features → source_reputation_service → aggregator → auto_indexer →
#   explanation_generator → response_formatter
# =============================================================================

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pipeline.analysis.stance_model import StanceModel
import os

logger = logging.getLogger(__name__)

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
            f"[env] valor inválido para {name}={value!r}; usando padrão {default}"
        )
        return default


def env_flag(name: str, default: bool = False) -> bool:
    """
    Lê booleano do .env.
    Aceita: true, 1, yes, y, sim, s, on.
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
#
# Cada módulo lê o que precisa e escreve no seu campo.
# Campos opcionais permanecem como None quando uma etapa não produz resultado.
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

    # ── similarity ────────────────────────────────────────────────────────────
    similarity_scores: list | None = None

    # ── stance_model ──────────────────────────────────────────────────────────
    stance_results: list | None = None

    # ── bertimbau_classifier ─────────────────────────────────────────────────
    classification: dict | None = None
    # ex: {"label": "fake", "score": 0.87, "probabilities": {...}}

    # ── text_features ────────────────────────────────────────────────────────
    text_features: dict | None = None
    # ex: {"sensationalism": 0.4, "emotional_language": 0.6, ...}

    # ── reputação da fonte ───────────────────────────────────────────────────
    reputation: dict | None = None
    # ex: {"status": "evaluated", "domain": "...", "note": 86, "score": 0.86}

    # ── aggregator ────────────────────────────────────────────────────────────
    score_final: float | None = None  # 0.0 a 100.0
    label_final: str | None = (
        None  # "confiável" | "parcialmente confiável" | "não confiável"
    )
    score_breakdown: dict | None = None  # scores individuais por componente

    # ── explanation_generator (pendente) ──────────────────────────────────────
    explanation: str | None = None

    # ── response_formatter (pendente) ─────────────────────────────────────────
    response: dict | None = None  # JSON final para a extensão

    # ── auto_indexer ─────────────────────────────────────────────────────────────
    auto_indexing: dict | None = None

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
                    "rag_score": r.rag_score,
                    "layers_used": r.layers_used,
                    "layers_failed": r.layers_failed,
                    "retrieval_time": r.retrieval_time,
                    "evidences": [
                        {
                            "evidence_id": e.evidence_id,
                            "text": e.text,
                            "source": e.source,
                            "title": e.title,
                            "url": e.url,
                            "domain": e.domain,
                            "similarity": e.similarity,
                            "published_at": e.published_at,
                            "retrieval_layer": e.retrieval_layer,
                            "source_type": e.source_type,
                            "trusted_source": e.trusted_source,
                            "stance": e.stance,
                            "metadata": e.metadata,
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
                        "claim_text": r.claim_text,
                        "score": r.score,
                        "has_evidence": r.has_evidence,
                        "has_sufficient_evidence": getattr(
                            r,
                            "has_sufficient_evidence",
                            False,
                        ),
                        "top_evidence": (
                            {
                                "text": r.top_evidence.evidence_text[:200],
                                "source": r.top_evidence.evidence_source,
                                "url": r.top_evidence.evidence_url,
                                "layer": r.top_evidence.evidence_layer,
                                "source_type": r.top_evidence.source_type,
                                "trusted_source": r.top_evidence.trusted_source,
                                "similarity_final": r.top_evidence.similarity_final,
                                "similarity_semantic": r.top_evidence.similarity_semantic,
                                "similarity_retriever": r.top_evidence.similarity_retriever,
                                "is_sufficient": getattr(
                                    r.top_evidence,
                                    "is_sufficient",
                                    False,
                                ),
                            }
                            if r.top_evidence
                            else None
                        ),
                        "evidences": [
                            {
                                "rank": e.rank,
                                "text": e.evidence_text[:200],
                                "source": e.evidence_source,
                                "url": e.evidence_url,
                                "layer": e.evidence_layer,
                                "source_type": e.source_type,
                                "trusted_source": e.trusted_source,
                                "published_at": e.published_at,
                                "similarity_final": e.similarity_final,
                                "similarity_semantic": e.similarity_semantic,
                                "similarity_retriever": e.similarity_retriever,
                                "is_sufficient": getattr(e, "is_sufficient", False),
                            }
                            for e in r.evidences
                        ],
                    }
                    for r in self.similarity_scores
                ]
                if self.similarity_scores
                else None
            ),
            # resultados das etapas de análise
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
            "explanation": self.explanation,
            "auto_indexing": self.auto_indexing,
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

        result.retrieval_results = retriever.retrieve_batch(
            result.claims,
            top_k=env_int("HIBRIA_RETRIEVAL_TOP_K", 10),
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
    def _step_bertimbau(result: PipelineResult) -> PipelineResult:
        """
        Etapa 8: classificação textual com BERTimbau.

        Usa blocks_bert, que preserva o texto em formato adequado para
        Transformers. O resultado é um componente auxiliar do score final;
        ele não substitui RAG, stance ou reputação da fonte.
        """
        from pipeline.analysis.bertimbau_classifier import BERTimbauClassifier

        result.classification = BERTimbauClassifier().classify(result.blocks_bert)

        status = result.classification.get("status")
        label = result.classification.get("label", "indefinido")
        score = result.classification.get("score")

        if status == "ok":
            logger.info(f"[bertimbau] label={label} · score={score}")
        else:
            message = result.classification.get("message", "sem detalhes")
            result.warnings.append(f"[bertimbau] {status}: {message}")
            logger.warning(f"[bertimbau] {status}: {message}")

        return result

    @staticmethod
    def _step_text_features(result: PipelineResult) -> PipelineResult:
        """
        Etapa 9: extração de características linguísticas e estruturais.

        text_features.py → sinais textuais auxiliares para aggregator e
        explanation_generator. Esta etapa não decide verdade/falsidade; ela
        mede risco linguístico, como sensacionalismo, clickbait, subjetividade,
        pontuação enfática e presença de atribuições/dados verificáveis.
        """
        from pipeline.analysis.text_features import TextFeatureExtractor

        result.text_features = TextFeatureExtractor.extract(
            result.blocks_clean,
            title=result.title,
            description=result.description,
            sentences=result.sentences or [],
            claims=result.claims or [],
        )

        status = result.text_features.get("status")
        label = result.text_features.get("label", "indefinido")
        score = result.text_features.get("score")
        risk = result.text_features.get("risk_score")

        if status == "ok":
            logger.info(
                f"[text_features] label={label} · "
                f"score={score} · risk={risk}"
            )
        else:
            message = result.text_features.get("message", "sem detalhes")
            result.warnings.append(f"[text_features] {status}: {message}")
            logger.warning(f"[text_features] {status}: {message}")

        return result

    @staticmethod
    def _step_reputation(result: PipelineResult) -> PipelineResult:
        """
        Avalia a reputação da fonte por um serviço único e genérico.

        Se o domínio já estiver persistido, reutiliza a avaliação. Caso contrário,
        pesquisa os critérios na web, calcula a nota e salva a fonte para as
        próximas análises. Não existem notas ou domínios definidos no código.
        """
        from pipeline.analysis.reputation.service import SourceReputationService

        reputation = SourceReputationService().get_or_evaluate(
            result.url,
            trigger="pipeline",
        )
        result.reputation = reputation.to_dict()

        logger.info(
            f"[reputation] domínio={result.reputation.get('canonical_domain') or result.reputation.get('domain')} · "
            f"status={result.reputation.get('status')} · "
            f"nota={result.reputation.get('note')}"
        )

        return result

    @staticmethod
    def _step_aggregate(result: PipelineResult) -> PipelineResult:
        """
        Etapa 9: agregação dos scores parciais.
        aggregator.py → score_final + label_final + score_breakdown.
        """
        from pipeline.output.aggregator import Aggregator

        aggregated = Aggregator.aggregate(result)

        result.score_final = aggregated["score"]
        result.label_final = aggregated["label"]
        result.score_breakdown = aggregated["breakdown"]

        logger.info(
            f"[aggregator] score={result.score_final} · " f"label={result.label_final}"
        )

        return result

    @staticmethod
    def _step_auto_index(result: PipelineResult) -> PipelineResult:
        """
        Etapa 10: indexação automática de notícias aprovadas.
        auto_indexer.py → adiciona ao FAISS como analyzed_news se score_final for confiável.
        """
        from pipeline.retrieval.auto_indexer import AutoIndexer

        vector_store = HibriaPipeline._get_vector_store()

        result.auto_indexing = AutoIndexer.index_result(
            result,
            vector_store=vector_store,
        )

        logger.info(
            f"[auto_indexer] indexed={result.auto_indexing['indexed']} · "
            f"reason={result.auto_indexing['reason']}"
        )

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
            ("stance", cls._step_stance),
            ("bertimbau", cls._step_bertimbau),
            ("text_features", cls._step_text_features),
            ("reputation", cls._step_reputation),
            ("aggregator", cls._step_aggregate),
            ("auto_indexer", cls._step_auto_index),
        ]

        # ── etapas de saída ainda incrementais ─────────────────────────────────
        steps_pending = [
            ("explanation", cls._step_explain),
            ("formatter", cls._step_format),
        ]

        all_steps = steps_implemented + steps_pending

        show_progress = env_flag("HIBRIA_SHOW_PIPELINE_PROGRESS", True)

        for name, step in all_steps:
            step_start = time.time()

            if show_progress:
                print(f"[pipeline] Iniciando etapa: {name}...", flush=True)

            try:
                result = step(result)
            except ExtractionError:
                raise
            except Exception as e:
                msg = f"[{name}] falhou: {type(e).__name__}: {e}"
                result.warnings.append(msg)
                logger.error(msg, exc_info=True)

            result._processing_time[name] = round(time.time() - step_start, 3)

            if show_progress:
                print(
                    f"[pipeline] Etapa concluída: {name} "
                    f"({result._processing_time[name]}s)",
                    flush=True,
                )

        result._processing_time["total"] = round(time.time() - start_time, 3)

        logger.info(
            f"[pipeline] concluído em "
            f"{result._processing_time['total']}s · "
            f"{result.claim_count} claims · "
            f"{result.evidence_count} evidências"
        )
        return result

    @staticmethod
    def _step_stance(result: PipelineResult) -> PipelineResult:
        """
        Analisa a relação entre cada claim e suas evidências recuperadas.

        Esta etapa classifica se a evidência apoia, contradiz, é neutra ou
        insuficiente em relação à claim. O resultado ainda não altera o score
        final nesta versão; ele apenas enriquece a saída do pipeline.
        """
        result.stance_results = StanceModel.analyze(
            retrieval_results=result.retrieval_results or [],
            similarity_scores=result.similarity_scores or [],
        )

        logger.info(
            f"[stance] {len(result.stance_results or [])} relações claim-evidência analisadas"
        )

        return result
