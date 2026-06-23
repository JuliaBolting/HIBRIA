# =============================================================================
# similarity.py
# Calcula similaridade semântica entre claims e evidências.
#
# Posição no pipeline:
#   retriever.py → similarity.py → stance_model.py → aggregator.py
#
# Responsabilidades:
#   - Gerar embeddings para claims e evidências (via embeddings.py)
#   - Calcular cosine similarity entre cada par claim × evidência
#   - Retornar SimilarityResult por claim com evidências rerankeadas
#
# Por que recalcular similaridade aqui se o retriever já tem scores?
#   O retriever usa scores do FAISS (busca por query normalizada) e
#   TF-IDF para camadas 2-4. Aqui usamos embeddings do texto completo
#   da claim (não normalizado) × texto completo da evidência —
#   mais preciso para o aggregator.
#
# Entrada:  list[Claim], list[RetrievalResult]
# Saída:    list[SimilarityResult]
# =============================================================================

from __future__ import annotations
import logging
from dataclasses import dataclass, field
import os
import numpy as np

logger = logging.getLogger(__name__)

def env_float(name: str, default: float) -> float:
    """
    Lê float do .env com fallback seguro.
    """
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return float(value)
    except ValueError:
        logger.warning(
            f"[env] valor inválido para {name}={value!r}; usando padrão {default}"
        )
        return default


# =============================================================================
# Estruturas de dados
# =============================================================================

@dataclass
class EvidenceSimilarity:
    """
    Score de similaridade de uma evidência em relação ao seu claim.

    similarity_retriever: score original do retriever (FAISS/TF-IDF)
    similarity_semantic:  score recalculado aqui com embeddings completos
    similarity_final:     média ponderada dos dois (usado pelo aggregator)
    """
    evidence_text:        str
    evidence_source:      str
    evidence_url:         str
    evidence_layer:       str           # "vector_store" | "wikipedia" | ...
    source_type:          str
    trusted_source:       bool
    published_at:         str | None

    similarity_retriever: float         # score original do retriever [0, 1]
    similarity_semantic:  float         # cosine similarity via embeddings [0, 1]
    similarity_final:     float         # score combinado [0, 1]
    is_sufficient:        bool          # True se a evidência passou do corte mínimo de similaridade para ser considerada relevante

    rank:                 int           # posição no ranking (0 = mais similar)


@dataclass
class SimilarityResult:
    """
    Resultado de similaridade para um claim específico.

    claim_id:       id do claim (para cruzar com retrieval_results)
    claim_text:     texto original do claim
    top_evidence:   evidência mais similar (convenência para stance_model)
    evidences:      lista completa ordenada por similarity_final desc
    score:          score agregado do claim = max similarity_final das evidências
                    (usado pelo aggregator como score_similarity deste claim)
    has_evidence:   True se encontrou pelo menos 1 evidência acima do threshold
    """
    claim_id:     str
    claim_text:   str
    top_evidence: EvidenceSimilarity | None
    evidences:    list[EvidenceSimilarity]
    score:        float                     # [0, 1] — máximo das similaridades
    has_evidence: bool
    has_sufficient_evidence: bool


# =============================================================================
# SimilarityCalculator
# =============================================================================

class SimilarityCalculator:
    """
    Calcula similaridade semântica entre claims e evidências via embeddings.

    Usa o mesmo EmbeddingModel do vector_store.py — singleton, carregado
    uma vez na RAM e reutilizado entre chamadas.

    Arquitetura de scoring em duas camadas:
      1. similarity_retriever  — score já calculado pelo retriever (FAISS/TF-IDF)
      2. similarity_semantic   — cosine similarity com embeddings do texto completo

    similarity_final = α × semantic + (1-α) × retriever
    onde α = SEMANTIC_WEIGHT (padrão 0.7 — embeddings são mais precisos)
    """

    # peso do score semântico vs score do retriever no score final
    # 0.7 → embeddings dominam; retriever é sinal de suporte
    SEMANTIC_WEIGHT: float = env_float("HIBRIA_SIMILARITY_SEMANTIC_WEIGHT", 0.7)

    # threshold mínimo para manter uma evidência como candidata.
    # Abaixo disso, o resultado é ruído e não deve ir para stance/aggregator.
    MIN_SIMILARITY: float = env_float("HIBRIA_SIMILARITY_MIN_SCORE", 0.25)

    # threshold mínimo para considerar que a claim possui evidência suficiente.
    # Esse valor deve conversar com o retriever e com o aggregator.
    MIN_SUFFICIENT_SIMILARITY: float = env_float("HIBRIA_SIMILARITY_MIN_SUFFICIENT", 0.55)

    # máximo de evidências por claim no resultado final
    # reduz ruído para o stance_model e aggregator
    TOP_K: int = 10

    # modelo de embeddings — mesmo do vector_store para consistência
    _embedding_model = None

    @classmethod
    def _get_model(cls):
        """
        Lazy loading do EmbeddingModel.
        Singleton — carregado uma vez e reutilizado.
        """
        if cls._embedding_model is None:
            from pipeline.analysis.embeddings import EmbeddingModel
            cls._embedding_model = EmbeddingModel("multilingual-minilm")
            logger.info("[similarity] EmbeddingModel carregado")
        return cls._embedding_model

    @classmethod
    def _cosine_similarity(cls, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Cosine similarity entre dois vetores normalizados.
        Se os vetores já são L2-normalizados (como os do EmbeddingModel),
        cosine = dot product.
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    @classmethod
    def _combine_scores(
        cls,
        semantic:  float,
        retriever: float,
    ) -> float:
        """
        Combina score semântico e score do retriever em um score final.

        Por que não usar só o semântico:
          O score do retriever carrega sinal de relevância lexical (TF-IDF)
          que complementa a similaridade semântica — especialmente útil para
          claims com números e entidades específicas onde embeddings podem
          ser menos discriminativos.
        """
        combined = cls.SEMANTIC_WEIGHT * semantic + (1 - cls.SEMANTIC_WEIGHT) * retriever
        return round(min(combined, 1.0), 4)

    @classmethod
    def calculate(
        cls,
        claims,               # list[Claim] do claim_detector
        retrieval_results,    # list[RetrievalResult] do retriever
    ) -> list[SimilarityResult]:
        """
        Calcula similaridade semântica para todos os claims × evidências.

        Fluxo:
          1. Coleta todos os textos únicos (claims + evidências) para embed em batch
          2. Gera embeddings em uma única chamada (mais eficiente que N chamadas)
          3. Calcula cosine similarity para cada par claim × evidência
          4. Combina com score do retriever
          5. Filtra, ordena e retorna SimilarityResult por claim

        Batch embedding é crítico para performance:
          20 claims × 10 evidências = 200+ textos
          1 chamada batch vs 200+ chamadas individuais
        """
        if not claims or not retrieval_results:
            logger.warning("[similarity] sem claims ou evidências — pulando")
            return []

        model = cls._get_model()

        # ── passo 1: mapeia claim_id → claim e claim_id → retrieval_result ──
        claim_map      = {c.claim_id: c for c in claims}
        retrieval_map  = {r.claim.claim_id: r for r in retrieval_results}

        # ── passo 2: coleta todos os textos para embedding em batch ──────────
        # usa blocks_similarity (normalizado para comparação semântica)
        # mas preserva texto original para exibição

        claim_texts: list[str] = []   # textos dos claims para embedding
        claim_ids:   list[str] = []   # ids correspondentes

        for claim in claims:
            # usa normalized se disponível — melhor para embeddings
            text = getattr(claim, "normalized", "") or claim.text
            claim_texts.append(text)
            claim_ids.append(claim.claim_id)

        # coleta textos de evidências — deduplica por texto para não embedar duplicatas
        evidence_texts_unique: list[str] = []
        evidence_text_to_idx:  dict[str, int] = {}

        for rr in retrieval_results:
            for ev in rr.evidences:
                key = ev.text[:200]  # chave de deduplicação
                if key not in evidence_text_to_idx:
                    evidence_text_to_idx[key] = len(evidence_texts_unique)
                    evidence_texts_unique.append(ev.text)

        if not evidence_texts_unique:
            logger.warning("[similarity] sem evidências para calcular")
            return [
                SimilarityResult(
                    claim_id                 = c.claim_id,
                    claim_text               = c.text,
                    top_evidence             = None,
                    evidences                = [],
                    score                    = 0.0,
                    has_evidence             = False,
                    has_sufficient_evidence  = False,
                )
                for c in claims
            ]

        # ── passo 3: embeddings em batch ─────────────────────────────────────
        all_texts    = claim_texts + evidence_texts_unique
        all_embeddings = model.embed_batch(all_texts, normalize=True)

        n_claims              = len(claim_texts)
        claim_embeddings      = all_embeddings[:n_claims]           # shape: (n_claims, dims)
        evidence_embeddings   = all_embeddings[n_claims:]           # shape: (n_evidences, dims)

        logger.info(
            f"[similarity] {n_claims} claims × "
            f"{len(evidence_texts_unique)} evidências únicas embedadas"
        )

        # ── passo 4: calcula similaridade por claim ───────────────────────────
        results: list[SimilarityResult] = []

        for i, (claim_id, claim_emb) in enumerate(zip(claim_ids, claim_embeddings)):
            claim      = claim_map.get(claim_id)
            retrieval  = retrieval_map.get(claim_id)

            if not claim or not retrieval or not retrieval.evidences:
                results.append(SimilarityResult(
                    claim_id                 = claim_id,
                    claim_text               = claim.text if claim else "",
                    top_evidence             = None,
                    evidences                = [],
                    score                    = 0.0,
                    has_evidence             = False,
                    has_sufficient_evidence  = False,
                ))
                continue

            ev_similarities: list[EvidenceSimilarity] = []

            for ev in retrieval.evidences:
                # recupera o embedding desta evidência pelo índice deduplificado
                ev_key = ev.text[:200]
                ev_idx = evidence_text_to_idx.get(ev_key)
                if ev_idx is None:
                    continue

                ev_emb = evidence_embeddings[ev_idx]

                # cosine similarity com o embedding do claim
                semantic_score = cls._cosine_similarity(claim_emb, ev_emb)

                # score combinado
                final_score = cls._combine_scores(
                    semantic  = semantic_score,
                    retriever = ev.similarity,
                )

                if final_score < cls.MIN_SIMILARITY:
                    continue

                is_sufficient = final_score >= cls.MIN_SUFFICIENT_SIMILARITY

                ev_similarities.append(EvidenceSimilarity(
                    evidence_text        = ev.text,
                    evidence_source      = ev.source,
                    evidence_url         = ev.url,
                    evidence_layer       = ev.retrieval_layer,
                    source_type          = getattr(ev, "source_type", "external_document"),
                    trusted_source       = bool(getattr(ev, "trusted_source", False)),
                    published_at         = ev.published_at,
                    similarity_retriever = ev.similarity,
                    similarity_semantic  = round(semantic_score, 4),
                    similarity_final     = final_score,
                    is_sufficient        = is_sufficient,
                    rank                 = 0,  # atualizado abaixo
                ))

            # ordena por similarity_final decrescente e atribui rank
            ev_similarities.sort(key=lambda e: e.similarity_final, reverse=True)
            for rank, ev_sim in enumerate(ev_similarities[:cls.TOP_K]):
                ev_sim.rank = rank

            ev_similarities = ev_similarities[:cls.TOP_K]
            top_evidence    = ev_similarities[0] if ev_similarities else None

            # score do claim = similaridade da evidência mais próxima
            # representa "quão bem suportado é este claim por evidências externas"
            score = top_evidence.similarity_final if top_evidence else 0.0
            has_sufficient_evidence = any(
                evidence.is_sufficient for evidence in ev_similarities
            )

            results.append(SimilarityResult(
                claim_id                 = claim_id,
                claim_text               = claim.text,
                top_evidence             = top_evidence,
                evidences                = ev_similarities,
                score                    = round(score, 4),
                has_evidence             = bool(ev_similarities),
                has_sufficient_evidence  = has_sufficient_evidence,
            ))

        # ── log resumido ──────────────────────────────────────────────────────
        has_evidence_count = sum(1 for r in results if r.has_evidence)
        has_sufficient_count = sum(1 for r in results if r.has_sufficient_evidence)

        avg_score = (
            sum(r.score for r in results) / len(results)
            if results else 0.0
        )

        logger.info(
            f"[similarity] {has_evidence_count}/{len(results)} claims com candidato · "
            f"{has_sufficient_count}/{len(results)} com evidência suficiente · "
            f"score médio: {avg_score:.3f}"
        )

        return results


# =============================================================================
# Serialização — helper para pipeline.py
# =============================================================================

def similarity_results_to_dict(results: list[SimilarityResult]) -> list[dict]:
    """
    Serializa list[SimilarityResult] para JSON.
    Usado pelo pipeline.py no PipelineResult.to_dict().
    """
    return [
        {
            "claim_id":     r.claim_id,
            "claim_text":   r.claim_text,
            "score":        r.score,
            "has_evidence": r.has_evidence,
            "has_sufficient_evidence": r.has_sufficient_evidence,
            "top_evidence": {
                "text":                 r.top_evidence.evidence_text[:200],
                "source":               r.top_evidence.evidence_source,
                "url":                  r.top_evidence.evidence_url,
                "layer":                r.top_evidence.evidence_layer,
                "source_type":          r.top_evidence.source_type,
                "trusted_source":       r.top_evidence.trusted_source,
                "similarity_semantic":  r.top_evidence.similarity_semantic,
                "similarity_retriever": r.top_evidence.similarity_retriever,
                "similarity_final":     r.top_evidence.similarity_final,
                "is_sufficient":        r.top_evidence.is_sufficient,
            } if r.top_evidence else None,
            "evidences": [
                {
                    "rank":                  e.rank,
                    "text":                  e.evidence_text[:200],
                    "source":                e.evidence_source,
                    "url":                   e.evidence_url,
                    "layer":                 e.evidence_layer,
                    "source_type":           e.source_type,
                    "trusted_source":        e.trusted_source,
                    "published_at":          e.published_at,
                    "similarity_semantic":   e.similarity_semantic,
                    "similarity_retriever":  e.similarity_retriever,
                    "similarity_final":      e.similarity_final,
                    "is_sufficient":         e.is_sufficient,
                }
                for e in r.evidences
            ],
        }
        for r in results
    ]