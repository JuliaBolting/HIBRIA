# =============================================================================
# retriever.py
# Etapa de Recuperação do pipeline RAG (Retrieval-Augmented Generation).
#
# Recebe claims identificados pelo claim_detector.py e busca evidências
# externas em múltiplas fontes, em camadas de prioridade:
#
#   Camada 1 — Base vetorial local     ⭐⭐⭐⭐⭐  (sempre disponível)
#   Camada 2 — Wikipedia API           ⭐⭐⭐⭐   (sem API key)
#   Camada 3 — APIs de fact-checking   ⭐⭐⭐⭐   (requer API key)
#   Camada 4 — Google/Bing Search      ⭐⭐      (requer API key, fallback)
#
# O retriever é tolerante a falhas: se uma camada falhar ou não tiver
# API key configurada, registra o motivo e continua com as demais.
# O pipeline nunca para por falha de uma fonte individual.
#
# Entrada:  list[Claim]    — saída do claim_detector.py
# Saída:    list[Evidence] — evidências rankeadas por relevância
# =============================================================================

from __future__ import annotations

import os
import time
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import requests

logger = logging.getLogger(__name__)


# =============================================================================
# Estruturas de dados
# =============================================================================

@dataclass
class Claim:
    """
    Afirmação verificável vinda do claim_detector.py.
    Campos mínimos que o retriever consome — o claim_detector pode adicionar mais.
    """
    text:       str                    # texto original da afirmação
    normalized: str                    # texto normalizado (sem stopwords, lematizado)
    entities:   list[str] = field(default_factory=list)  # entidades nomeadas
    subject:    str        = ""        # sujeito principal da afirmação
    claim_id:   str        = ""        # id único gerado pelo claim_detector


@dataclass
class Evidence:
    """
    Evidência recuperada de uma fonte externa.

    similarity: score de relevância semântica em relação ao claim [0.0, 1.0].
                Calculado por embeddings (camada 1) ou heurística TF-IDF (demais).

    stance:     posicionamento da evidência em relação ao claim.
                Preenchido pelo stance_model.py downstream.
                "supports" | "refutes" | "neutral" | None (não avaliado ainda)

    retrieval_layer: qual camada encontrou essa evidência.
                     Útil para o aggregator.py pesar fontes diferentes.
    """
    text:             str
    source:           str               # nome legível da fonte
    url:              str
    similarity:       float             # [0.0, 1.0]
    published_at:     str | None = None # ISO 8601 quando disponível
    claim_id:         str        = ""   # id do claim que gerou esta busca
    retrieval_layer:  str        = ""   # "vector_store" | "wikipedia" | "factcheck" | "web_search"
    stance:           str | None = None # preenchido pelo stance_model.py
    metadata:         dict       = field(default_factory=dict)  # dados extras da fonte


@dataclass
class RetrievalResult:
    """
    Resultado completo de uma rodada de retrieval para um claim.
    Agrupa evidências encontradas com metadados do processo.
    """
    claim:           Claim
    evidences:       list[Evidence]
    layers_used:     list[str]          # camadas que retornaram resultado
    layers_failed:   dict[str, str]     # camada → motivo da falha
    retrieval_time:  float              # segundos


# =============================================================================
# Protocol: interface que toda fonte de busca deve implementar
#
# Usar Protocol (duck typing estrutural) em vez de ABC permite adicionar
# novas fontes sem herança — basta implementar os dois métodos.
# As camadas futuras (fact-check, web search) seguem o mesmo contrato.
# =============================================================================

class EvidenceSource(Protocol):

    @property
    def name(self) -> str:
        """Nome identificador da fonte (usado em Evidence.retrieval_layer)."""
        ...

    def is_available(self) -> bool:
        """
        Verifica se a fonte está disponível para uso.
        Camadas que precisam de API key retornam False quando a key não está
        configurada — o retriever pula e registra o motivo.
        """
        ...

    def search(self, claim: Claim, top_k: int = 5) -> list[Evidence]:
        """
        Busca evidências para o claim e retorna as top_k mais relevantes.
        Nunca lança exceção — retorna lista vazia em caso de falha.
        """
        ...


# =============================================================================
# Utilitário compartilhado: similaridade TF-IDF simples
#
# Usado pelas camadas 2, 3 e 4 como aproximação de relevância quando
# não há embeddings disponíveis. A camada 1 (vector store) usa
# similaridade vetorial real via embeddings.py.
# =============================================================================

def _tfidf_similarity(query: str, text: str) -> float:
    """
    Similaridade aproximada por sobreposição de tokens normalizados.
    Rápida, sem dependências, suficiente para rankeamento inicial.
    O similarity.py downstream recalcula com embeddings reais.
    """
    import re
    from math import log

    def tokenize(s: str) -> dict[str, int]:
        tokens = re.findall(r"\b\w{3,}\b", s.lower())
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        return freq

    q_tokens = tokenize(query)
    t_tokens = tokenize(text)

    if not q_tokens or not t_tokens:
        return 0.0

    # interseção ponderada pela frequência no query
    overlap = sum(
        min(q_tokens[t], t_tokens.get(t, 0))
        for t in q_tokens
        if t in t_tokens
    )
    # normaliza pelo tamanho do query para evitar favorecer textos longos
    return round(min(overlap / len(q_tokens), 1.0), 4)


# =============================================================================
# Camada 1: Base Vetorial Local  ⭐⭐⭐⭐⭐
#
# Núcleo principal do retrieval. Documentos pré-indexados com embeddings
# semânticos — busca por similaridade cosine via FAISS ou ChromaDB.
#
# O vector_store.py gerencia a indexação e consulta.
# Esta camada apenas delega para ele.
# =============================================================================

class VectorStoreSource:
    """
    Busca na base vetorial local gerenciada pelo vector_store.py.
    Sempre disponível — não precisa de API key nem de internet.
    """

    name = "vector_store"

    def __init__(self, vector_store=None):
        """
        vector_store: instância de VectorStore (vector_store.py).
        Aceita None para permitir inicialização sem base indexada —
        is_available() retornará False e a camada será pulada.
        """
        self._store = vector_store

    def is_available(self) -> bool:
        return self._store is not None

    def search(self, claim: Claim, top_k: int = 5) -> list[Evidence]:
        """
        Delega para vector_store.query() que retorna documentos com scores.
        O texto normalizado é usado para a query — melhor match semântico.
        """
        if not self._store:
            return []

        try:
            # vector_store.query() retorna list[dict] com keys:
            # text, source, url, score, published_at, metadata
            results = self._store.query(
                query_text=claim.normalized or claim.text,
                top_k=top_k,
            )
            return [
                Evidence(
                    text             = r["text"],
                    source           = r.get("source", "Base local"),
                    url              = r.get("url", ""),
                    similarity       = float(r.get("score", 0.0)),
                    published_at     = r.get("published_at"),
                    claim_id         = claim.claim_id,
                    retrieval_layer  = self.name,
                    metadata         = r.get("metadata", {}),
                )
                for r in results
            ]
        except Exception as e:
            logger.warning(f"[vector_store] falha na busca: {e}")
            return []


# =============================================================================
# Camada 2: Wikipedia API  ⭐⭐⭐⭐
#
# Complemento factual/enciclopédico. Sem API key, sempre disponível.
# Estratégia de busca em duas etapas:
#   1. Busca por entidades nomeadas do claim (mais preciso)
#   2. Fallback: busca pelo texto completo do claim
#
# A Wikipedia API retorna resumos de artigos (intro), não o artigo completo.
# Isso é suficiente para evidências factuais básicas.
# =============================================================================

class WikipediaSource:
    """
    Busca na Wikipedia via API REST pública (sem autenticação).
    Usa as entidades nomeadas do claim para queries mais precisas.
    """

    name = "wikipedia"

    _BASE_URL    = "https://pt.wikipedia.org/api/rest_v1"
    _SEARCH_URL  = "https://pt.wikipedia.org/w/api.php"
    _HEADERS     = {
        "User-Agent": "HIBRIA-FactChecker/1.0 (TCC; educational)",
        "Accept": "application/json",
    }
    _TIMEOUT     = 8   # segundos por requisição
    _MAX_SUMMARY = 800 # caracteres do resumo a preservar

    def is_available(self) -> bool:
        # sempre disponível — não requer chave
        return True

    def _search_titles(self, query: str, limit: int = 3) -> list[str]:
        """
        Busca títulos de artigos da Wikipedia para uma query.
        Usa a Action API (opensearch) que retorna sugestões relevantes.
        """
        try:
            resp = requests.get(
                self._SEARCH_URL,
                params={
                    "action":   "opensearch",
                    "search":   query,
                    "limit":    limit,
                    "namespace": 0,         # só artigos (não discussão, usuário etc.)
                    "format":   "json",
                },
                headers=self._HEADERS,
                timeout=self._TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            # opensearch retorna [query, [títulos], [descrições], [urls]]
            return data[1] if len(data) > 1 else []
        except Exception as e:
            logger.debug(f"[wikipedia] busca de títulos falhou: {e}")
            return []

    def _get_summary(self, title: str) -> dict | None:
        """
        Recupera o resumo (introdução) de um artigo pelo título.
        Usa a REST API v1 que retorna JSON estruturado com extract.
        """
        try:
            url = f"{self._BASE_URL}/page/summary/{requests.utils.quote(title)}"
            resp = requests.get(
                url,
                headers=self._HEADERS,
                timeout=self._TIMEOUT,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.debug(f"[wikipedia] resumo de '{title}' falhou: {e}")
            return None

    def search(self, claim: Claim, top_k: int = 5) -> list[Evidence]:
        """
        Estratégia de busca em duas etapas:
          1. Usa entidades nomeadas para queries específicas (mais preciso)
          2. Fallback para o texto completo do claim normalizado

        Deduplica artigos pelo título para não retornar o mesmo artigo
        duas vezes quando entidades diferentes apontam para ele.
        """
        evidences: list[Evidence] = []
        seen_titles: set[str]     = set()

        # monta queries: entidades primeiro, texto completo como fallback
        queries = []
        if claim.entities:
            queries.extend(claim.entities[:3])  # top 3 entidades
        queries.append(claim.normalized or claim.text)

        for query in queries:
            if len(evidences) >= top_k:
                break

            titles = self._search_titles(query, limit=2)
            for title in titles:
                if title in seen_titles or len(evidences) >= top_k:
                    continue
                seen_titles.add(title)

                summary = self._get_summary(title)
                if not summary:
                    continue

                extract = summary.get("extract", "")
                if not extract or len(extract) < 50:
                    continue

                # trunca o resumo para evitar textos muito longos
                text = extract[:self._MAX_SUMMARY]
                if len(extract) > self._MAX_SUMMARY:
                    text += "..."

                similarity = _tfidf_similarity(
                    claim.normalized or claim.text,
                    text,
                )

                evidences.append(Evidence(
                    text            = text,
                    source          = "Wikipedia",
                    url             = summary.get("content_urls", {})
                                              .get("desktop", {})
                                              .get("page", ""),
                    similarity      = similarity,
                    published_at    = None,   # Wikipedia não expõe data de edição nessa API
                    claim_id        = claim.claim_id,
                    retrieval_layer = self.name,
                    metadata        = {
                        "title":       title,
                        "description": summary.get("description", ""),
                        "lang":        "pt",
                    },
                ))

                # respeita rate limit da API da Wikipedia
                time.sleep(0.2)

        return evidences


# =============================================================================
# Camada 3: APIs de Fact-Checking  ⭐⭐⭐⭐
#
# Validação especializada por agências de checagem.
# Requer API key — retorna lista vazia se não configurada.
#
# Suporte atual:
#   - Google Fact Check Tools API (GOOGLE_FACTCHECK_API_KEY)
#
# Interface pronta para adicionar:
#   - ClaimBuster API (CLAIMBUSTER_API_KEY)
#   - Full Fact API
#   - Agência Lupa / Aos Fatos (quando disponibilizarem API)
# =============================================================================

class FactCheckSource:
    """
    Busca em APIs de fact-checking especializadas.
    Plugue sua API key em variável de ambiente — sem ela, a camada é pulada.
    """

    name = "factcheck"

    # variáveis de ambiente suportadas
    _GOOGLE_KEY_ENV    = "GOOGLE_FACTCHECK_API_KEY"
    _CLAIMBUSTER_ENV   = "CLAIMBUSTER_API_KEY"

    _GOOGLE_URL        = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    _CLAIMBUSTER_URL   = "https://idir.uta.edu/claimbuster/api/v2/score/text/"
    _TIMEOUT           = 10

    def is_available(self) -> bool:
        """
        Disponível se pelo menos uma API key estiver configurada.
        """
        return bool(
            os.getenv(self._GOOGLE_KEY_ENV)
            or os.getenv(self._CLAIMBUSTER_ENV)
        )

    def _search_google_factcheck(
        self,
        claim: Claim,
        top_k: int,
    ) -> list[Evidence]:
        """
        Busca na Google Fact Check Tools API.
        Retorna checagens de agências parceiras (Lupa, Aos Fatos, Agência Pública etc.)
        """
        api_key = os.getenv(self._GOOGLE_KEY_ENV)
        if not api_key:
            return []

        try:
            resp = requests.get(
                self._GOOGLE_URL,
                params={
                    "key":            api_key,
                    "query":          claim.text,
                    "languageCode":   "pt",
                    "pageSize":       top_k,
                },
                timeout=self._TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            evidences = []
            for item in data.get("claims", []):
                # cada item pode ter múltiplas revisões de agências diferentes
                for review in item.get("claimReview", []):
                    text = (
                        f"Afirmação verificada: {item.get('text', '')}. "
                        f"Veredicto: {review.get('textualRating', '')}. "
                        f"Publicado por: {review.get('publisher', {}).get('name', '')}."
                    )
                    evidences.append(Evidence(
                        text            = text,
                        source          = review.get("publisher", {}).get("name", "Fact Check"),
                        url             = review.get("url", ""),
                        similarity      = _tfidf_similarity(
                            claim.normalized or claim.text, text
                        ),
                        published_at    = review.get("reviewDate"),
                        claim_id        = claim.claim_id,
                        retrieval_layer = self.name,
                        metadata        = {
                            "rating":    review.get("textualRating"),
                            "language":  review.get("languageCode"),
                            "provider":  "google_factcheck",
                        },
                    ))
            return evidences[:top_k]

        except Exception as e:
            logger.warning(f"[factcheck/google] falhou: {e}")
            return []

    def search(self, claim: Claim, top_k: int = 5) -> list[Evidence]:
        evidences: list[Evidence] = []

        # tenta Google Fact Check primeiro
        evidences.extend(self._search_google_factcheck(claim, top_k))

        # FUTURO: adicionar ClaimBuster, Full Fact, etc.
        # evidences.extend(self._search_claimbuster(claim, top_k - len(evidences)))

        return evidences[:top_k]


# =============================================================================
# Camada 4: Web Search (Google/Bing)  ⭐⭐
#
# Fallback em tempo real. Última camada — só acionada quando as anteriores
# não retornaram evidências suficientes.
#
# Suporte atual:
#   - Bing Search API (BING_SEARCH_API_KEY)
#   - Google Custom Search (GOOGLE_SEARCH_API_KEY + GOOGLE_CSE_ID)
#
# Sem API key, retorna lista vazia silenciosamente.
# =============================================================================

class WebSearchSource:
    """
    Fallback de busca web em tempo real.
    Acionado apenas quando as camadas superiores não trouxeram evidências.
    """

    name = "web_search"

    _BING_KEY_ENV      = "BING_SEARCH_API_KEY"
    _GOOGLE_KEY_ENV    = "GOOGLE_SEARCH_API_KEY"
    _GOOGLE_CSE_ENV    = "GOOGLE_CSE_ID"

    _BING_URL          = "https://api.bing.microsoft.com/v7.0/search"
    _GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
    _TIMEOUT           = 10

    def is_available(self) -> bool:
        return bool(
            os.getenv(self._BING_KEY_ENV)
            or (os.getenv(self._GOOGLE_KEY_ENV) and os.getenv(self._GOOGLE_CSE_ENV))
        )

    def _search_bing(self, claim: Claim, top_k: int) -> list[Evidence]:
        api_key = os.getenv(self._BING_KEY_ENV)
        if not api_key:
            return []

        try:
            resp = requests.get(
                self._BING_URL,
                headers={"Ocp-Apim-Subscription-Key": api_key},
                params={
                    "q":      f'fact check {claim.text}',
                    "count":  top_k,
                    "mkt":    "pt-BR",
                    "setLang": "pt",
                },
                timeout=self._TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            return [
                Evidence(
                    text            = r.get("snippet", ""),
                    source          = r.get("name", "Web"),
                    url             = r.get("url", ""),
                    similarity      = _tfidf_similarity(
                        claim.normalized or claim.text,
                        r.get("snippet", ""),
                    ),
                    published_at    = r.get("datePublished"),
                    claim_id        = claim.claim_id,
                    retrieval_layer = self.name,
                    metadata        = {"provider": "bing"},
                )
                for r in data.get("webPages", {}).get("value", [])
                if r.get("snippet")
            ][:top_k]

        except Exception as e:
            logger.warning(f"[web_search/bing] falhou: {e}")
            return []

    def search(self, claim: Claim, top_k: int = 5) -> list[Evidence]:
        evidences = self._search_bing(claim, top_k)

        # FUTURO: Google Custom Search como alternativa ao Bing
        # if not evidences:
        #     evidences = self._search_google(claim, top_k)

        return evidences[:top_k]


# =============================================================================
# EvidenceRetriever — orquestrador principal
#
# Executa as camadas em ordem de prioridade, agrega os resultados,
# deduplica por URL e rerankeai por similaridade.
# =============================================================================

class EvidenceRetriever:
    """
    Orquestra a busca de evidências em múltiplas fontes em camadas.

    Fluxo para cada claim:
      1. Executa as camadas disponíveis em ordem de prioridade
      2. Agrega todas as evidências encontradas
      3. Deduplica por URL
      4. Rerankeai por similaridade decrescente
      5. Retorna as top_k evidências mais relevantes

    A estratégia de parada antecipada (early stopping) evita chamar
    camadas desnecessárias quando as prioritárias já retornaram
    evidências suficientes. Camada 4 (web_search) raramente é acionada.
    """

    # threshold mínimo de similaridade para incluir uma evidência
    MIN_SIMILARITY: float = 0.05

    # número mínimo de evidências antes de considerar early stopping
    EARLY_STOP_COUNT: int = 5

    def __init__(self, vector_store=None):
        """
        Inicializa as camadas em ordem de prioridade.
        vector_store: instância de VectorStore ou None se não indexado ainda.
        """
        self._sources: list[EvidenceSource] = [
            VectorStoreSource(vector_store),  # camada 1 — ⭐⭐⭐⭐⭐
            WikipediaSource(),                # camada 2 — ⭐⭐⭐⭐
            FactCheckSource(),                # camada 3 — ⭐⭐⭐⭐ (requer key)
            WebSearchSource(),                # camada 4 — ⭐⭐     (requer key)
        ]

    def retrieve(
        self,
        claim: Claim,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Busca evidências para um único claim em todas as camadas disponíveis.

        Early stopping: se as camadas de alta prioridade já retornaram
        EARLY_STOP_COUNT evidências, camadas de baixa prioridade (web_search)
        são puladas para economizar latência e custo de API.
        """
        start_time     = time.time()
        all_evidences: list[Evidence] = []
        layers_used:   list[str]      = []
        layers_failed: dict[str, str] = {}

        for source in self._sources:
            # pula camadas indisponíveis (sem API key)
            if not source.is_available():
                layers_failed[source.name] = "não disponível (sem API key ou base local)"
                continue

            # early stopping: camadas de baixa prioridade puladas se já há evidências suficientes
            # camadas 1 e 2 sempre executam (alta prioridade); 3 e 4 podem ser puladas
            is_low_priority = source.name in ("factcheck", "web_search")
            if is_low_priority and len(all_evidences) >= self.EARLY_STOP_COUNT:
                layers_failed[source.name] = (
                    f"pulada — early stopping ({len(all_evidences)} evidências já encontradas)"
                )
                continue

            try:
                evidences = source.search(claim, top_k=top_k)
                if evidences:
                    all_evidences.extend(evidences)
                    layers_used.append(source.name)
                    logger.debug(
                        f"[{source.name}] {len(evidences)} evidências para: "
                        f"{claim.text[:60]}..."
                    )
                else:
                    layers_failed[source.name] = "sem resultados"

            except Exception as e:
                # captura exceções não tratadas dentro de cada source
                layers_failed[source.name] = f"erro inesperado: {e}"
                logger.error(f"[{source.name}] erro inesperado: {e}", exc_info=True)

        # ── pós-processamento ────────────────────────────────────────────────

        # 1. filtra evidências abaixo do threshold de similaridade
        all_evidences = [
            e for e in all_evidences
            if e.similarity >= self.MIN_SIMILARITY
        ]

        # 2. deduplica por URL (mesma fonte pode aparecer em camadas diferentes)
        seen_urls: set[str] = set()
        deduped: list[Evidence] = []
        for ev in all_evidences:
            key = ev.url or ev.text[:80]   # usa texto como fallback se não tiver URL
            if key not in seen_urls:
                seen_urls.add(key)
                deduped.append(ev)

        # 3. rerankeai por similaridade decrescente e retorna top_k
        ranked = sorted(deduped, key=lambda e: e.similarity, reverse=True)[:top_k]

        return RetrievalResult(
            claim          = claim,
            evidences      = ranked,
            layers_used    = layers_used,
            layers_failed  = layers_failed,
            retrieval_time = round(time.time() - start_time, 3),
        )

    def retrieve_batch(
        self,
        claims: list[Claim],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """
        Processa múltiplos claims em sequência.
        Adiciona delay entre claims para respeitar rate limits das APIs.

        FUTURO: processamento paralelo com asyncio para claims independentes.
        """
        results = []
        for i, claim in enumerate(claims):
            result = self.retrieve(claim, top_k=top_k)
            results.append(result)

            # delay entre claims — evita rate limiting em APIs externas
            # não aplica no último claim
            if i < len(claims) - 1 and result.layers_used:
                time.sleep(0.5)

        return results