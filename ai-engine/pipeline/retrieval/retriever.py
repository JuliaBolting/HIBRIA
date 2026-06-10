# =============================================================================
# Etapa de Recuperação do pipeline RAG (Retrieval-Augmented Generation).
#
# Recebe claims identificados pelo claim_detector.py e busca evidências
# externas em múltiplas fontes, em camadas de prioridade:
#
#   Camada 1 — Base vetorial local     ⭐⭐⭐⭐⭐  (sempre disponível)
#   Camada 2 — Wikipedia API           ⭐⭐⭐⭐   (sem API key)
#   Camada 3 — APIs de fact-checking   ⭐⭐⭐⭐   (requer API key)
#   Camada 4 — Google                  ⭐⭐      (requer API key, fallback)
#
# O retriever é tolerante a falhas: se uma camada falhar ou não tiver
# API key configurada, registra o motivo e continua com as demais.
# O pipeline nunca para por falha de uma fonte individual.
#
# Entrada:  list[Claim]    — saída do claim_detector.py
# Saída:    list[Evidence] — evidências rankeadas por relevância
# =============================================================================

from __future__ import annotations

from datetime import datetime
import hashlib
import logging
import os
from pathlib import Path
import re
import time
from dataclasses import dataclass, field
from typing import Protocol
from unittest import result
from urllib.parse import urlparse

from flask import json
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

    text:                 str
    source:               str                                # nome legível da fonte
    url:                  str
    similarity:           float                              # [0.0, 1.0]

    claim_id:             str = ""                           # id do claim que gerou esta busca
    evidence_id:          str = ""
    title:                str = ""
    domain:               str = ""
    published_at:         str | None = None                  # ISO 8601 quando disponível
    retrieval_layer:      str = ""                           # "vector_store" | "wikipedia" | "factcheck" | "web_search"
    source_type:          str = "external_document"
    trusted_source:       bool = False
    stance:               str | None = None                  # preenchido pelo stance_model.py
    metadata:             dict = field(default_factory=dict) # dados extras da fonte

    def __post_init__(self) -> None:
        if not self.domain:
            self.domain = extract_domain(self.url)

        if not self.evidence_id:
            self.evidence_id = make_evidence_id(
                self.claim_id,
                self.retrieval_layer,
                self.url,
                self.text,
            )

        self.similarity = round(clamp(float(self.similarity)), 4)

    def to_db_payload(self) -> dict:
        """
        Formato pronto para ser salvo futuramente em resultados_claims.
        O PostgreSQL ainda não está integrado; por isso este método só prepara
        os dados estruturados que serão usados pela API/DAO depois.
        """
        return {
            "claim_id": self.claim_id,
            "evidence_id": self.evidence_id,
            "fonte_nome": self.source,
            "dominio": self.domain,
            "url": self.url,
            "titulo": self.title,
            "trecho": self.text,
            "camada_rag": self.retrieval_layer,
            "score_rag": self.similarity,
            "tipo_fonte": self.source_type,
            "fonte_confiavel": self.trusted_source,
            "data_publicacao": self.published_at,
            "stance": self.stance,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    """Resultado de retrieval para uma claim."""

    claim: Claim
    evidences: list[Evidence]
    layers_used: list[str]
    layers_failed: dict[str, str]
    retrieval_time: float

    @property
    def rag_score(self) -> float:
        """Maior score RAG encontrado para a claim."""
        if not self.evidences:
            return 0.0
        return max(e.similarity for e in self.evidences)

    def to_db_payload(self) -> dict:
        """Resumo estruturado para persistência futura em resultados_claims."""
        return {
            "claim_id": self.claim.claim_id,
            "score_rag": round(self.rag_score, 4),
            "qtd_evidencias": len(self.evidences),
            "camadas_utilizadas": self.layers_used,
            "camadas_falhas": self.layers_failed,
            "tempo_recuperacao": self.retrieval_time,
            "evidencias": [e.to_db_payload() for e in self.evidences],
        }


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


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def make_evidence_id(claim_id: str, layer: str, url: str, text: str) -> str:
    raw = f"{claim_id}|{layer}|{url}|{text[:300]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def extract_domain(url: str) -> str:
    if not url:
        return ""

    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_env_list(name: str) -> set[str]:
    raw = os.getenv(name, "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    return raw.strip().lower() in {"1", "true", "sim", "yes", "on"}


def build_queries(claim: Claim) -> list[str]:
    """
    Monta consultas para o RAG priorizando a afirmação completa.

    A ideia é evitar buscas muito genéricas, sem depender de uma lista manual
    de palavras proibidas. Para isso, sujeito e entidades só são usados como
    consultas adicionais quando têm tamanho suficiente e mais de uma palavra.
    """
    queries: list[str] = []
    seen: set[str] = set()

    def add_query(value: str, *, min_chars: int, min_words: int) -> None:
        value = normalize_space(value)

        if not value:
            return

        words = value.split()

        if len(value) < min_chars:
            return

        if len(words) < min_words:
            return

        key = value.lower()

        if key in seen:
            return

        seen.add(key)
        queries.append(value)

    # 1. Prioridade: claim inteira em versão normalizada
    add_query(claim.normalized, min_chars=25, min_words=5)

    # 2. Fallback: claim original completa
    add_query(claim.text, min_chars=25, min_words=5)

    # 3. Sujeito só entra se for informativo o suficiente
    add_query(claim.subject, min_chars=20, min_words=3)

    # 4. Entidades só entram se forem nomes compostos ou expressões específicas
    for entity in claim.entities:
        add_query(entity, min_chars=20, min_words=3)

    return queries or [claim.text]


def _tfidf_similarity(query: str, text: str) -> float:
    """
    Similaridade aproximada por sobreposição de tokens normalizados.
    Rápida, sem dependências, suficiente para rankeamento inicial.
    O similarity.py downstream recalcula com embeddings reais.
    """
    import re

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
        min(q_tokens[t], t_tokens.get(t, 0)) for t in q_tokens if t in t_tokens
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

    def __init__(self, vector_store=None, current_url: str = ""):
        """
        vector_store: instância de VectorStore (vector_store.py).
        Aceita None para permitir inicialização sem base indexada —
        is_available() retornará False e a camada será pulada.
        """
        self._store = vector_store
        self._trusted_domains = split_env_list("HIBRIA_TRUSTED_DOMAINS")
        self._current_url = current_url

    def is_available(self) -> bool:
        return self._store is not None

    def search(self, claim: Claim, top_k: int = 5) -> list[Evidence]:
        if not self._store:
            return []

        evidences: list[Evidence] = []
        seen_ids: set[str] = set()

        try:
            for query in build_queries(claim):
                if len(evidences) >= top_k:
                    break

                results = self._store.query(
                    query_text=query,
                    top_k=top_k * 2,
                    min_similarity=float(
                        os.getenv("HIBRIA_VECTOR_MIN_SIMILARITY", "0.25")
                    ),
                )

                for result in results:
                    meta = result.get("metadata", {}) or {}
                    url = result.get("url", "")
                    source = result.get("source", "")

                    # verifica se a evidência é a mesma notícia que está sendo analisada — evita auto-evidência
                    if self._is_same_url(url):
                        continue

                    # Bases rotuladas, como Fake.Br, não devem ser usadas como evidência factual no RAG.
                    if (
                        meta.get("source_type") == "labeled_dataset"
                        or meta.get("corpus") == "fake.br"
                        or source.lower().startswith("fake.br")
                        or url.startswith("fakebr://")
                    ):
                        continue

                    doc_id = result.get("doc_id", "")

                    if doc_id in seen_ids:
                        continue

                    seen_ids.add(doc_id)

                    domain = meta.get("domain") or extract_domain(url)
                    source = (
                        result.get("source")
                        or meta.get("source")
                        or "Base vetorial local"
                    )
                    source_type = meta.get("source_type", "external_document")

                    trusted = bool(
                        meta.get("trusted_source") is True
                        or domain in self._trusted_domains
                    )

                    evidences.append(
                        Evidence(
                            text              =normalize_space(result.get("text", "")),
                            source            =source,
                            url               =url,
                            similarity        =float(result.get("score", 0.0)),
                            claim_id          =claim.claim_id,
                            title             =meta.get("title", ""),
                            domain            =domain,
                            published_at      =result.get("published_at"),
                            retrieval_layer   =self.name,
                            source_type       =source_type,
                            trusted_source    =trusted,
                            metadata          ={
                                **meta,
                                "doc_id": doc_id,
                                "base_doc_id": result.get("base_doc_id"),
                                "faiss_rank": result.get("faiss_rank"),
                                "retrieval_query": query,
                                "source_reference": "faiss_metadata",
                            },
                        )
                    )

                    if len(evidences) >= top_k:
                        break

            return evidences[:top_k]

        except Exception as exc:
            logger.warning("[vector_store] falha na busca: %s", exc)
            return []

    def _is_same_url(self, url: str) -> bool:
        """
        Evita recuperar como evidência a mesma notícia que está sendo analisada.

        Para testes controlados, é possível permitir a própria URL como evidência
        usando no .env:
            HIBRIA_ALLOW_SELF_EVIDENCE=true
        """
        if env_flag("HIBRIA_ALLOW_SELF_EVIDENCE", default=False):
            return False

        if not self._current_url or not url:
            return False

        return url.strip().rstrip("/") == self._current_url.strip().rstrip("/")


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
        return env_flag("HIBRIA_ENABLE_WIKIPEDIA", default=False)

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

                page_url = (
                    summary.get("content_urls", {})
                    .get("desktop", {})
                    .get("page", "")
                )

                evidences.append(
                    Evidence(
                        text=text,
                        source="Wikipedia",
                        url=page_url,
                        similarity=similarity,
                        published_at=None,
                        claim_id=claim.claim_id,
                        title=title,
                        domain="pt.wikipedia.org",
                        retrieval_layer=self.name,
                        source_type="encyclopedia",
                        trusted_source=False,
                        metadata={
                            "title": title,
                            "description": summary.get("description", ""),
                            "lang": "pt",
                            "source": "pt.wikipedia.org",
                            "query": query,
                        },
                    )
                )

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


class FactCheckSource(EvidenceSource):
    name = "factcheck"

    API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(self):
        self._api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip()

    def is_available(self) -> bool:
        return bool(self._api_key)

    def search(self, claim: Claim, top_k: int = 5) -> list[Evidence]:
        query = claim.normalized or claim.text

        if not query:
            return []

        params = {
            "query": query,
            "languageCode": "pt",
            "pageSize": min(top_k, 10),
            "key": self._api_key,
        }

        response = requests.get(
            self.API_URL,
            params=params,
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()
        claims = data.get("claims", [])

        evidences: list[Evidence] = []

        for item in claims:
            claim_text = item.get("text", "")
            claim_date = item.get("claimDate")
            claim_reviews = item.get("claimReview", [])

            for review in claim_reviews:
                publisher = review.get("publisher", {}) or {}

                title = review.get("title", "") or claim_text
                url = review.get("url", "")
                source = publisher.get("name", "") or "Google Fact Check"

                rating = review.get("textualRating", "")
                review_date = review.get("reviewDate")

                evidence_text = " ".join(
                    part
                    for part in [
                        claim_text,
                        f"Classificação da checagem: {rating}" if rating else "",
                        f"Título da checagem: {title}" if title else "",
                    ]
                    if part
                )

                if not evidence_text or not url:
                    continue

                evidences.append(
                    Evidence(
                        text=evidence_text,
                        source=source,
                        url=url,
                        similarity=0.75,
                        claim_id=claim.claim_id,
                        title=title,
                        published_at=review_date or claim_date,
                        retrieval_layer=self.name,
                        source_type="fact_check",
                        trusted_source=True,
                        metadata={
                            "claim_text": claim_text,
                            "rating": rating,
                            "publisher": source,
                            "review_date": review_date,
                            "claim_date": claim_date,
                        },
                    )
                )

        return evidences[:top_k]


# =============================================================================
# Camada 4: Web Search (Google/Bing)  ⭐⭐
#
# Fallback em tempo real. Última camada — só acionada quando as anteriores
# não retornaram evidências suficientes.
#
# Suporte atual:
#   - Google Custom Search (GOOGLE_SEARCH_API_KEY + GOOGLE_CSE_ID)
#
# Sem API key, retorna lista vazia silenciosamente.
# =============================================================================


class WebSearchSource(EvidenceSource):
    name = "web_search"

    API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self):
        self._api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        self._enabled = env_flag("HIBRIA_ENABLE_WEB_SEARCH", default=False)
        self._trusted_domains = split_env_list("HIBRIA_TRUSTED_DOMAINS")
        self._daily_limit = int(os.getenv("HIBRIA_WEB_SEARCH_DAILY_LIMIT", "80"))

    def is_available(self) -> bool:
        return bool(self._enabled and self._api_key)

    def _quota_file(self) -> Path:
        return Path("data/runtime/brave_search_quota.json")

    def _read_quota(self) -> dict:
        today = datetime.now().strftime("%Y-%m-%d")
        quota_file = self._quota_file()

        if not quota_file.exists():
            return {"date": today, "count": 0}

        try:
            with open(quota_file, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception:
            return {"date": today, "count": 0}

        if data.get("date") != today:
            return {"date": today, "count": 0}

        return {
            "date": today,
            "count": int(data.get("count", 0)),
        }

    def _write_quota(self, data: dict) -> None:
        quota_file = self._quota_file()
        quota_file.parent.mkdir(parents=True, exist_ok=True)

        with open(quota_file, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def _can_search(self) -> bool:
        quota = self._read_quota()
        return quota["count"] < self._daily_limit

    def _register_search(self) -> None:
        quota = self._read_quota()
        quota["count"] += 1
        self._write_quota(quota)

    def _is_trusted_domain(self, url: str) -> bool:
        domain = extract_domain(url)

        if not domain:
            return False

        if not self._trusted_domains:
            return False

        return domain in self._trusted_domains

    def _build_query(self, claim: Claim) -> str:
        query = claim.text or claim.normalized

        entities = [
            entity for entity in claim.entities if entity and len(entity.strip()) >= 4
        ]

        if entities:
            query = f"{query} {' '.join(entities[:3])}"

        return normalize_space(query)

    def search(self, claim: Claim, top_k: int = 5) -> list[Evidence]:
        if not self._can_search():
            raise RuntimeError(
                f"limite diário local de busca web atingido ({self._daily_limit})"
            )

        query = self._build_query(claim)

        if not query:
            return []

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }

        params = {
            "q": query,
            "country": "BR",
            "search_lang": "pt-br",
            "count": min(top_k, 10),
        }

        response = requests.get(
            self.API_URL,
            headers=headers,
            params=params,
            timeout=10,
        )

        self._register_search()

        response.raise_for_status()

        data = response.json()
        items = data.get("web", {}).get("results", [])

        evidences: list[Evidence] = []

        for item in items:
            title = item.get("title", "")
            url = item.get("url", "")
            snippet = item.get("description", "")

            if not url or not snippet:
                continue

            domain = extract_domain(url)
            trusted = self._is_trusted_domain(url)

            # Por enquanto, só aceitamos resultados de domínios confiáveis.
            if not trusted:
                continue

            evidence_text = normalize_space(f"{title}. {snippet}")

            evidences.append(
                Evidence(
                    text=evidence_text,
                    source=domain or "Brave Search",
                    url=url,
                    similarity=0.55,
                    claim_id=claim.claim_id,
                    title=title,
                    domain=domain,
                    retrieval_layer=self.name,
                    source_type="web",
                    trusted_source=trusted,
                    metadata={
                        "search_engine": "brave_search",
                        "query": query,
                        "snippet": snippet,
                        "trusted_domain": trusted,
                    },
                )
            )

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

    # bloqueia evidências com similaridade muito baixa, mesmo que venham de camadas confiáveis
    MIN_EVIDENCE_CHARS: int = 120

    LAYER_WEIGHT = {
        "vector_store": 1.00,
        "factcheck": 0.95,
        "wikipedia": 0.65,
        "web_search": 0.45,
    }

    def __init__(self, vector_store=None, current_url: str = ""):
        """
        Inicializa as camadas em ordem de prioridade.
        vector_store: instância de VectorStore ou None se não indexado ainda.
        current_url: URL da notícia que está sendo analisada.
        """
        self._sources: list[EvidenceSource] = [
            VectorStoreSource(
                vector_store, current_url=current_url
            ),  # camada 1 — ⭐⭐⭐⭐⭐
            FactCheckSource(),  # camada 3 — ⭐⭐⭐⭐ (requer key)
            WikipediaSource(),  # camada 2 — ⭐⭐⭐⭐
            WebSearchSource(),  # camada 4 — ⭐⭐     (requer key)
        ]

    def _rank_score(self, evidence: Evidence) -> float:
        layer_weight = self.LAYER_WEIGHT.get(evidence.retrieval_layer, 0.5)
        trusted_bonus = 0.05 if evidence.trusted_source else 0.0
        return clamp(evidence.similarity * layer_weight + trusted_bonus)

    def _deduplicate(self, evidences: list[Evidence]) -> list[Evidence]:
        seen: set[str] = set()
        deduped: list[Evidence] = []

        for evidence in evidences:
            key = evidence.url or evidence.text[:120].lower()

            if key in seen:
                continue

            seen.add(key)
            deduped.append(evidence)

        return deduped

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
        start_time = time.time()
        all_evidences: list[Evidence] = []
        layers_used: list[str] = []
        layers_failed: dict[str, str] = {}

        for source in self._sources:
            # pula camadas indisponíveis (sem API key)
            if not source.is_available():
                layers_failed[source.name] = (
                    "não disponível (sem API key ou base local)"
                )
                continue

            # early stopping: camadas de baixa prioridade puladas se já há evidências suficientes
            # camadas 1 e 2 sempre executam (alta prioridade); 3 e 4 podem ser puladas
            is_low_priority = source.name in ("wikipedia", "web_search")
            if is_low_priority and len(all_evidences) >= self.EARLY_STOP_COUNT:
                layers_failed[source.name] = (
                    f"pulada — early stopping ({len(all_evidences)} evidências já encontradas)"
                )
                continue

            try:
                evidences = source.search(claim, top_k=top_k)
            except Exception as exc:
                layers_failed[source.name] = f"erro inesperado: {exc}"
                logger.error("[%s] erro inesperado", source.name, exc_info=True)
                continue

            valid_evidences = [
                evidence
                for evidence in evidences
                if evidence.text
                and len(evidence.text.strip()) >= self.MIN_EVIDENCE_CHARS
                and evidence.similarity >= self.MIN_SIMILARITY
            ]

            if valid_evidences:
                all_evidences.extend(valid_evidences)
                layers_used.append(source.name)
            else:
                layers_failed[source.name] = "sem resultados relevantes"

        # 2. deduplica por URL (mesma fonte pode aparecer em camadas diferentes)
        all_evidences = self._deduplicate(all_evidences)

        # 3. rerankeai por similaridade decrescente e retorna top_k
        ranked = sorted(all_evidences, key=self._rank_score, reverse=True)[:top_k]

        return RetrievalResult(
            claim=claim,
            evidences=ranked,
            layers_used=layers_used,
            layers_failed=layers_failed,
            retrieval_time=round(time.time() - start_time, 3),
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
        results: list[RetrievalResult] = []
        for index, claim in enumerate(claims):
            results.append(self.retrieve(claim, top_k=top_k))

            if index < len(claims) - 1:
                if any(
                    layer in results[-1].layers_used
                    for layer in {"factcheck", "wikipedia", "web_search"}
                ):
                    time.sleep(0.3)

        return results
