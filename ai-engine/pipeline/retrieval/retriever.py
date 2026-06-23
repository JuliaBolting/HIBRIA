# =============================================================================
# Etapa de Recuperação do pipeline RAG (Retrieval-Augmented Generation).
#
# Recebe claims identificados pelo claim_detector.py e busca evidências
# externas em múltiplas fontes, em camadas de prioridade:
#
#   Camada 1 — Base vetorial local / FAISS
#   Camada 2 — Google Fact Check
#   Camada 3 — Brave Search
#   Camada 4 — Tavily Search
#   Camada 5 — Serper/SerpApi/SearchAPI
#   Camada 6 — GDELT
#   Camada 7 — IA fallback com Gemini Flash, apenas se ainda faltar evidência
#   Camada 8 — Wikipedia (contexto enciclopédico, não prova principal)
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
import json
import logging
import os
from pathlib import Path
import re
import time
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


def safe_json_response(response, source_name: str):
    """
    Converte resposta HTTP para JSON de forma segura.
    """
    if response is None:
        return None

    content_type = response.headers.get("Content-Type", "").lower()
    text = response.text or ""

    if not text.strip():
        logger.warning(f"[{source_name}] resposta vazia; camada ignorada")
        return None

    if "json" not in content_type and not text.lstrip().startswith(("{", "[")):
        preview = re.sub(r"\s+", " ", text[:500]).strip()
        logger.warning(
            f"[{source_name}] resposta não parece JSON "
            f"(status={response.status_code}, content_type={content_type!r}); "
            f"prévia={preview!r}; camada ignorada"
        )
        return None

    try:
        return response.json()
    except ValueError:
        preview = re.sub(r"\s+", " ", text[:500]).strip()
        logger.warning(
            f"[{source_name}] resposta inválida/não JSON "
            f"(status={response.status_code}); prévia={preview!r}; camada ignorada"
        )
        return None


def should_skip_http_response(response, source_name: str) -> bool:
    """
    Decide se uma resposta HTTP deve ser ignorada com segurança.
    """
    if response is None:
        return True

    status = response.status_code

    if status == 429:
        logger.warning(f"[{source_name}] limite/rate limit atingido; camada ignorada")
        return True

    if status >= 500:
        logger.warning(
            f"[{source_name}] erro temporário do servidor "
            f"(status={status}); camada ignorada"
        )
        return True

    if status >= 400:
        logger.warning(
            f"[{source_name}] requisição rejeitada "
            f"(status={status}); camada ignorada"
        )
        return True

    return False


# =============================================================================
# Estruturas de dados
# =============================================================================


@dataclass
class Claim:
    """
    Afirmação verificável vinda do claim_detector.py.
    Campos mínimos que o retriever consome — o claim_detector pode adicionar mais.
    """

    text: str  # texto original da afirmação
    normalized: str  # texto normalizado (sem stopwords, lematizado)
    entities: list[str] = field(default_factory=list)  # entidades nomeadas
    subject: str = ""  # sujeito principal da afirmação
    claim_id: str = ""  # id único gerado pelo claim_detector


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

    text: str
    source: str  # nome legível da fonte
    url: str
    similarity: float  # [0.0, 1.0]

    claim_id: str = ""  # id do claim que gerou esta busca
    evidence_id: str = ""
    title: str = ""
    domain: str = ""
    published_at: str | None = None  # ISO 8601 quando disponível
    retrieval_layer: str = (
        ""  # "vector_store" | "factcheck" | "gdelt" | "newsapi" | "web_search" | "wikipedia" | "ai_fallback"
    )
    source_type: str = "external_document"
    trusted_source: bool = False
    stance: str | None = None  # preenchido pelo stance_model.py
    metadata: dict = field(default_factory=dict)  # dados extras da fonte

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

    def __init__(
        self, vector_store=None, current_url: str = "", document_context: str = ""
    ):
        """
        vector_store: instância de VectorStore (vector_store.py).
        Aceita None para permitir inicialização sem base indexada —
        is_available() retornará False e a camada será pulada.
        """
        self._store = vector_store
        self._trusted_domains = split_env_list("HIBRIA_TRUSTED_DOMAINS")
        self._current_url = current_url
        self._document_context = normalize_space(document_context)

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
                            text=normalize_space(result.get("text", "")),
                            source=source,
                            url=url,
                            similarity=float(result.get("score", 0.0)),
                            claim_id=claim.claim_id,
                            title=meta.get("title", ""),
                            domain=domain,
                            published_at=result.get("published_at"),
                            retrieval_layer=self.name,
                            source_type=source_type,
                            trusted_source=trusted,
                            metadata={
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

    _BASE_URL = "https://pt.wikipedia.org/api/rest_v1"
    _SEARCH_URL = "https://pt.wikipedia.org/w/api.php"
    _HEADERS = {
        "User-Agent": "HIBRIA-FactChecker/1.0 (TCC; educational)",
        "Accept": "application/json",
    }
    _TIMEOUT = 8  # segundos por requisição
    _MAX_SUMMARY = 800  # caracteres do resumo a preservar

    def __init__(self):
        self._enabled = env_flag("HIBRIA_ENABLE_WIKIPEDIA", default=False)
        self._rate_limited_until = 0.0

    def is_available(self) -> bool:
        return self._enabled and time.time() >= self._rate_limited_until

    def _search_titles(self, query: str, limit: int = 3) -> list[str]:
        """
        Busca títulos de artigos da Wikipedia para uma query.
        Usa a Action API (opensearch) que retorna sugestões relevantes.
        """
        try:
            resp = requests.get(
                self._SEARCH_URL,
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": limit,
                    "namespace": 0,  # só artigos (não discussão, usuário etc.)
                    "format": "json",
                },
                headers=self._HEADERS,
                timeout=self._TIMEOUT,
            )

            if resp.status_code == 429:
                self._rate_limited_until = time.time() + 10 * 60
                logger.warning(
                    "[wikipedia] limite/rate limit atingido; camada pausada nesta execução"
                )
                return []

            if should_skip_http_response(resp, "wikipedia"):
                return []
            data = safe_json_response(resp, "wikipedia")

            if not data:
                return []
            # opensearch retorna [query, [títulos], [descrições], [urls]]
            return data[1] if len(data) > 1 else []
        except Exception as e:
            logger.debug(f"[wikipedia] busca de títulos falhou: {e}")
            return []

    def _get_summary(self, title: str) -> dict | None:
        """
        Recupera o resumo de um artigo pelo título.
        Usa JSON seguro para evitar erro quando a Wikipedia retorna HTML, vazio ou erro.
        """
        try:
            url = f"{self._BASE_URL}/page/summary/{requests.utils.quote(title)}"
            resp = requests.get(url, headers=self._HEADERS, timeout=self._TIMEOUT)

            if resp.status_code == 404:
                return None

            if resp.status_code == 429:
                self._rate_limited_until = time.time() + 10 * 60
                logger.warning(
                    "[wikipedia] limite/rate limit atingido; camada pausada nesta execução"
                )
                return None

            if should_skip_http_response(resp, "wikipedia"):
                return None

            data = safe_json_response(resp, "wikipedia")

            if not data:
                return None

            return data

        except requests.RequestException as exc:
            logger.debug(f"[wikipedia] resumo de '{title}' falhou: {exc}")
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
        seen_titles: set[str] = set()

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
                text = extract[: self._MAX_SUMMARY]
                if len(extract) > self._MAX_SUMMARY:
                    text += "..."

                similarity = _tfidf_similarity(
                    claim.normalized or claim.text,
                    text,
                )

                page_url = (
                    summary.get("content_urls", {}).get("desktop", {}).get("page", "")
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
        return bool(
            self._api_key
            and self._api_key.lower() not in {"sua_chave_aqui", "your_key_here"}
        )

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

        try:
            response = requests.get(self.API_URL, params=params, timeout=10)
        except requests.RequestException as exc:
            logger.warning(f"[factcheck] falha de conexão; camada ignorada: {exc}")
            return []

        if should_skip_http_response(response, "factcheck"):
            return []

        data = safe_json_response(response, "factcheck")

        if not data:
            return []

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


class TavilySearchSource(EvidenceSource):
    name = "tavily_search"

    API_URL = "https://api.tavily.com/search"

    def __init__(self, current_url: str = "", document_context: str = ""):
        self._api_key = os.getenv("TAVILY_API_KEY", "").strip()
        self._enabled = env_flag("HIBRIA_ENABLE_TAVILY", default=False)
        self._trusted_domains = split_env_list("HIBRIA_TRUSTED_DOMAINS")
        self._daily_limit = env_int("HIBRIA_TAVILY_DAILY_LIMIT", 80)
        self._max_results = env_int("HIBRIA_TAVILY_MAX_RESULTS", 5)
        self._sleep_seconds = env_float("HIBRIA_TAVILY_SLEEP_SECONDS", 0.8)
        self._current_url = current_url
        self._document_context = normalize_space(document_context)
        self._rate_limited_until = 0.0

    def is_available(self) -> bool:
        return bool(
            self._enabled
            and self._api_key
            and self._api_key.lower() not in {"sua_chave_aqui", "your_key_here"}
            and time.time() >= self._rate_limited_until
        )

    def _quota_file(self) -> Path:
        return Path("data/runtime/tavily_quota.json")

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

        return {"date": today, "count": int(data.get("count", 0))}

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

    def _is_same_url(self, url: str) -> bool:
        if env_flag("HIBRIA_ALLOW_SELF_EVIDENCE", default=False):
            return False

        if not self._current_url or not url:
            return False

        return url.strip().rstrip("/") == self._current_url.strip().rstrip("/")

    def _is_trusted_domain(self, url: str) -> bool:
        domain = extract_domain(url)
        return bool(
            domain and self._trusted_domains and domain in self._trusted_domains
        )

    def _build_query(self, claim: Claim) -> str:
        parts: list[str] = []

        if self._document_context:
            parts.append(" ".join(self._document_context.split()[:12]))

        if claim.entities:
            parts.append(" ".join(claim.entities[:4]))

        if claim.subject:
            parts.append(claim.subject)

        text = normalize_space(claim.text or claim.normalized)
        words = text.split()

        if len(words) > 16:
            text = " ".join(words[:16])

        parts.append(text)

        seen: set[str] = set()
        clean_parts: list[str] = []

        for part in parts:
            part = normalize_space(part)
            key = part.lower()

            if not part or key in seen:
                continue

            seen.add(key)
            clean_parts.append(part)

        return " ".join(clean_parts)[:380]

    def search(self, claim: Claim, top_k: int = 5) -> list[Evidence]:
        if not self._can_search():
            self._rate_limited_until = time.time() + 10 * 60
            logger.warning(
                "[tavily_search] limite diário local atingido; camada pausada nesta execução"
            )
            return []

        query = self._build_query(claim)

        if not query:
            return []

        payload = {
            "query": query,
            "search_depth": os.getenv("HIBRIA_TAVILY_SEARCH_DEPTH", "basic"),
            "max_results": min(top_k, self._max_results),
            "include_answer": False,
            "include_raw_content": False,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=15,
            )
        except requests.RequestException as exc:
            self._rate_limited_until = time.time() + 5 * 60
            logger.warning(
                f"[tavily_search] falha de conexão/timeout; camada pausada: {exc}"
            )
            return []

        self._register_search()
        time.sleep(self._sleep_seconds)

        if response.status_code == 429:
            self._rate_limited_until = time.time() + 10 * 60
            logger.warning(
                "[tavily_search] limite/rate limit atingido; camada pausada nesta execução"
            )
            return []

        if should_skip_http_response(response, "tavily_search"):
            return []

        data = safe_json_response(response, "tavily_search")

        if not data:
            return []

        evidences: list[Evidence] = []
        seen_urls: set[str] = set()

        for item in data.get("results", []) or []:
            url = item.get("url", "")
            title = normalize_space(item.get("title", ""))
            content = normalize_space(item.get("content", ""))

            if not url or not content or url in seen_urls:
                continue

            if self._is_same_url(url):
                continue

            seen_urls.add(url)

            domain = extract_domain(url)
            trusted = self._is_trusted_domain(url)
            evidence_text = normalize_space(f"{title}. {content}")

            similarity = max(
                0.55,
                _tfidf_similarity(claim.normalized or claim.text, evidence_text),
            )

            evidences.append(
                Evidence(
                    text=evidence_text,
                    source=domain or "Tavily Search",
                    url=url,
                    similarity=similarity,
                    claim_id=claim.claim_id,
                    title=title,
                    domain=domain,
                    retrieval_layer=self.name,
                    source_type="web",
                    trusted_source=trusted,
                    metadata={
                        "provider": "tavily",
                        "query": query,
                        "tavily_score": item.get("score"),
                        "trusted_domain": trusted,
                    },
                )
            )

            if len(evidences) >= top_k:
                break

        return evidences[:top_k]


class SerpSearchSource(EvidenceSource):
    """
    Camada de busca SERP.

    Suporta três provedores, em ordem de prioridade:
      1. Serper.dev, quando HIBRIA_ENABLE_SERPER=true
      2. SerpApi, quando HIBRIA_ENABLE_SERPAPI=true
      3. SearchAPI.io, quando HIBRIA_ENABLE_SEARCHAPI=true
    """

    name = "serp_search"

    SERPER_URL = "https://google.serper.dev/search"
    SERPAPI_URL = "https://serpapi.com/search"
    SEARCHAPI_URL = "https://www.searchapi.io/api/v1/search"

    def __init__(self, current_url: str = "", document_context: str = ""):
        self._serper_enabled = env_flag("HIBRIA_ENABLE_SERPER", default=False)
        self._serpapi_enabled = env_flag("HIBRIA_ENABLE_SERPAPI", default=False)
        self._searchapi_enabled = env_flag("HIBRIA_ENABLE_SEARCHAPI", default=False)

        self._serper_key = os.getenv("SERPER_API_KEY", "").strip()
        self._serpapi_key = os.getenv("SERPAPI_API_KEY", "").strip()
        self._searchapi_key = os.getenv("SEARCHAPI_API_KEY", "").strip()

        self._trusted_domains = split_env_list("HIBRIA_TRUSTED_DOMAINS")
        self._daily_limit = env_int("HIBRIA_SERP_SEARCH_DAILY_LIMIT", 80)
        self._sleep_seconds = env_float("HIBRIA_SERP_SEARCH_SLEEP_SECONDS", 0.8)
        self._current_url = current_url
        self._document_context = normalize_space(document_context)
        self._rate_limited_until = 0.0

    def is_available(self) -> bool:
        serper_available = bool(
            self._serper_enabled
            and self._serper_key
            and self._serper_key.lower()
            not in {"", "sua_chave_aqui", "your_key_here", "sua_chave_serper_aqui"}
        )

        serpapi_available = bool(
            self._serpapi_enabled
            and self._serpapi_key
            and self._serpapi_key.lower() not in {"sua_chave_aqui", "your_key_here"}
        )

        searchapi_available = bool(
            self._searchapi_enabled
            and self._searchapi_key
            and self._searchapi_key.lower() not in {"sua_chave_aqui", "your_key_here"}
        )

        return (
            serper_available or serpapi_available or searchapi_available
        ) and time.time() >= self._rate_limited_until

    def _provider(self) -> str:
        if self._serper_enabled and self._serper_key:
            return "serper"

        if self._serpapi_enabled and self._serpapi_key:
            return "serpapi"

        return "searchapi"

    def _quota_file(self) -> Path:
        return Path("data/runtime/serp_search_quota.json")

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

        return {"date": today, "count": int(data.get("count", 0))}

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

    def _is_same_url(self, url: str) -> bool:
        if env_flag("HIBRIA_ALLOW_SELF_EVIDENCE", default=False):
            return False

        if not self._current_url or not url:
            return False

        return url.strip().rstrip("/") == self._current_url.strip().rstrip("/")

    def _is_trusted_domain(self, url: str) -> bool:
        domain = extract_domain(url)
        return bool(
            domain and self._trusted_domains and domain in self._trusted_domains
        )

    def _build_query(self, claim: Claim) -> str:
        parts: list[str] = []

        if self._document_context:
            parts.append(" ".join(self._document_context.split()[:12]))

        if claim.entities:
            parts.append(" ".join(claim.entities[:4]))

        if claim.subject:
            parts.append(claim.subject)

        text = normalize_space(claim.text or claim.normalized)
        words = text.split()

        if len(words) > 16:
            text = " ".join(words[:16])

        parts.append(text)

        seen: set[str] = set()
        clean_parts: list[str] = []

        for part in parts:
            part = normalize_space(part)
            key = part.lower()

            if not part or key in seen:
                continue

            seen.add(key)
            clean_parts.append(part)

        return " ".join(clean_parts)[:220]

    def _request(self, query: str, top_k: int):
        provider = self._provider()

        if provider == "serper":
            payload = {
                "q": query,
                "gl": "br",
                "hl": "pt-br",
                "num": min(top_k, 10),
            }
            headers = {
                "X-API-KEY": self._serper_key,
                "Content-Type": "application/json",
            }
            source_name = "serper"
            try:
                response = requests.post(
                    self.SERPER_URL,
                    headers=headers,
                    json=payload,
                    timeout=15,
                )
            except requests.RequestException as exc:
                self._rate_limited_until = time.time() + 5 * 60
                logger.warning(
                    f"[{source_name}] falha de conexão/timeout; camada pausada: {exc}"
                )
                return None, source_name

            return response, source_name

        if provider == "serpapi":
            params = {
                "engine": "google",
                "q": query,
                "api_key": self._serpapi_key,
                "gl": "br",
                "hl": "pt-br",
                "num": min(top_k, 10),
            }
            source_name = "serpapi"
            url = self.SERPAPI_URL
        else:
            params = {
                "engine": "google",
                "q": query,
                "api_key": self._searchapi_key,
                "gl": "br",
                "hl": "pt-br",
                "num": min(top_k, 10),
            }
            source_name = "searchapi"
            url = self.SEARCHAPI_URL

        try:
            response = requests.get(url, params=params, timeout=15)
        except requests.RequestException as exc:
            self._rate_limited_until = time.time() + 5 * 60
            logger.warning(
                f"[{source_name}] falha de conexão/timeout; camada pausada: {exc}"
            )
            return None, source_name

        return response, source_name

    @staticmethod
    def _items_from_response(data: dict, source_name: str) -> list[dict]:
        if source_name == "serper":
            return data.get("organic", []) or []

        return data.get("organic_results", []) or []

    def search(self, claim: Claim, top_k: int = 5) -> list[Evidence]:
        if not self._can_search():
            self._rate_limited_until = time.time() + 10 * 60
            logger.warning(
                "[serp_search] limite diário local atingido; camada pausada nesta execução"
            )
            return []

        query = self._build_query(claim)

        if not query:
            return []

        response, source_name = self._request(query, top_k=top_k)

        if response is None:
            return []

        self._register_search()
        time.sleep(self._sleep_seconds)

        if response.status_code == 429:
            self._rate_limited_until = time.time() + 10 * 60
            logger.warning(
                f"[{source_name}] limite/rate limit atingido; camada pausada nesta execução"
            )
            return []

        if should_skip_http_response(response, source_name):
            return []

        data = safe_json_response(response, source_name)

        if not data:
            return []

        items = self._items_from_response(data, source_name)
        evidences: list[Evidence] = []
        seen_urls: set[str] = set()

        for item in items:
            title = normalize_space(item.get("title", ""))
            url = item.get("link") or item.get("url") or ""
            snippet = normalize_space(item.get("snippet", ""))

            if not url or not snippet or url in seen_urls:
                continue

            if self._is_same_url(url):
                continue

            seen_urls.add(url)

            domain = extract_domain(url)
            trusted = self._is_trusted_domain(url)
            evidence_text = normalize_space(f"{title}. {snippet}")

            similarity = max(
                0.55,
                _tfidf_similarity(claim.normalized or claim.text, evidence_text),
            )

            evidences.append(
                Evidence(
                    text=evidence_text,
                    source=domain or source_name,
                    url=url,
                    similarity=similarity,
                    claim_id=claim.claim_id,
                    title=title,
                    domain=domain,
                    retrieval_layer=self.name,
                    source_type="web",
                    trusted_source=trusted,
                    metadata={
                        "provider": source_name,
                        "query": query,
                        "snippet": snippet,
                        "position": item.get("position"),
                        "trusted_domain": trusted,
                    },
                )
            )

            if len(evidences) >= top_k:
                break

        return evidences[:top_k]


# =============================================================================
# Camada 3: GDELT  ⭐⭐⭐⭐
#
# Busca notícias atuais em uma base jornalística aberta e gratuita.
# Não substitui a validação por similarity + stance; apenas recupera
# candidatos externos para a claim.
# =============================================================================


class GdeltSource(EvidenceSource):
    name = "gdelt"

    API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, document_context: str = ""):
        self._enabled = env_flag("HIBRIA_ENABLE_GDELT", default=False)
        self._trusted_domains = split_env_list("HIBRIA_TRUSTED_DOMAINS")
        self._document_context = normalize_space(document_context)
        self._daily_limit = env_int("HIBRIA_GDELT_DAILY_LIMIT", 40)
        self._max_queries_per_claim = env_int("HIBRIA_GDELT_MAX_QUERIES_PER_CLAIM", 1)
        self._sleep_seconds = env_float("HIBRIA_GDELT_SLEEP_SECONDS", 1.2)
        self._rate_limited_until = 0.0

    def is_available(self) -> bool:
        return self._enabled and time.time() >= self._rate_limited_until

    def _quota_file(self) -> Path:
        return Path("data/runtime/gdelt_quota.json")

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

        return {"date": today, "count": int(data.get("count", 0))}

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

    @staticmethod
    def _sanitize_query(value: str, max_words: int = 14) -> str:
        """
        Sanitiza a query do GDELT.

        O GDELT rejeita consultas com termos muito curtos, como "de", "e" ou "no".
        Por isso removemos tokens com menos de 3 caracteres antes da requisição.
        """
        value = normalize_space(value)
        value = re.sub(r"[“”\"'’‘()\[\]{}:;,.!?|/\\]", " ", value)
        value = normalize_space(value)

        words = [word for word in value.split() if len(word) >= 3]

        if len(words) > max_words:
            words = words[:max_words]

        return " ".join(words)

    def _build_gdelt_queries(self, claim: Claim) -> list[str]:
        candidates: list[str] = []

        if self._document_context:
            candidates.append(self._document_context)

        if claim.entities:
            candidates.append(" ".join(claim.entities[:4]))

        if claim.subject:
            candidates.append(claim.subject)

        candidates.append(claim.text or claim.normalized)

        queries: list[str] = []
        seen: set[str] = set()

        for candidate in candidates:
            query = self._sanitize_query(candidate)

            if len(query) < 12:
                continue

            key = query.lower()

            if key in seen:
                continue

            seen.add(key)
            queries.append(query)

        return queries[: max(1, self._max_queries_per_claim)]

    @staticmethod
    def _safe_json_response(response) -> dict | None:
        if response is None:
            return None

        content_type = response.headers.get("Content-Type", "").lower()
        text = response.text or ""

        if not text.strip():
            logger.warning("[gdelt] resposta vazia; camada pausada nesta execução")
            return None

        if "json" not in content_type and not text.lstrip().startswith(("{", "[")):
            preview = re.sub(r"\s+", " ", text[:500]).strip()
            logger.warning(
                "[gdelt] resposta não parece JSON "
                f"(status={response.status_code}, content_type={content_type!r}); "
                f"prévia={preview!r}; camada pausada nesta execução"
            )
            return None

        try:
            return response.json()
        except ValueError:
            preview = re.sub(r"\s+", " ", text[:500]).strip()
            logger.warning(
                "[gdelt] resposta inválida/não JSON "
                f"(status={response.status_code}); prévia={preview!r}; camada pausada nesta execução"
            )
            return None

    def search(self, claim: Claim, top_k: int = 5) -> list[Evidence]:
        if not self._can_search():
            self._rate_limited_until = time.time() + 10 * 60
            logger.warning(
                "[gdelt] limite diário local atingido; camada pausada nesta execução"
            )
            return []

        evidences: list[Evidence] = []
        seen_urls: set[str] = set()

        for query in self._build_gdelt_queries(claim):
            if len(evidences) >= top_k or not self._can_search():
                break

            params = {
                "query": query,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": min(top_k, 5),
                "sort": "HybridRel",
            }

            try:
                response = requests.get(self.API_URL, params=params, timeout=8)
            except requests.RequestException as exc:
                self._rate_limited_until = time.time() + 5 * 60
                logger.warning(
                    f"[gdelt] falha de conexão/timeout; camada pausada: {exc}"
                )
                return []

            self._register_search()

            if response.status_code == 429:
                self._rate_limited_until = time.time() + 10 * 60
                logger.warning(
                    "[gdelt] limite/rate limit atingido; camada pausada nesta execução"
                )
                return []

            if response.status_code >= 500:
                self._rate_limited_until = time.time() + 5 * 60
                logger.warning(
                    f"[gdelt] erro temporário do servidor "
                    f"(status={response.status_code}); camada pausada nesta execução"
                )
                return []

            if response.status_code >= 400:
                logger.warning(
                    f"[gdelt] requisição rejeitada "
                    f"(status={response.status_code}); camada ignorada"
                )
                return []

            data = self._safe_json_response(response)

            if not data:
                # Resposta inválida/HTML do GDELT não deve travar o restante do RAG.
                # Apenas abandona esta camada e deixa o retriever seguir para IA/Wikipedia.
                return []

            for item in data.get("articles", []):
                url = item.get("url", "")
                title = normalize_space(item.get("title", ""))

                if not url or not title or url in seen_urls:
                    continue

                seen_urls.add(url)
                domain = extract_domain(url) or item.get("domain", "")
                source = item.get("source", "") or domain or "GDELT"
                published_at = item.get("seendate")

                evidence_text = normalize_space(
                    f"{title}. Fonte: {source}. Publicado/observado em: {published_at or 'data não informada'}."
                )

                similarity = max(
                    0.55,
                    _tfidf_similarity(claim.normalized or claim.text, evidence_text),
                )

                evidences.append(
                    Evidence(
                        text=evidence_text,
                        source=source,
                        url=url,
                        similarity=similarity,
                        claim_id=claim.claim_id,
                        title=title,
                        domain=domain,
                        published_at=published_at,
                        retrieval_layer=self.name,
                        source_type="news",
                        trusted_source=domain in self._trusted_domains,
                        metadata={
                            "provider": "gdelt_doc_2",
                            "query": query,
                            "language": item.get("language"),
                            "source_country": item.get("sourcecountry"),
                            "social_image": item.get("socialimage"),
                        },
                    )
                )

                if len(evidences) >= top_k:
                    break

            time.sleep(self._sleep_seconds)

        return evidences[:top_k]


# =============================================================================
# Camada 4: NewsAPI  ⭐⭐⭐⭐
#
# Busca complementar em notícias. No plano gratuito, é indicada para
# desenvolvimento/testes e pode ter atraso, por isso não substitui GDELT.
# =============================================================================


class NewsApiSource(EvidenceSource):
    name = "newsapi"

    API_URL = "https://newsapi.org/v2/everything"

    def __init__(self, document_context: str = ""):
        self._api_key = os.getenv("NEWSAPI_KEY", "").strip()
        self._enabled = env_flag("HIBRIA_ENABLE_NEWSAPI", default=False)
        self._trusted_domains = split_env_list("HIBRIA_TRUSTED_DOMAINS")
        self._daily_limit = env_int("HIBRIA_NEWSAPI_DAILY_LIMIT", 80)
        self._document_context = normalize_space(document_context)
        self._rate_limited_until = 0.0

    def is_available(self) -> bool:
        return bool(
            self._enabled
            and self._api_key
            and self._api_key.lower() not in {"sua_chave_aqui", "your_key_here"}
            and time.time() >= self._rate_limited_until
        )

    def _quota_file(self) -> Path:
        return Path("data/runtime/newsapi_quota.json")

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

        return {"date": today, "count": int(data.get("count", 0))}

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

    def search(self, claim: Claim, top_k: int = 5) -> list[Evidence]:
        if not self._can_search():
            self._rate_limited_until = time.time() + 10 * 60
            logger.warning(
                "[newsapi] limite diário local atingido; camada pausada nesta execução"
            )
            return []

        evidences: list[Evidence] = []
        seen_urls: set[str] = set()
        queries = build_queries(claim)

        if self._document_context:
            contextual_query = normalize_space(
                f"{self._document_context} {claim.subject or claim.text}"
            )
            queries = [contextual_query[:220]] + queries

        for query in queries:
            if len(evidences) >= top_k or not self._can_search():
                break

            params = {
                "q": query,
                "language": "pt",
                "sortBy": "relevancy",
                "pageSize": min(top_k, 10),
                "apiKey": self._api_key,
            }

            try:
                response = requests.get(self.API_URL, params=params, timeout=10)
            except requests.RequestException as exc:
                self._rate_limited_until = time.time() + 5 * 60
                logger.warning(
                    f"[newsapi] falha de conexão/timeout; camada pausada: {exc}"
                )
                return []

            if response.status_code == 429:
                self._rate_limited_until = time.time() + 10 * 60
                logger.warning(
                    "[newsapi] limite/rate limit atingido; camada pausada nesta execução"
                )
                return []

            if should_skip_http_response(response, "newsapi"):
                return []

            data = safe_json_response(response, "newsapi")

            if not data:
                return []

            self._register_search()

            for item in data.get("articles", []):
                url = item.get("url", "")
                title = normalize_space(item.get("title", ""))
                description = normalize_space(item.get("description", ""))
                content = normalize_space(item.get("content", ""))

                if not url or url in seen_urls:
                    continue

                evidence_text = normalize_space(
                    ". ".join(part for part in [title, description, content] if part)
                )

                if not evidence_text:
                    continue

                seen_urls.add(url)
                domain = extract_domain(url)
                source_info = item.get("source", {}) or {}
                source = source_info.get("name") or domain or "NewsAPI"
                published_at = item.get("publishedAt")

                similarity = max(
                    0.55,
                    _tfidf_similarity(claim.normalized or claim.text, evidence_text),
                )

                evidences.append(
                    Evidence(
                        text=evidence_text,
                        source=source,
                        url=url,
                        similarity=similarity,
                        claim_id=claim.claim_id,
                        title=title,
                        domain=domain,
                        published_at=published_at,
                        retrieval_layer=self.name,
                        source_type="news",
                        trusted_source=domain in self._trusted_domains,
                        metadata={
                            "provider": "newsapi",
                            "query": query,
                            "author": item.get("author"),
                            "source_id": source_info.get("id"),
                        },
                    )
                )

                if len(evidences) >= top_k:
                    break

        return evidences[:top_k]


# =============================================================================
# Camada 5: Web Search (Brave / Tavily / SerpApi)  ⭐⭐
#
# Fallback web complementar. É acionada apenas quando as camadas anteriores
# ainda não retornaram evidência boa o suficiente.
#
# Suporte atual:
#   - Brave Search API (BRAVE_SEARCH_API_KEY)
#
# Sem API key, retorna lista vazia silenciosamente.
# =============================================================================


class WebSearchSource(EvidenceSource):
    name = "web_search"

    API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, current_url: str = "", document_context: str = ""):
        self._api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        self._enabled = env_flag("HIBRIA_ENABLE_WEB_SEARCH", default=False)
        self._trusted_domains = split_env_list("HIBRIA_TRUSTED_DOMAINS")
        self._daily_limit = env_int("HIBRIA_WEB_SEARCH_DAILY_LIMIT", 80)
        self._current_url = current_url
        self._document_context = normalize_space(document_context)
        self._rate_limited_until = 0.0

    def is_available(self) -> bool:
        return bool(
            self._enabled
            and self._api_key
            and self._api_key.lower() not in {"sua_chave_aqui", "your_key_here"}
            and time.time() >= self._rate_limited_until
        )

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

    def _is_same_url(self, url: str) -> bool:
        """
        Evita recuperar como evidência a mesma notícia que está sendo analisada.
        """
        if env_flag("HIBRIA_ALLOW_SELF_EVIDENCE", default=False):
            return False

        if not self._current_url or not url:
            return False

        return url.strip().rstrip("/") == self._current_url.strip().rstrip("/")

    def _build_query(self, claim: Claim) -> str:
        """
        Monta uma query curta para o Brave.

        O Brave retorna 422 quando a consulta fica muito longa ou muito carregada
        de pontuação. Por isso priorizamos contexto, entidades, sujeito e uma
        versão reduzida da claim.
        """
        parts: list[str] = []

        if self._document_context:
            context_words = self._document_context.split()

            if len(context_words) > 12:
                context = " ".join(context_words[:12])
            else:
                context = self._document_context

            parts.append(context)

        entities = [
            normalize_space(entity)
            for entity in getattr(claim, "entities", [])
            if entity and len(entity.strip()) >= 4
        ]

        parts.extend(entities[:4])

        subject = normalize_space(getattr(claim, "subject", ""))
        if subject and len(subject) >= 4:
            parts.append(subject)

        text = normalize_space(claim.text or claim.normalized)
        text = re.sub(r"[“”\"'’‘()\[\]{}:;]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        words = text.split()
        if len(words) > 14:
            text = " ".join(words[:14])

        parts.append(text)

        seen: set[str] = set()
        clean_parts: list[str] = []

        for part in parts:
            part = normalize_space(part)
            key = part.lower()

            if not part or key in seen:
                continue

            seen.add(key)
            clean_parts.append(part)

        query = " ".join(clean_parts)

        return query[:220]

    def search(self, claim: Claim, top_k: int = 5) -> list[Evidence]:
        if not self._can_search():
            self._rate_limited_until = time.time() + 10 * 60
            logger.warning(
                "[web_search] limite diário local atingido; camada pausada nesta execução"
            )
            return []

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

        try:
            response = requests.get(
                self.API_URL,
                headers=headers,
                params=params,
                timeout=10,
            )
        except requests.RequestException as exc:
            self._rate_limited_until = time.time() + 5 * 60
            logger.warning(
                f"[web_search] falha de conexão/timeout; camada pausada: {exc}"
            )
            return []

        self._register_search()

        if response.status_code == 429:
            self._rate_limited_until = time.time() + 10 * 60
            logger.warning(
                "[web_search] limite/rate limit atingido; camada pausada nesta execução"
            )
            return []

        if response.status_code == 422:
            logger.warning(
                "[web_search] query rejeitada pelo Brave; consulta longa ou inválida"
            )
            return []

        if should_skip_http_response(response, "web_search"):
            return []

        data = safe_json_response(response, "web_search")

        if not data:
            return []

        items = data.get("web", {}).get("results", [])
        evidences: list[Evidence] = []

        for item in items:
            title = item.get("title", "")
            url = item.get("url", "")
            snippet = item.get("description", "")

            if not url or not snippet:
                continue

            if self._is_same_url(url):
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
# Camada 7: IA fallback com Gemini Flash  ⭐
#
# Último recurso de recuperação. Só deve ser acionado quando as camadas
# anteriores não encontrarem evidência externa relevante. A IA não decide a
# veracidade: ela apenas tenta devolver uma fonte externa verificável.
# A URL retornada é validada antes de virar Evidence.
# =============================================================================


class AIFallbackSource(EvidenceSource):
    name = "ai_fallback"

    API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, current_url: str = "", document_context: str = ""):
        self._enabled = env_flag("HIBRIA_ENABLE_AI_FALLBACK", default=False)
        self._provider = (
            os.getenv("HIBRIA_AI_FALLBACK_PROVIDER", "gemini").strip().lower()
        )
        self._api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self._model = os.getenv("HIBRIA_AI_FALLBACK_MODEL", "gemini-2.5-flash").strip()
        self._current_url = current_url
        self._document_context = normalize_space(document_context)
        self._trusted_domains = split_env_list("HIBRIA_TRUSTED_DOMAINS")
        self._min_confidence = env_float("HIBRIA_AI_FALLBACK_MIN_CONFIDENCE", 0.45)
        self._daily_limit = env_int("HIBRIA_AI_FALLBACK_DAILY_LIMIT", 20)
        self._max_queries_per_claim = env_int(
            "HIBRIA_AI_FALLBACK_MAX_QUERIES_PER_CLAIM", 1
        )
        self._sleep_seconds = env_float("HIBRIA_AI_FALLBACK_SLEEP_SECONDS", 1.0)
        self._rate_limited_until = 0.0

    def is_available(self) -> bool:
        return bool(
            self._enabled
            and self._provider == "gemini"
            and self._api_key
            and self._api_key.lower() not in {"sua_chave_aqui", "your_key_here"}
            and time.time() >= self._rate_limited_until
        )

    def _quota_file(self) -> Path:
        return Path("data/runtime/ai_fallback_quota.json")

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

        return {"date": today, "count": int(data.get("count", 0))}

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

    def _is_same_url(self, url: str) -> bool:
        if env_flag("HIBRIA_ALLOW_SELF_EVIDENCE", default=False):
            return False

        if not self._current_url or not url:
            return False

        return url.strip().rstrip("/") == self._current_url.strip().rstrip("/")

    def _build_prompt(self, claim: Claim) -> str:
        queries = build_queries(claim)
        trusted_domains = (
            sorted(self._trusted_domains)
            if self._trusted_domains
            else [
                "g1.globo.com",
                "agenciabrasil.ebc.com.br",
                "estadao.com.br",
                "folha.uol.com.br",
                "bbc.com",
                "cnnbrasil.com.br",
                "reuters.com",
                "apnews.com",
                "checamos.afp.com",
                "aosfatos.org",
                "lupa.uol.com.br",
            ]
        )

        return f"""
Você é um assistente de recuperação de evidências jornalísticas para um sistema acadêmico de análise de confiabilidade.

Sua função NÃO é decidir se a claim é verdadeira ou falsa.
Sua única função é encontrar uma fonte externa verificável que possa ser analisada depois pelo sistema.

URL da notícia analisada:
"{self._current_url}"

Contexto geral da notícia analisada:
"{self._document_context}"

Claim a verificar:
"{claim.text}"

Consultas já usadas pelo sistema:
{json.dumps(queries, ensure_ascii=False)}

Domínios jornalísticos preferenciais (só como base, não precisa se limitar a eles):
{json.dumps(trusted_domains, ensure_ascii=False)}

Regras obrigatórias:
- Não responda com conhecimento próprio.
- Não invente fatos, URLs, títulos, datas, nomes ou fontes.
- Use busca web/grounding quando disponível.
- Priorize fontes jornalísticas, oficiais ou de checagem.
- Priorize os domínios preferenciais quando houver resultado relevante.
- Não use a mesma URL da notícia analisada.
- Não use redes sociais, blogs pessoais, fóruns ou páginas sem identificação editorial.
- A fonte precisa mencionar diretamente a claim ou os elementos centrais dela.
- Se a fonte apenas tratar de assunto parecido, mas não da mesma claim, retorne {{"found": false}}.
- Se não encontrar fonte verificável, retorne exatamente {{"found": false}}.
- O campo "text" deve ser um trecho factual curto da fonte, não uma explicação sua.
- O campo "relation" deve indicar apenas a relação aparente entre fonte e claim.
- Mesmo se a relation for "support", o sistema ainda fará similarity e stance depois.

Responda apenas em JSON válido, sem markdown e sem texto fora do JSON.

Formato se encontrar:
{{
  "found": true,
  "source": "...",
  "title": "...",
  "url": "...",
  "text": "...",
  "relation": "support|contradict|neutral",
  "confidence": 0.0
}}

Formato se não encontrar:
{{
  "found": false
}}
""".strip()

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = (text or "").strip()

        if not text:
            return {"found": False}

        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)

            if not match:
                return {"found": False}

            try:
                return json.loads(match.group(0))
            except Exception:
                return {"found": False}

    @staticmethod
    def _response_text(data: dict) -> str:
        candidates = data.get("candidates", []) or []

        if not candidates:
            return ""

        parts = candidates[0].get("content", {}).get("parts", []) or []
        return "\n".join(part.get("text", "") for part in parts if part.get("text"))

    def _validate_url(self, url: str) -> bool:
        if not url or self._is_same_url(url):
            return False

        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False

        try:
            response = requests.get(
                url,
                headers={"User-Agent": "HIBRIA-FactChecker/1.0 (TCC; educational)"},
                timeout=10,
                allow_redirects=True,
                stream=True,
            )
            return response.status_code < 400
        except Exception:
            return False

    def search(self, claim: Claim, top_k: int = 5) -> list[Evidence]:
        if not self._can_search():
            self._rate_limited_until = time.time() + 10 * 60
            logger.warning(
                "[ai_fallback] limite diário local atingido; camada pausada nesta execução"
            )
            return []

        prompt = self._build_prompt(claim)
        url = self.API_URL_TEMPLATE.format(model=self._model)

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 1024,
            },
        }

        try:
            response = requests.post(
                url,
                params={"key": self._api_key},
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
        except requests.RequestException as exc:
            self._rate_limited_until = time.time() + 5 * 60
            logger.warning(
                f"[ai_fallback] falha de conexão/timeout; camada pausada: {exc}"
            )
            return []

        self._register_search()
        time.sleep(self._sleep_seconds)

        if response.status_code == 429:
            self._rate_limited_until = time.time() + 10 * 60
            logger.warning(
                "[ai_fallback] limite/rate limit atingido; camada pausada nesta execução"
            )
            return []

        if should_skip_http_response(response, "ai_fallback"):
            return []

        data = safe_json_response(response, "ai_fallback")

        if not data:
            return []

        answer = self._extract_json(self._response_text(data))

        if not answer.get("found"):
            return []

        evidence_url = normalize_space(answer.get("url", ""))
        evidence_text = normalize_space(answer.get("text", ""))
        title = normalize_space(answer.get("title", ""))
        source = normalize_space(answer.get("source", ""))
        relation = normalize_space(answer.get("relation", "neutral"))
        confidence = float(answer.get("confidence", 0.0) or 0.0)

        if confidence < self._min_confidence:
            return []

        if (
            not evidence_url
            or not evidence_text
            or not self._validate_url(evidence_url)
        ):
            return []

        domain = extract_domain(evidence_url)
        source = source or domain or "Gemini Flash"
        similarity = max(
            0.55, _tfidf_similarity(claim.normalized or claim.text, evidence_text)
        )

        return [
            Evidence(
                text=evidence_text,
                source=source,
                url=evidence_url,
                similarity=similarity,
                claim_id=claim.claim_id,
                title=title,
                domain=domain,
                published_at=None,
                retrieval_layer=self.name,
                source_type="ai_retrieved_web",
                trusted_source=domain in self._trusted_domains,
                metadata={
                    "provider": "gemini",
                    "model": self._model,
                    "relation_declared_by_model": relation,
                    "confidence_declared_by_model": confidence,
                    "validated_url": True,
                    "note": "A IA apenas recuperou uma fonte candidata; similarity e stance ainda devem validar a relação com a claim.",
                },
            )
        ][:top_k]


# =============================================================================
# EvidenceRetriever — orquestrador principal
#
# Executa as camadas em ordem de prioridade, agrega os resultados,
# deduplica por URL e rerankeia por similaridade.
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
    MIN_SIMILARITY: float = env_float("HIBRIA_RETRIEVER_MIN_SIMILARITY", 0.05)

    # número mínimo de evidências antes de considerar early stopping
    EARLY_STOP_COUNT: int = env_int("HIBRIA_RETRIEVAL_TOP_K", 10)

    # snippets de NewsAPI/GDELT/Brave podem ser curtos, então o corte não pode ser alto demais
    MIN_EVIDENCE_CHARS: int = env_int("HIBRIA_MIN_EVIDENCE_CHARS", 50)

    # limiar usado apenas para decidir se vale acionar a IA fallback
    FALLBACK_TRIGGER_SCORE: float = env_float("HIBRIA_FALLBACK_TRIGGER_SCORE", 0.55)

    # limiar para considerar que uma claim já possui evidência suficiente
    SUFFICIENT_EVIDENCE_SCORE: float = env_float(
        "HIBRIA_SUFFICIENT_EVIDENCE_SCORE", 0.55
    )

    # número mínimo de evidências suficientes para parar de buscar para uma claim
    MIN_SUFFICIENT_EVIDENCES: int = env_int("HIBRIA_MIN_SUFFICIENT_EVIDENCES", 1)

    # Por padrão, o retriever NÃO para em Tavily/Serper só porque achou uma
    # evidência candidata. A confirmação de suficiência acontece depois, no
    # similarity.py e no stance_model.py.
    # Se quiser economizar APIs em testes, ative no .env:
    # HIBRIA_RETRIEVER_EARLY_STOP=true
    EARLY_STOP_ENABLED: bool = env_flag("HIBRIA_RETRIEVER_EARLY_STOP", False)

    LAYER_WEIGHT = {
        "vector_store": 1.00,
        "factcheck": 0.95,
        "web_search": 0.70,
        "tavily_search": 0.68,
        "serp_search": 0.68,
        "gdelt": 0.80,
        "newsapi": 0.80,
        "ai_fallback": 0.70,
        "wikipedia": 0.20,
    }

    def __init__(
        self, vector_store=None, current_url: str = "", document_context: str = ""
    ):
        """
        Inicializa as camadas em ordem de prioridade.
        vector_store: instância de VectorStore ou None se não indexado ainda.
        current_url: URL da notícia que está sendo analisada.
        document_context: contexto do documento que está sendo analisado.
        """
        self._document_context = normalize_space(document_context)
        self._sources: list[EvidenceSource] = [
            VectorStoreSource(vector_store, current_url=current_url),
            FactCheckSource(),
            WebSearchSource(
                current_url=current_url,
                document_context=self._document_context,
            ),
            TavilySearchSource(
                current_url=current_url,
                document_context=self._document_context,
            ),
            SerpSearchSource(
                current_url=current_url,
                document_context=self._document_context,
            ),
            GdeltSource(
                document_context=self._document_context,
            ),
            AIFallbackSource(
                current_url=current_url,
                document_context=self._document_context,
            ),
            WikipediaSource(),
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

    def _has_sufficient_external_evidence(self, evidences: list[Evidence]) -> bool:
        """
        Verifica se já existe evidência externa forte o suficiente para evitar
        acionar a IA fallback. Wikipedia não conta aqui, pois é contexto.
        """
        for evidence in evidences:
            if evidence.source_type == "encyclopedia":
                continue

            if evidence.retrieval_layer == "wikipedia":
                continue

            if evidence.similarity >= self.FALLBACK_TRIGGER_SCORE:
                return True

        return False

    def _sufficient_evidences(self, evidences: list[Evidence]) -> list[Evidence]:
        """
        Retorna apenas evidências fortes o suficiente para considerar
        que a claim já foi preenchida.

        Wikipedia não conta como evidência suficiente, pois é apenas contexto.
        """
        sufficient: list[Evidence] = []

        for evidence in evidences:
            if evidence.source_type == "encyclopedia":
                continue

            if evidence.retrieval_layer == "wikipedia":
                continue

            if evidence.similarity >= self.SUFFICIENT_EVIDENCE_SCORE:
                sufficient.append(evidence)

        return sufficient

    def _claim_is_filled(self, evidences: list[Evidence]) -> bool:
        """
        Verifica se a claim já possui evidência suficiente.
        Se sim, o retriever para de gastar outras APIs nessa claim.
        """
        return (
            len(self._sufficient_evidences(evidences)) >= self.MIN_SUFFICIENT_EVIDENCES
        )

    def retrieve(
        self,
        claim: Claim,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Busca evidências para uma única claim em modo sequencial.

        Estratégia:
        1. Tenta uma fonte por vez, em ordem de prioridade.
        2. Mantém as evidências candidatas recuperadas.
        3. Por padrão, continua para as próximas camadas, porque a suficiência
           real só é confirmada depois pelo similarity.py e pelo stance_model.py.
        4. Só faz parada antecipada se HIBRIA_RETRIEVER_EARLY_STOP=true.
        5. Se nenhuma fonte retornar evidência suficiente depois da validação
           semântica, as próximas etapas marcam a claim como evidência insuficiente.
        """
        start_time = time.time()
        all_evidences: list[Evidence] = []
        layers_used: list[str] = []
        layers_failed: dict[str, str] = {}

        for source in self._sources:
            if not source.is_available():
                layers_failed[source.name] = (
                    "não disponível (sem API key, limite atingido ou base local ausente)"
                )
                continue

            # Não pula Tavily/Serper/GDELT só porque uma camada anterior retornou
            # uma evidência candidata. A suficiência real é calculada depois,
            # em similarity.py e stance_model.py.
            #
            # A única exceção opcional é a parada antecipada controlada por env,
            # útil apenas para economizar APIs em testes.
            if self.EARLY_STOP_ENABLED and self._claim_is_filled(all_evidences):
                layers_failed[source.name] = (
                    "pulada — parada antecipada ativada e claim já possui evidência candidata suficiente"
                )
                continue

            # Gemini é último recurso: só roda quando as fontes anteriores ainda
            # não trouxeram nenhuma evidência externa candidata forte.
            if source.name == "ai_fallback" and self._has_sufficient_external_evidence(
                all_evidences
            ):
                layers_failed[source.name] = (
                    "pulada — já existe evidência externa candidata antes da IA"
                )
                continue

            try:
                evidences = source.search(claim, top_k=top_k)
            except Exception as exc:
                layers_failed[source.name] = f"erro inesperado: {exc}"

                # Evita repetir o mesmo traceback para todas as claims quando
                # uma fonte quebra por erro interno durante a execução.
                if hasattr(source, "_rate_limited_until"):
                    try:
                        setattr(source, "_rate_limited_until", time.time() + 10 * 60)
                    except Exception:
                        pass

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

                # Deduplica antes de decidir se já pode parar.
                all_evidences = self._deduplicate(all_evidences)

                if self.EARLY_STOP_ENABLED and self._claim_is_filled(all_evidences):
                    break
            else:
                layers_failed[source.name] = "sem resultados relevantes"

        all_evidences = self._deduplicate(all_evidences)

        ranked = sorted(
            all_evidences,
            key=self._rank_score,
            reverse=True,
        )[:top_k]

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
                    for layer in {
                        "factcheck",
                        "web_search",
                        "tavily_search",
                        "serp_search",
                        "gdelt",
                        "newsapi",
                        "wikipedia",
                        "ai_fallback",
                    }
                ):
                    time.sleep(
                        float(os.getenv("HIBRIA_RETRIEVAL_SLEEP_SECONDS", "0.3"))
                    )

        return results
