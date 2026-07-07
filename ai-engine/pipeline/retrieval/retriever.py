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
#   Camada 7 — Crawlee Search (rastreamento direcionado de notícias)
#   Camada 8 — IA fallback com Gemini Flash, apenas se ainda faltar evidência
#   Camada 9 — Wikipedia (contexto enciclopédico, não prova principal)
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
import unicodedata
import asyncio
import threading
import concurrent.futures
from datetime import timedelta
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

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
# Filtros de qualidade para evidências web
# =============================================================================

DEFAULT_BLOCKED_EVIDENCE_DOMAINS = {
    "youtube.com",
    "youtu.be",
    "instagram.com",
    "facebook.com",
    "tiktok.com",
    "x.com",
    "twitter.com",
    "threads.net",
    "reddit.com",
    "pinterest.com",
    "kwai.com",
    "podcast.com.br",
    "news.google.com",
}

STOPWORDS_RETRIEVAL = {
    "a",
    "o",
    "as",
    "os",
    "um",
    "uma",
    "uns",
    "umas",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "por",
    "para",
    "com",
    "sem",
    "sob",
    "sobre",
    "entre",
    "ate",
    "até",
    "e",
    "ou",
    "mas",
    "que",
    "se",
    "ao",
    "aos",
    "sua",
    "seu",
    "suas",
    "seus",
    "foi",
    "ser",
    "ter",
    "tem",
    "tinha",
    "havia",
    "disse",
    "tambem",
    "também",
    "nesta",
    "neste",
    "desta",
    "deste",
    "essa",
    "esse",
    "isso",
    "como",
    "apos",
    "após",
    "pelo",
    "pela",
    "pelos",
    "pelas",
    "mais",
    "menos",
}


def _plain(text: str) -> str:
    """Normaliza texto para comparação simples, removendo acentos."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def _significant_tokens(text: str) -> set[str]:
    plain = _plain(text)
    tokens = re.findall(r"\b[a-z0-9][a-z0-9-]{2,}\b", plain)
    return {
        token
        for token in tokens
        if token not in STOPWORDS_RETRIEVAL and len(token) >= 3
    }


def _number_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text or ""))


def _blocked_domains() -> set[str]:
    return DEFAULT_BLOCKED_EVIDENCE_DOMAINS | split_env_list(
        "HIBRIA_BLOCKED_EVIDENCE_DOMAINS"
    )


def _is_blocked_evidence_domain(url: str) -> bool:
    domain = extract_domain(url)
    if not domain:
        return False
    blocked = _blocked_domains()
    return domain in blocked or any(domain.endswith(f".{item}") for item in blocked)


def _canonical_article_key(url: str) -> tuple[str, str]:
    """Cria uma chave aproximada para evitar usar a própria notícia como evidência."""
    if not url:
        return "", ""

    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]

    path = parsed.path.lower()
    path = path.replace("/google/amp/", "/")
    path = path.replace("/amp/", "/")
    path = path.replace(".amp", "")
    path = re.sub(r"\.(ghtml|html|htm)$", "", path)
    path = re.sub(r"[^a-z0-9]+", "-", path).strip("-")

    # Em notícias, o slug costuma estar no fim da URL e é a melhor comparação.
    parts = [part for part in path.split("-") if len(part) >= 3]
    slug = "-".join(parts[-18:])
    return domain, slug


def _is_same_article_url(current_url: str, evidence_url: str) -> bool:
    if not current_url or not evidence_url:
        return False

    current_clean = current_url.strip().rstrip("/")
    evidence_clean = evidence_url.strip().rstrip("/")

    if current_clean == evidence_clean:
        return True

    current_domain, current_slug = _canonical_article_key(current_url)
    evidence_domain, evidence_slug = _canonical_article_key(evidence_url)

    if (
        not current_domain
        or not evidence_domain
        or not current_slug
        or not evidence_slug
    ):
        return False

    if current_domain != evidence_domain:
        return False

    current_tokens = set(current_slug.split("-"))
    evidence_tokens = set(evidence_slug.split("-"))

    if not current_tokens or not evidence_tokens:
        return False

    overlap = len(current_tokens & evidence_tokens)
    return overlap / max(1, min(len(current_tokens), len(evidence_tokens))) >= 0.65


def _entity_tokens(entity: str) -> set[str]:
    return _significant_tokens(entity)


def _has_required_entity_overlap(claim: Claim, evidence_text: str) -> bool:
    entities = [entity for entity in getattr(claim, "entities", []) if entity]

    # Se o detector não trouxe entidade, não bloqueia por entidade.
    if not entities:
        return True

    evidence_tokens = _significant_tokens(evidence_text)

    for entity in entities:
        tokens = _entity_tokens(entity)
        if not tokens:
            continue

        # Entidades de uma palavra: exige a palavra.
        if len(tokens) == 1 and next(iter(tokens)) in evidence_tokens:
            return True

        # Entidades compostas: exige pelo menos metade dos termos.
        if len(tokens & evidence_tokens) / max(1, len(tokens)) >= 0.5:
            return True

    return False


def _has_required_number_overlap(claim: Claim, evidence_text: str) -> bool:
    claim_numbers = _number_tokens(claim.text)

    if not claim_numbers:
        return True

    evidence_numbers = _number_tokens(evidence_text)

    if not evidence_numbers:
        return False

    # Compara também sem %, vírgula e ponto para não perder 13% vs 13.
    normalized_claim = {re.sub(r"\D", "", item) for item in claim_numbers}
    normalized_evidence = {re.sub(r"\D", "", item) for item in evidence_numbers}

    normalized_claim.discard("")
    normalized_evidence.discard("")

    return bool(normalized_claim & normalized_evidence)


def _is_relevant_candidate(claim: Claim, evidence_text: str, url: str = "") -> bool:
    """
    Filtra candidatos obviamente ruins antes de virarem Evidence.

    Evita aceitar resultados genéricos só porque a busca retornou algo sobre o
    título da notícia, redes sociais ou temas laterais como Taylor Swift.
    """
    if not evidence_text or len(evidence_text.strip()) < 50:
        return False

    if _is_blocked_evidence_domain(url):
        return False

    if not _has_required_entity_overlap(claim, evidence_text):
        return False

    if not _has_required_number_overlap(claim, evidence_text):
        return False

    claim_tokens = _significant_tokens(claim.normalized or claim.text)
    evidence_tokens = _significant_tokens(evidence_text)

    if not claim_tokens or not evidence_tokens:
        return False

    overlap = claim_tokens & evidence_tokens

    # Claims curtas precisam de pelo menos 2 termos fortes em comum.
    min_overlap = 2 if len(claim_tokens) <= 8 else 3

    return len(overlap) >= min_overlap


def _candidate_similarity(claim: Claim, evidence_text: str) -> float:
    claim_text = claim.normalized or claim.text
    claim_tokens = _significant_tokens(claim_text)
    evidence_tokens = _significant_tokens(evidence_text)

    if not claim_tokens or not evidence_tokens:
        return 0.0

    overlap = claim_tokens & evidence_tokens
    overlap_ratio = len(overlap) / max(1, min(len(claim_tokens), 14))
    tfidf_score = _tfidf_similarity(claim_text, evidence_text)

    entity_bonus = 0.08 if _has_required_entity_overlap(claim, evidence_text) else 0.0
    number_bonus = 0.07 if _has_required_number_overlap(claim, evidence_text) else 0.0

    return round(
        clamp(max(tfidf_score, overlap_ratio) + entity_bonus + number_bonus), 4
    )


def _build_claim_search_query(
    claim: Claim,
    document_context: str = "",
    max_chars: int = 220,
) -> str:
    """
    Monta query centrada na claim, não no título inteiro da notícia.

    O título da matéria estava contaminando as buscas de todas as claims
    com termos como "Taylor Swift", mesmo quando a afirmação era sobre
    regras do Partido Trabalhista ou aprovação de Starmer.
    """
    parts: list[str] = []

    for entity in getattr(claim, "entities", [])[:4]:
        entity = normalize_space(entity)
        if entity and len(entity) >= 4:
            parts.append(entity)

    subject = normalize_space(getattr(claim, "subject", ""))
    if subject and len(subject) >= 4 and subject.lower() not in STOPWORDS_RETRIEVAL:
        parts.append(subject)

    numbers = sorted(_number_tokens(claim.text))
    if numbers:
        parts.append(" ".join(numbers))

    text = normalize_space(claim.text or claim.normalized)
    text = re.sub(r"[“”\"'’‘()\[\]{}:;|/\\]", " ", text)
    text = normalize_space(text)

    words = text.split()
    if len(words) > 18:
        text = " ".join(words[:18])

    parts.append(text)

    # Só usa contexto do documento para claims muito curtas ou sem entidade.
    # Mesmo assim, usa poucos termos, para não contaminar a busca inteira.
    if len(_significant_tokens(text)) < 5 and document_context:
        context_tokens = list(_significant_tokens(document_context))[:5]
        if context_tokens:
            parts.append(" ".join(context_tokens))

    seen: set[str] = set()
    clean_parts: list[str] = []

    for part in parts:
        part = normalize_space(part)
        key = _plain(part)

        if not part or key in seen:
            continue

        seen.add(key)
        clean_parts.append(part)

    return " ".join(clean_parts)[:max_chars]


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

        return _is_same_article_url(self._current_url, url)


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

        return _is_same_article_url(self._current_url, url)

    def _is_trusted_domain(self, url: str) -> bool:
        domain = extract_domain(url)
        return bool(
            domain and self._trusted_domains and domain in self._trusted_domains
        )

    def _build_query(self, claim: Claim) -> str:
        return _build_claim_search_query(
            claim,
            document_context=self._document_context,
            max_chars=280,
        )

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

            if not _is_relevant_candidate(claim, evidence_text, url):
                continue

            similarity = _candidate_similarity(claim, evidence_text)

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

        return _is_same_article_url(self._current_url, url)

    def _is_trusted_domain(self, url: str) -> bool:
        domain = extract_domain(url)
        return bool(
            domain and self._trusted_domains and domain in self._trusted_domains
        )

    def _build_query(self, claim: Claim) -> str:
        return _build_claim_search_query(
            claim,
            document_context=self._document_context,
            max_chars=220,
        )

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

            if not _is_relevant_candidate(claim, evidence_text, url):
                continue

            similarity = _candidate_similarity(claim, evidence_text)

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

                if not _is_relevant_candidate(claim, evidence_text, url):
                    continue

                similarity = _candidate_similarity(claim, evidence_text)

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

                if not _is_relevant_candidate(claim, evidence_text, url):
                    continue

                similarity = _candidate_similarity(claim, evidence_text)

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

        return _is_same_article_url(self._current_url, url)

    def _build_query(self, claim: Claim) -> str:
        return _build_claim_search_query(
            claim,
            document_context=self._document_context,
            max_chars=220,
        )

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

            if not _is_relevant_candidate(claim, evidence_text, url):
                continue

            similarity = _candidate_similarity(claim, evidence_text)

            evidences.append(
                Evidence(
                    text=evidence_text,
                    source=domain or "Brave Search",
                    url=url,
                    similarity=similarity,
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
# Camada 7: Crawlee Search  ⭐⭐⭐
#
# Busca por rastreamento direcionado de páginas jornalísticas.
# A camada usa feeds públicos de notícias como ponto de partida, coleta URLs
# candidatas e abre as páginas uma única vez. Depois, cada claim é comparada
# com os trechos extraídos dessas páginas.
#
# Estratégia atual:
#   1. Monta consultas globais a partir do contexto da notícia e das claims.
#   2. Consulta Google News RSS e Bing News RSS sem usar APIs pagas.
#   3. Abre um conjunto limitado de URLs candidatas e guarda o texto em cache.
#   4. Para cada claim, escolhe o melhor trecho das páginas já rastreadas.
#   5. Só faz busca pontual por claim quando o cache global não encontra nada.
# =============================================================================


# =============================================================================
# Camada 7: Crawlee Search  ⭐⭐⭐
#
# Busca por rastreamento direcionado de páginas jornalísticas.
# A camada usa feeds públicos de notícias como ponto de partida, coleta URLs
# candidatas e abre as páginas uma única vez. Depois, cada claim é comparada
# com os trechos extraídos dessas páginas.
#
# Esta implementação é genérica: não contém nomes de fontes, pessoas, lugares,
# institutos, partidos ou temas fixos de uma notícia específica. As entidades,
# números e termos usados na busca são extraídos dinamicamente da notícia
# analisada e das claims recebidas do pipeline.
# =============================================================================


CRAWLEE_PROVIDER_NAME = "rss_crawlee_search"

CRAWLEE_GENERIC_WEAK_TOKENS = {
    "brasil",
    "noticia",
    "notícia",
    "jornal",
    "site",
    "portal",
    "fonte",
    "segundo",
    "acordo",
    "tambem",
    "também",
    "deste",
    "desta",
    "neste",
    "nesta",
    "disse",
    "afirma",
    "aponta",
    "mostra",
    "informa",
    "divulgado",
    "divulgada",
    "publicado",
    "publicada",
    "levantamento",
    "pesquisa",
    "estudo",
    "relatorio",
    "relatório",
    "dados",
    "numero",
    "número",
    "valores",
    "resultado",
    "resultados",
}

CRAWLEE_GENERIC_NUMERIC_CONTEXT_TERMS = {
    "porcentagem",
    "percentual",
    "percentuais",
    "pontos",
    "margem",
    "erro",
    "nivel",
    "nível",
    "confianca",
    "confiança",
    "voto",
    "votos",
    "intencao",
    "intenção",
    "intencoes",
    "intenções",
    "valor",
    "valores",
    "preco",
    "preço",
    "aumento",
    "queda",
    "redução",
    "reducao",
    "crescimento",
    "habitantes",
    "pessoas",
    "entrevistados",
    "municipios",
    "municípios",
    "casos",
    "mortes",
    "bilhao",
    "bilhão",
    "bilhoes",
    "bilhões",
    "milhao",
    "milhão",
    "milhoes",
    "milhões",
}

CRAWLEE_GENERIC_SPECIFIC_TERMS = {
    "rejeicao": {"rejeicao", "rejeição", "nao votaria", "não votaria", "rejeita", "rejeitam"},
    "registro": {"registrada", "registrado", "registro", "protocolo", "identificada", "identificado"},
    "margem": {"margem de erro", "pontos percentuais", "nivel de confianca", "nível de confiança"},
    "votos_validos": {"votos validos", "votos válidos"},
    "segundo_turno": {"segundo turno", "2º turno", "2o turno"},
    "primeiro_turno": {"primeiro turno", "1º turno", "1o turno"},
}


def _crawlee_phrase_tokens(text: str) -> set[str]:
    tokens = _significant_tokens(text)
    return {token for token in tokens if token not in CRAWLEE_GENERIC_WEAK_TOKENS}


def _crawlee_text_phrases(text: str, max_phrases: int = 8) -> list[str]:
    """Extrai frases informativas de forma genérica a partir de entidades/capitalização."""
    value = normalize_space(text)
    if not value:
        return []

    phrases: list[str] = []

    # Expressões entre aspas costumam ser nomes de obras, órgãos, eventos ou fontes.
    for match in re.finditer(r"[\"“”'‘’]([^\"“”'‘’]{3,80})[\"“”'‘’]", value):
        phrases.append(normalize_space(match.group(1)))

    # Sequências capitalizadas: pessoas, instituições, locais, eventos etc.
    pattern = r"\b(?:[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ.-]{2,})(?:\s+(?:de|da|do|dos|das|e|&|[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ.-]{1,})){0,6}"
    for match in re.finditer(pattern, value):
        phrase = normalize_space(match.group(0))
        if len(phrase) >= 4:
            phrases.append(phrase)

    # Códigos/identificadores fortes: registros, protocolos, processos, siglas com números.
    for match in re.finditer(r"\b[A-Z]{2,}[- ]?\d{2,}[A-Z0-9/-]*\b", value, flags=re.IGNORECASE):
        phrases.append(normalize_space(match.group(0)))

    seen: set[str] = set()
    clean: list[str] = []
    for phrase in phrases:
        phrase = normalize_space(phrase)
        tokens = _crawlee_phrase_tokens(phrase)
        if not phrase or not tokens:
            continue
        key = _plain(phrase)
        if key in seen:
            continue
        seen.add(key)
        clean.append(phrase)
        if len(clean) >= max_phrases:
            break
    return clean


def _crawlee_strong_entities(claim: Claim, document_context: str = "") -> list[str]:
    """Entidades fortes sem usar nomes fixos; vem de claim.entities e do próprio texto."""
    candidates: list[str] = []
    for entity in getattr(claim, "entities", []) or []:
        candidates.append(entity)
    if getattr(claim, "subject", ""):
        candidates.append(claim.subject)
    candidates.extend(_crawlee_text_phrases(claim.text, max_phrases=6))

    # Para claims curtas/genéricas, adiciona poucas entidades do contexto da notícia.
    if len(_crawlee_phrase_tokens(claim.text)) <= 5 and document_context:
        candidates.extend(_crawlee_text_phrases(document_context, max_phrases=5))

    seen: set[str] = set()
    result: list[str] = []
    for candidate in candidates:
        candidate = normalize_space(candidate)
        tokens = _crawlee_phrase_tokens(candidate)
        if not candidate or not tokens:
            continue
        key = _plain(candidate)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _crawlee_number_mentions(text: str) -> list[dict]:
    """Extrai números preservando percentuais, milhares e identificadores alfanuméricos."""
    value = text or ""
    mentions: list[dict] = []

    for match in re.finditer(r"\b[A-Z]{1,5}[- ]?\d{2,}[A-Z0-9/-]*\b", value, flags=re.IGNORECASE):
        raw = match.group(0)
        norm = re.sub(r"\D", "", raw)
        if norm:
            mentions.append({
                "raw": raw,
                "norm": norm,
                "start": match.start(),
                "end": match.end(),
                "percent": False,
                "identifier": True,
            })

    pattern = (
        r"(?<![\w/])\d{1,3}(?:\.\d{3})+(?:,\d+)?\s*%?"
        r"|(?<![\w/])\d+(?:[.,]\d+)?\s*%?"
    )
    for match in re.finditer(pattern, value):
        raw = match.group(0).strip()
        norm = re.sub(r"\D", "", raw)
        if not norm:
            continue
        mentions.append({
            "raw": raw,
            "norm": norm,
            "start": match.start(),
            "end": match.end(),
            "percent": "%" in raw,
            "identifier": False,
        })

    seen: set[tuple[int, int, str]] = set()
    unique: list[dict] = []
    for item in mentions:
        key = (int(item["start"]), int(item["end"]), str(item["norm"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _crawlee_number_norms(text: str) -> set[str]:
    return {str(item["norm"]) for item in _crawlee_number_mentions(text)}


def _crawlee_required_numbers(claim_text: str) -> set[str]:
    """Seleciona números verificáveis, evitando dia/mês/ordem quando não são centrais."""
    plain = _plain(claim_text)
    tokens = _significant_tokens(claim_text)
    required: set[str] = set()

    for item in _crawlee_number_mentions(claim_text):
        norm = str(item["norm"])
        raw = str(item["raw"])
        if not norm:
            continue

        has_percent = bool(item.get("percent"))
        is_identifier = bool(item.get("identifier"))
        is_large = len(norm) >= 3
        is_year = len(norm) == 4 and norm.startswith("20") and not is_identifier
        is_small_ordinal = norm in {"1", "2", "3", "4", "5"}
        has_numeric_context = bool(tokens & CRAWLEE_GENERIC_NUMERIC_CONTEXT_TERMS)
        is_margin_or_confidence = bool({"margem", "erro", "confianca", "confiança"} & tokens)

        if is_year:
            continue

        if is_small_ordinal and not (has_percent or is_margin_or_confidence or is_identifier):
            continue

        if has_percent or is_identifier or is_large:
            required.add(norm)
            continue

        if has_numeric_context:
            required.add(norm)

    return required


def _crawlee_entity_groups_from_claim(claim: Claim, document_context: str = "") -> list[set[str]]:
    groups: list[set[str]] = []
    for entity in _crawlee_strong_entities(claim, document_context):
        tokens = _crawlee_phrase_tokens(entity)
        if tokens:
            groups.append(tokens)

    seen: set[tuple[str, ...]] = set()
    unique: list[set[str]] = []
    for group in groups:
        key = tuple(sorted(group))
        if key and key not in seen:
            seen.add(key)
            unique.append(group)
    return unique


def _crawlee_entity_positions(text: str, entities: list[str]) -> dict[str, list[int]]:
    plain = _plain(text)
    positions: dict[str, list[int]] = {}
    for entity in entities:
        key_tokens = _crawlee_phrase_tokens(entity)
        if not key_tokens:
            continue
        key = " ".join(sorted(key_tokens))
        for token in key_tokens:
            for match in re.finditer(rf"\b{re.escape(token)}\b", plain):
                positions.setdefault(key, []).append(match.start())
    return positions


def _crawlee_number_entity_pairs(claim: Claim, document_context: str = "") -> list[tuple[set[str], str]]:
    """Liga números curtos à entidade mais próxima na própria claim, sem nomes fixos."""
    entities = _crawlee_strong_entities(claim, document_context)
    if not entities:
        return []

    positions = _crawlee_entity_positions(claim.text, entities)
    if not positions:
        return []

    required = _crawlee_required_numbers(claim.text)
    pairs: list[tuple[set[str], str]] = []

    for item in _crawlee_number_mentions(claim.text):
        norm = str(item["norm"])
        if norm not in required:
            continue
        if len(norm) >= 3 or norm in {"50", "95"} or item.get("identifier"):
            continue

        midpoint = (int(item["start"]) + int(item["end"])) // 2
        best_key = ""
        best_distance = 10**9
        for key, poss in positions.items():
            for pos in poss:
                distance = abs(pos - midpoint)
                if distance < best_distance:
                    best_distance = distance
                    best_key = key
        if best_key and best_distance <= 160:
            pairs.append((set(best_key.split()), norm))

    seen: set[tuple[tuple[str, ...], str]] = set()
    unique: list[tuple[set[str], str]] = []
    for group, number in pairs:
        key = (tuple(sorted(group)), number)
        if key in seen:
            continue
        seen.add(key)
        unique.append((group, number))
    return unique


def _crawlee_pair_appears_near(evidence_text: str, entity_tokens: set[str], number: str) -> bool:
    plain = _plain(evidence_text)
    number_patterns = [re.escape(number)]
    if len(number) > 3:
        number_patterns.append(re.escape(f"{number[:-3]}.{number[-3:]}"))

    for token in entity_tokens:
        for match in re.finditer(rf"\b{re.escape(token)}\b", plain):
            left = max(0, match.start() - 180)
            right = min(len(plain), match.end() + 180)
            window = plain[left:right]
            for pattern in number_patterns:
                if re.search(rf"(?<!\d){pattern}(?!\d)", window):
                    if re.search(rf"(?<!\d){pattern}(?!\d)\s+dias?", window):
                        continue
                    return True
    return False


def _crawlee_base_relevance(claim: Claim, evidence_text: str, url: str) -> tuple[bool, str]:
    if not evidence_text or len(evidence_text.strip()) < 50:
        return False, "trecho muito curto"
    if _is_blocked_evidence_domain(url):
        return False, "domínio bloqueado"

    claim_tokens = _crawlee_phrase_tokens(claim.normalized or claim.text)
    evidence_tokens = _significant_tokens(evidence_text)
    if not claim_tokens or not evidence_tokens:
        return False, "sem tokens relevantes"

    overlap = claim_tokens & evidence_tokens
    min_overlap = 1 if len(claim_tokens) <= 5 else 2
    if len(overlap) < min_overlap:
        return False, "baixa sobreposição lexical"
    return True, "relevância base compatível"


def _crawlee_entity_matches(document_context: str, claim: Claim, evidence_text: str) -> tuple[bool, str]:
    groups = _crawlee_entity_groups_from_claim(claim, document_context)
    if not groups:
        return True, "sem entidade forte obrigatória"

    evidence_tokens = _significant_tokens(evidence_text)
    matched = 0
    for group in groups:
        if len(group) == 1:
            if next(iter(group)) in evidence_tokens:
                matched += 1
        else:
            if len(group & evidence_tokens) / max(1, len(group)) >= 0.5:
                matched += 1

    if matched >= 1:
        return True, "entidade compatível"
    return False, "nenhuma entidade forte da claim aparece na evidência"


def _crawlee_numbers_match(document_context: str, claim: Claim, evidence_text: str) -> tuple[bool, str]:
    required = _crawlee_required_numbers(claim.text)
    if not required:
        return True, "sem números obrigatórios"

    evidence_numbers = _crawlee_number_norms(evidence_text)
    missing = required - evidence_numbers
    if missing:
        return False, f"números obrigatórios ausentes: {sorted(missing)}"

    missing_pairs: list[str] = []
    for entity_tokens, number in _crawlee_number_entity_pairs(claim, document_context):
        if not _crawlee_pair_appears_near(evidence_text, entity_tokens, number):
            missing_pairs.append(f"{'/'.join(sorted(entity_tokens))}+{number}")
    if missing_pairs:
        return False, f"número distante da entidade correta: {missing_pairs}"

    return True, "números compatíveis"


def _crawlee_specific_terms_match(claim: Claim, evidence_text: str) -> tuple[bool, str]:
    claim_plain = _plain(claim.text)
    evidence_plain = _plain(evidence_text)

    checks = []
    if "rejeicao" in claim_plain or "rejeição" in claim_plain:
        checks.append(("rejeicao", "claim de rejeição sem rejeição na evidência"))
    if "registr" in claim_plain or "protocolo" in claim_plain:
        checks.append(("registro", "claim de registro/protocolo sem registro na evidência"))
    if "margem" in claim_plain and "erro" in claim_plain:
        checks.append(("margem", "claim de margem de erro sem margem/nível na evidência"))
    if "votos valid" in claim_plain:
        checks.append(("votos_validos", "claim de votos válidos sem votos válidos na evidência"))
    if "2º turno" in claim_plain or "2o turno" in claim_plain or "segundo turno" in claim_plain:
        checks.append(("segundo_turno", "claim de segundo turno sem segundo turno na evidência"))
    if "1º turno" in claim_plain or "1o turno" in claim_plain or "primeiro turno" in claim_plain:
        checks.append(("primeiro_turno", "claim de primeiro turno sem primeiro turno na evidência"))

    for key, reason in checks:
        terms = CRAWLEE_GENERIC_SPECIFIC_TERMS[key]
        if not any(_plain(term) in evidence_plain for term in terms):
            return False, reason

    return True, "termos específicos compatíveis"


def _crawlee_context_matches(document_context: str, claim: Claim, evidence_text: str) -> tuple[bool, str]:
    """Garante contexto mínimo para claims muito genéricas, sem tema fixo."""
    claim_tokens = _crawlee_phrase_tokens(claim.text)
    doc_tokens = _crawlee_phrase_tokens(document_context)
    evidence_tokens = _significant_tokens(evidence_text)

    # Claims curtas como “segundo o levantamento” herdam um pouco do contexto.
    if len(claim_tokens) <= 5 and doc_tokens:
        context_overlap = evidence_tokens & doc_tokens
        if len(context_overlap) < 2:
            return False, "claim genérica sem contexto suficiente da notícia"

    return True, "contexto compatível"


def _crawlee_strict_filter(document_context: str, claim: Claim, evidence_text: str, url: str) -> tuple[bool, str]:
    checks = [
        _crawlee_base_relevance(claim, evidence_text, url),
        _crawlee_entity_matches(document_context, claim, evidence_text),
        _crawlee_numbers_match(document_context, claim, evidence_text),
        _crawlee_specific_terms_match(claim, evidence_text),
        _crawlee_context_matches(document_context, claim, evidence_text),
    ]

    for ok, reason in checks:
        if not ok:
            return False, reason
    return True, "aprovada pelo filtro Crawlee Search"


def _crawlee_similarity(document_context: str, claim: Claim, evidence_text: str) -> float:
    claim_text = claim.normalized or claim.text
    claim_tokens = _crawlee_phrase_tokens(claim_text)
    evidence_tokens = _significant_tokens(evidence_text)
    if not claim_tokens or not evidence_tokens:
        return 0.0

    overlap = claim_tokens & evidence_tokens
    overlap_ratio = len(overlap) / max(1, min(len(claim_tokens), 14))
    tfidf_score = _tfidf_similarity(claim_text, evidence_text)

    _, number_reason = _crawlee_numbers_match(document_context, claim, evidence_text)
    number_bonus = 0.10 if "compatíveis" in number_reason else 0.0
    _, entity_reason = _crawlee_entity_matches(document_context, claim, evidence_text)
    entity_bonus = 0.07 if "compatível" in entity_reason else 0.0
    _, context_reason = _crawlee_context_matches(document_context, claim, evidence_text)
    context_bonus = 0.04 if "compatível" in context_reason else 0.0

    return round(clamp(max(tfidf_score, overlap_ratio) + number_bonus + entity_bonus + context_bonus), 4)


class CrawleeSearchSource(EvidenceSource):
    """
    Busca gratuita por notícias usando RSS público + crawling direcionado.

    A descoberta de URLs usa Google News RSS e Bing News RSS. A extração das
    páginas usa requests + BeautifulSoup para manter compatibilidade no Windows
    e evitar problemas de event loop. O texto das páginas é cacheado durante a
    execução para que várias claims possam reaproveitar as mesmas fontes.
    """

    name = "crawlee_search"

    DEFAULT_NEWS_SEARCH_SEEDS = {
        "google_news": "https://news.google.com/rss/search?q={query}&hl=pt-BR&gl=BR&ceid=BR:pt-419",
        "bing_news": "https://www.bing.com/news/search?q={query}&format=rss&setlang=pt-BR&cc=BR",
    }

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36 HIBRIA-FactChecker/1.0"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    }

    def __init__(self, current_url: str = "", document_context: str = ""):
        self._enabled = env_flag("HIBRIA_ENABLE_CRAWLEE_SEARCH", default=False)
        self._trusted_domains = split_env_list("HIBRIA_TRUSTED_DOMAINS")
        self._current_url = current_url
        self._document_context = normalize_space(document_context)
        self._daily_limit = env_int("HIBRIA_CRAWLEE_DAILY_LIMIT", 40)
        self._timeout_seconds = env_int("HIBRIA_CRAWLEE_TIMEOUT_SECONDS", 10)
        self._sleep_seconds = env_float("HIBRIA_CRAWLEE_SLEEP_SECONDS", 0.1)
        self._debug = env_flag("HIBRIA_CRAWLEE_DEBUG", default=False)
        self._rate_limited_until = 0.0
        self._import_error = ""

        self._max_global_queries = env_int("HIBRIA_CRAWLEE_GLOBAL_MAX_QUERIES", 6)
        self._max_global_urls = env_int("HIBRIA_CRAWLEE_GLOBAL_MAX_URLS", 24)
        self._max_global_pages = env_int("HIBRIA_CRAWLEE_GLOBAL_MAX_PAGES", 10)
        self._max_urls_per_seed = env_int("HIBRIA_CRAWLEE_MAX_URLS_PER_SEED", 5)
        self._max_pages_per_claim = env_int("HIBRIA_CRAWLEE_MAX_PAGES_PER_CLAIM", 3)
        self._targeted_fallback = env_flag("HIBRIA_CRAWLEE_TARGETED_FALLBACK", default=True)
        self._targeted_max_queries = env_int("HIBRIA_CRAWLEE_TARGETED_MAX_QUERIES", 1)
        self._targeted_max_pages = env_int("HIBRIA_CRAWLEE_TARGETED_MAX_PAGES", 2)

        self._prepared = False
        self._global_queries: list[str] = []
        self._global_candidates: list[dict] = []
        self._page_cache: dict[str, dict] = {}

    def is_available(self) -> bool:
        if not self._enabled or time.time() < self._rate_limited_until:
            return False
        try:
            from bs4 import BeautifulSoup  # noqa: F401
        except Exception as exc:
            self._import_error = str(exc)
            logger.warning(
                "[crawlee_search] BeautifulSoup não disponível; instale com: pip install beautifulsoup4 lxml"
            )
            return False
        return True

    def _quota_file(self) -> Path:
        return Path("data/runtime/crawlee_search_quota.json")

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

    def _register_search(self, amount: int = 1) -> None:
        quota = self._read_quota()
        quota["count"] += max(1, amount)
        self._write_quota(quota)

    def _is_same_url(self, url: str) -> bool:
        if env_flag("HIBRIA_ALLOW_SELF_EVIDENCE", default=False):
            return False
        return _is_same_article_url(self._current_url, url)

    def _is_trusted_domain(self, url: str) -> bool:
        domain = extract_domain(url)
        return bool(domain and self._trusted_domains and domain in self._trusted_domains)

    def _seed_templates(self) -> list[str]:
        raw_custom = os.getenv("HIBRIA_CRAWLEE_SEED_URLS", "").strip()
        templates: list[str] = []
        if raw_custom:
            for item in re.split(r"[;,]", raw_custom):
                item = item.strip()
                if item and "{query}" in item:
                    templates.append(item)

        engines = split_env_list("HIBRIA_CRAWLEE_SEARCH_ENGINES") or {"google_news", "bing_news"}
        if env_flag("HIBRIA_CRAWLEE_ADD_GOOGLE_NEWS", default=True):
            engines.add("google_news")
        if env_flag("HIBRIA_CRAWLEE_ADD_BING_NEWS", default=True):
            engines.add("bing_news")

        preferred = ["google_news", "bing_news"]
        for engine in preferred + sorted(engines - set(preferred)):
            template = self.DEFAULT_NEWS_SEARCH_SEEDS.get(engine)
            if template:
                templates.append(template)

        seen: set[str] = set()
        unique: list[str] = []
        for template in templates:
            if template in seen:
                continue
            seen.add(template)
            unique.append(template)
        return unique

    def _seed_urls(self, query: str) -> list[str]:
        encoded = quote_plus(query)
        return [template.format(query=encoded) for template in self._seed_templates()]

    @staticmethod
    def _unwrap_candidate_url(url: str, base_url: str = "") -> str:
        url = normalize_space(url)
        if not url:
            return ""
        if base_url:
            url = urljoin(base_url, url)
        url = unquote(url)
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return ""
        params = parse_qs(parsed.query)
        for key in ("url", "u", "q"):
            value = params.get(key, [""])[0]
            value = unquote(value)
            if value.startswith(("http://", "https://")):
                return value
        return url

    def _candidate_url_allowed(self, url: str, *, allow_news_redirector: bool = False) -> bool:
        if not url:
            return False
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
        domain = extract_domain(url)
        if allow_news_redirector and domain == "news.google.com":
            return True
        if self._is_same_url(url):
            return False
        if _is_blocked_evidence_domain(url):
            return False
        if parsed.path.lower().endswith((
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".mp4",
            ".mp3", ".avi", ".mov", ".zip", ".rar", ".pdf",
        )):
            return False
        return True

    def _http_get(self, url: str, *, stream: bool = False) -> requests.Response | None:
        try:
            response = requests.get(
                url,
                headers=self._HEADERS,
                timeout=self._timeout_seconds,
                allow_redirects=True,
                stream=stream,
            )
        except requests.RequestException as exc:
            if self._debug:
                logger.info(f"[crawlee_search] falha HTTP em {url}: {exc}")
            return None
        if response.status_code >= 400:
            if self._debug:
                logger.info(f"[crawlee_search] HTTP {response.status_code} em {url}")
            try:
                response.close()
            except Exception:
                pass
            return None
        return response

    def _resolve_final_url(self, url: str) -> str:
        try:
            response = requests.get(
                url,
                headers=self._HEADERS,
                timeout=min(self._timeout_seconds, 6),
                allow_redirects=True,
                stream=True,
            )
            final_url = str(response.url or "")
            response.close()
            return final_url if final_url.startswith(("http://", "https://")) else ""
        except Exception:
            return ""

    def _discover_candidate_urls(self, query: str) -> list[dict]:
        import warnings
        from bs4 import BeautifulSoup

        try:
            from bs4 import XMLParsedAsHTMLWarning
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        except Exception:
            pass

        candidates: list[dict] = []
        seen: set[str] = set()
        seed_urls = self._seed_urls(query)

        for seed_url in seed_urls:
            per_seed = 0
            response = self._http_get(seed_url)
            if response is None:
                continue

            text = response.text or ""
            soup = BeautifulSoup(text, "xml")
            items = soup.find_all(["item", "entry"])
            if not items:
                soup = BeautifulSoup(text, "html.parser")
                items = soup.find_all("a", href=True)

            for item in items:
                if per_seed >= self._max_urls_per_seed:
                    break

                title = ""
                summary = ""
                raw_url = ""
                source_domain = ""

                if getattr(item, "name", "") in {"item", "entry"}:
                    title_tag = item.find("title")
                    desc_tag = item.find("description") or item.find("summary")
                    link_tag = item.find("link")
                    guid_tag = item.find("guid")
                    source_tag = item.find("source")

                    title = normalize_space(title_tag.get_text(" ", strip=True) if title_tag else "")
                    summary = normalize_space(desc_tag.get_text(" ", strip=True) if desc_tag else "")
                    if source_tag:
                        source_domain = normalize_space(source_tag.get("url") or source_tag.get_text(" ", strip=True))
                    if link_tag:
                        raw_url = link_tag.get("href") or link_tag.get_text(" ", strip=True)
                    if not raw_url and guid_tag:
                        raw_url = guid_tag.get_text(" ", strip=True)
                else:
                    title = normalize_space(item.get_text(" ", strip=True))
                    raw_url = item.get("href", "")

                url = self._unwrap_candidate_url(raw_url, base_url=seed_url)
                if not url:
                    continue
                if extract_domain(url) == "news.google.com":
                    resolved = self._resolve_final_url(url)
                    url = resolved or url
                if not self._candidate_url_allowed(url):
                    continue

                key = url.strip().rstrip("/")
                if key in seen:
                    continue
                seen.add(key)
                per_seed += 1
                candidates.append({
                    "url": url,
                    "title": title,
                    "summary": summary,
                    "seed_url": seed_url,
                    "source_domain": source_domain,
                    "query": query,
                })
        return candidates

    @staticmethod
    def _extract_text_from_soup(soup) -> tuple[str, str]:
        for tag in soup.find_all([
            "script", "style", "noscript", "nav", "footer", "header", "aside",
            "form", "button", "iframe", "svg",
        ]):
            tag.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        containers = soup.find_all(["article", "main"])
        if not containers:
            containers = [soup]

        parts: list[str] = []
        for container in containers:
            for element in container.find_all(["h1", "h2", "h3", "p", "li"]):
                text = normalize_space(element.get_text(" ", strip=True))
                if len(text) >= 35:
                    parts.append(text)

        if not parts:
            full_text = normalize_space(soup.get_text(" ", strip=True))
            parts = [full_text]
        return title, normalize_space("\n".join(parts))

    def _fetch_candidate_page(self, candidate: dict) -> dict | None:
        from bs4 import BeautifulSoup

        url = normalize_space(candidate.get("url", ""))
        if not url or not self._candidate_url_allowed(url):
            return None

        cached = self._page_cache.get(url)
        if cached:
            return cached

        title = normalize_space(candidate.get("title", ""))
        summary = normalize_space(candidate.get("summary", ""))
        final_url = url
        body = ""

        response = self._http_get(url)
        if response is not None:
            final_url = str(response.url or url)
            if not self._candidate_url_allowed(final_url):
                return None
            content_type = response.headers.get("Content-Type", "").lower()
            if "text" in content_type or "html" in content_type or not content_type:
                soup = BeautifulSoup(response.text or "", "html.parser")
                page_title, body = self._extract_text_from_soup(soup)
                title = page_title or title

        if not body and summary:
            body = summary
        if not body:
            return None

        page = {
            "url": final_url,
            "title": normalize_space(title),
            "body": normalize_space(body),
            "rss_summary": summary,
            "seed_url": candidate.get("seed_url", ""),
            "query": candidate.get("query", ""),
        }
        self._page_cache[url] = page
        self._page_cache[final_url] = page
        return page

    @staticmethod
    def _best_excerpt_for_claim(claim: Claim, title: str, body: str) -> str:
        raw_parts = [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", body or "") if part.strip()]
        candidates: list[str] = []
        for index, part in enumerate(raw_parts):
            if len(part) < 45:
                continue
            window = part
            next_index = index + 1
            while next_index < len(raw_parts) and len(window) < 550:
                window = normalize_space(f"{window} {raw_parts[next_index]}")
                next_index += 1
            candidates.append(window[:850])
        if not candidates and body:
            candidates = [body[:850]]
        if not candidates:
            return ""
        return normalize_space(max(candidates, key=lambda item: _tfidf_similarity(claim.normalized or claim.text, f"{title}. {item}")))

    def _global_query_candidates(self, claims: list[Claim]) -> list[str]:
        all_text = normalize_space(" ".join([self._document_context] + [claim.text for claim in claims]))
        global_entities = _crawlee_text_phrases(all_text, max_phrases=12)

        important_numbers: list[str] = []
        for claim in claims:
            for number in sorted(_crawlee_required_numbers(claim.text)):
                if number not in important_numbers:
                    important_numbers.append(number)

        queries: list[str] = []
        base = normalize_space(" ".join(global_entities[:6] + important_numbers[:6]))
        if base:
            queries.append(base)

        for claim in claims:
            entities = _crawlee_strong_entities(claim, self._document_context)[:4]
            numbers = sorted(_crawlee_required_numbers(claim.text))[:4]
            tokens = list(_crawlee_phrase_tokens(claim.text))[:8]
            query = normalize_space(" ".join(entities + numbers + tokens[:6]))
            if query:
                queries.append(query)

        suffix = os.getenv("HIBRIA_CRAWLEE_QUERY_SUFFIX", "notícia jornal").strip()
        clean: list[str] = []
        seen: set[str] = set()
        for query in queries:
            query = normalize_space(query)
            if not query:
                continue
            if suffix and suffix not in query:
                query = normalize_space(f"{query} {suffix}")
            query = query[:180]
            key = _plain(query)
            if key in seen:
                continue
            seen.add(key)
            clean.append(query)
        return clean[: self._max_global_queries]

    @staticmethod
    def _candidate_light_score(candidate: dict, global_text: str) -> float:
        candidate_text = normalize_space(" ".join([candidate.get("title", ""), candidate.get("summary", ""), candidate.get("url", "")]))
        global_tokens = _crawlee_phrase_tokens(global_text)
        candidate_tokens = _significant_tokens(candidate_text)
        if not global_tokens or not candidate_tokens:
            return 0.0
        overlap = global_tokens & candidate_tokens
        return len(overlap) / max(1, min(len(global_tokens), 24))

    def prepare_for_claims(self, claims: list[Claim]) -> None:
        if self._prepared or not self.is_available():
            return
        self._prepared = True

        if not self._can_search():
            self._rate_limited_until = time.time() + 10 * 60
            logger.warning("[crawlee_search] limite diário local atingido; camada pausada nesta execução")
            return

        self._global_queries = self._global_query_candidates(claims)
        if not self._global_queries:
            return

        all_candidates: list[dict] = []
        seen_urls: set[str] = set()
        for query in self._global_queries:
            for candidate in self._discover_candidate_urls(query):
                url = normalize_space(candidate.get("url", ""))
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                all_candidates.append(candidate)
                if len(all_candidates) >= self._max_global_urls:
                    break
            if len(all_candidates) >= self._max_global_urls:
                break

        global_text = normalize_space(" ".join([self._document_context] + [claim.text for claim in claims]))
        all_candidates.sort(key=lambda item: self._candidate_light_score(item, global_text), reverse=True)
        self._global_candidates = all_candidates[: self._max_global_urls]

        opened = 0
        for candidate in self._global_candidates:
            if opened >= self._max_global_pages:
                break
            page = self._fetch_candidate_page(candidate)
            if page is None:
                continue
            opened += 1
            time.sleep(self._sleep_seconds)
        self._register_search()

        if self._debug:
            logger.info(
                "[crawlee_search] cache global preparado: "
                f"queries={len(self._global_queries)} candidates={len(self._global_candidates)} pages={len(self._page_cache)}"
            )

    def _claim_query_candidates(self, claim: Claim) -> list[str]:
        entities = _crawlee_strong_entities(claim, self._document_context)[:5]
        numbers = sorted(_crawlee_required_numbers(claim.text))[:5]
        tokens = list(_crawlee_phrase_tokens(claim.text))[:8]
        query = normalize_space(" ".join(entities + numbers + tokens[:6]))
        if not query:
            query = _build_claim_search_query(claim, document_context=self._document_context, max_chars=140)
        suffix = os.getenv("HIBRIA_CRAWLEE_QUERY_SUFFIX", "notícia jornal").strip()
        if suffix:
            query = normalize_space(f"{query} {suffix}")
        return [query[:180]] if query else []

    def _pages_for_claim(self, claim: Claim) -> list[dict]:
        pages: list[dict] = []
        seen: set[str] = set()
        for page in self._page_cache.values():
            url = normalize_space(page.get("url", ""))
            if not url or url in seen:
                continue
            seen.add(url)
            title = normalize_space(page.get("title", ""))
            body = normalize_space(page.get("body", ""))
            excerpt = self._best_excerpt_for_claim(claim, title, body)
            if not excerpt:
                continue
            evidence_text = normalize_space(f"{title}. {excerpt}" if title else excerpt)
            ok, reason = _crawlee_strict_filter(self._document_context, claim, evidence_text, url)
            if ok:
                pages.append({**page, "text": excerpt, "strict_filter": reason, "from_global_cache": True})
        return pages

    def _targeted_pages_for_claim(self, claim: Claim) -> list[dict]:
        if not self._targeted_fallback or not self._can_search():
            return []
        pages: list[dict] = []
        opened = 0
        for query in self._claim_query_candidates(claim)[: self._targeted_max_queries]:
            for candidate in self._discover_candidate_urls(query):
                if opened >= self._targeted_max_pages:
                    break
                page = self._fetch_candidate_page(candidate)
                if page is None:
                    continue
                opened += 1
                title = normalize_space(page.get("title", ""))
                body = normalize_space(page.get("body", ""))
                excerpt = self._best_excerpt_for_claim(claim, title, body)
                if not excerpt:
                    continue
                evidence_text = normalize_space(f"{title}. {excerpt}" if title else excerpt)
                ok, reason = _crawlee_strict_filter(self._document_context, claim, evidence_text, page.get("url", ""))
                if ok:
                    pages.append({**page, "text": excerpt, "strict_filter": reason, "from_global_cache": False, "query": query})
                time.sleep(self._sleep_seconds)
            if pages or opened >= self._targeted_max_pages:
                break
        if opened:
            self._register_search()
        return pages

    def search(self, claim: Claim, top_k: int = 5) -> list[Evidence]:
        if not self.is_available():
            return []
        if not self._prepared:
            self.prepare_for_claims([claim])

        pages = self._pages_for_claim(claim)
        if not pages:
            pages = self._targeted_pages_for_claim(claim)

        evidences: list[Evidence] = []
        seen_urls: set[str] = set()
        for page in pages:
            url = normalize_space(page.get("url", ""))
            title = normalize_space(page.get("title", ""))
            excerpt = normalize_space(page.get("text", ""))
            if not url or url in seen_urls:
                continue
            if not self._candidate_url_allowed(url):
                continue

            evidence_text = normalize_space(f"{title}. {excerpt}" if title else excerpt)
            ok, reason = _crawlee_strict_filter(self._document_context, claim, evidence_text, url)
            if not ok:
                if self._debug:
                    logger.info(f"[crawlee_search] descartada: {reason} | {url} | {evidence_text[:260]}")
                continue

            seen_urls.add(url)
            domain = extract_domain(url)
            trusted = self._is_trusted_domain(url)
            evidences.append(
                Evidence(
                    text=evidence_text[:1200],
                    source=domain or "Crawlee Search",
                    url=url,
                    similarity=_crawlee_similarity(self._document_context, claim, evidence_text),
                    claim_id=claim.claim_id,
                    title=title,
                    domain=domain,
                    retrieval_layer=self.name,
                    source_type="crawled_news",
                    trusted_source=trusted,
                    metadata={
                        "provider": CRAWLEE_PROVIDER_NAME,
                        "query": page.get("query", ""),
                        "global_queries": self._global_queries,
                        "candidate_urls": [item.get("url", "") for item in self._global_candidates[:20]],
                        "rss_summary": page.get("rss_summary", ""),
                        "trusted_domain": trusted,
                        "from_global_cache": bool(page.get("from_global_cache", True)),
                        "strict_filter": reason,
                        "required_numbers": sorted(_crawlee_required_numbers(claim.text)),
                        "required_entities": _crawlee_strong_entities(claim, self._document_context)[:8],
                        "number_entity_pairs": [
                            f"{'/'.join(sorted(entity_tokens))}+{number}"
                            for entity_tokens, number in _crawlee_number_entity_pairs(claim, self._document_context)
                        ],
                        "note": "Camada gratuita por Google News RSS + Bing News RSS e crawling direcionado; consultas e filtros gerados dinamicamente a partir da claim e do contexto da notícia.",
                    },
                )
            )
            if len(evidences) >= top_k:
                break

        evidences.sort(key=lambda evidence: evidence.similarity, reverse=True)
        if not evidences and self._debug:
            logger.info(f"[crawlee_search] sem evidências aprovadas para claim={claim.claim_id}")
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

        return _is_same_article_url(self._current_url, url)

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
        if not _is_relevant_candidate(claim, evidence_text, evidence_url):
            return []

        similarity = _candidate_similarity(claim, evidence_text)

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
    MIN_SIMILARITY: float = env_float("HIBRIA_RETRIEVER_MIN_SIMILARITY", 0.12)

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
        "crawlee_search": 0.75,
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
            CrawleeSearchSource(
                current_url=current_url,
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
        for source in self._sources:
            prepare = getattr(source, "prepare_for_claims", None)
            if callable(prepare) and source.is_available():
                try:
                    prepare(claims)
                except Exception as exc:
                    logger.warning("[%s] falha na preparação global: %s", source.name, exc)

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
                        "crawlee_search",
                        "wikipedia",
                        "ai_fallback",
                    }
                ):
                    time.sleep(
                        float(os.getenv("HIBRIA_RETRIEVAL_SLEEP_SECONDS", "0.3"))
                    )

        return results
