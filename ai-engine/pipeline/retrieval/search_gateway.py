# =============================================================================
# Gateway compartilhado de buscas externas.
#
# A reputação usa os mesmos provedores configurados no RAG. O gateway aceita
# consultas genéricas e não aplica os filtros de relevância específicos de claims.
# =============================================================================

from __future__ import annotations

import logging
from typing import Iterable

from .search_providers import (
    BingSearchProvider,
    BraveSearchProvider,
    GoogleFactCheckProvider,
    SearchBatch,
    SearchHit,
    SerpSearchProvider,
    TavilySearchProvider,
)
from .search_providers.config import DEFAULT_MAX_PROVIDERS, DEFAULT_MAX_RESULTS

logger = logging.getLogger(__name__)


class SearchGateway:
    """Executa uma consulta em cadeia de fallback e devolve resultados deduplicados."""

    def __init__(self) -> None:
        self._providers = {
            "brave": BraveSearchProvider(),
            "tavily": TavilySearchProvider(),
            "serp": SerpSearchProvider(),
            "bing": BingSearchProvider(),
            "google_factcheck": GoogleFactCheckProvider(),
        }

    def available_providers(self) -> list[str]:
        return [name for name, provider in self._providers.items() if provider.is_available()]

    def search(
        self,
        query: str,
        *,
        max_results: int = DEFAULT_MAX_RESULTS,
        min_results: int = 2,
        max_providers: int = DEFAULT_MAX_PROVIDERS,
        provider_order: Iterable[str] | None = None,
    ) -> SearchBatch:
        order = list(provider_order or ("brave", "tavily", "serp", "bing"))
        batch = SearchBatch(query=query)
        seen_urls: set[str] = set()

        for name in order:
            if len(batch.providers_attempted) >= max_providers:
                break

            provider = self._providers.get(name)
            if provider is None or not provider.is_available():
                continue

            batch.providers_attempted.append(name)
            try:
                hits = provider.search(query, max_results=max_results)
                batch.providers_succeeded.append(name)
            except Exception as exc:
                batch.providers_failed[name] = f"{type(exc).__name__}: {exc}"
                logger.warning("[search_gateway] %s falhou: %s", name, exc)
                continue

            for hit in hits:
                key = hit.url.strip().lower().rstrip("/")
                if not key or key in seen_urls:
                    continue
                seen_urls.add(key)
                batch.hits.append(hit)
                if len(batch.hits) >= max_results:
                    break

            if len(batch.hits) >= min_results:
                break

        return batch

    def search_factchecks(self, query: str, *, max_results: int = 5) -> SearchBatch:
        batch = SearchBatch(query=query)
        provider = self._providers["google_factcheck"]
        if not provider.is_available():
            return batch

        batch.providers_attempted.append("google_factcheck")
        try:
            batch.hits = provider.search(query, max_results=max_results)
            batch.providers_succeeded.append("google_factcheck")
        except Exception as exc:
            batch.providers_failed["google_factcheck"] = f"{type(exc).__name__}: {exc}"
        return batch
