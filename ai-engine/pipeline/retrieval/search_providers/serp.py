from __future__ import annotations

import os
import requests

from .config import DEFAULT_TIMEOUT, env_flag, valid_api_key
from .quota import ProviderQuota
from .models import SearchHit


class SerpSearchProvider:
    """Usa Serper, SerpApi ou SearchAPI, na ordem configurada no .env."""

    name = "serp"

    def __init__(self) -> None:
        self.serper_key = os.getenv("SERPER_API_KEY", "").strip()
        self.serpapi_key = os.getenv("SERPAPI_API_KEY", "").strip()
        self.searchapi_key = os.getenv("SEARCHAPI_API_KEY", "").strip()

    def _provider_candidates(self) -> list[str]:
        reputation_auto = env_flag("HIBRIA_REPUTATION_USE_CONFIGURED_PROVIDERS", True)
        candidates: list[str] = []
        if (
            env_flag("HIBRIA_ENABLE_SERPER", False) or reputation_auto
        ) and valid_api_key(self.serper_key):
            candidates.append("serper")
        if (
            env_flag("HIBRIA_ENABLE_SERPAPI", False) or reputation_auto
        ) and valid_api_key(self.serpapi_key):
            candidates.append("serpapi")
        if (
            env_flag("HIBRIA_ENABLE_SEARCHAPI", False) or reputation_auto
        ) and valid_api_key(self.searchapi_key):
            candidates.append("searchapi")
        return candidates

    def is_available(self) -> bool:
        return any(
            ProviderQuota.can_use(provider)
            for provider in self._provider_candidates()
        )

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        last_error: requests.RequestException | None = None

        for provider in self._provider_candidates():
            if not ProviderQuota.try_register(provider):
                continue

            try:
                if provider == "serper":
                    return self._search_serper(query, max_results)
                if provider == "serpapi":
                    return self._search_serpapi(query, max_results)
                return self._search_searchapi(query, max_results)
            except requests.RequestException as exc:
                last_error = exc

        if last_error is not None:
            raise last_error
        return []

    def _search_serper(self, query: str, max_results: int) -> list[SearchHit]:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": self.serper_key, "Content-Type": "application/json"},
            json={"q": query, "num": max_results, "gl": "br", "hl": "pt-br"},
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return [
            SearchHit(
                provider="serper",
                title=item.get("title", "") or item.get("link", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", "") or "",
            )
            for item in data.get("organic", []) or []
            if item.get("link")
        ][:max_results]

    def _search_serpapi(self, query: str, max_results: int) -> list[SearchHit]:
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "api_key": self.serpapi_key,
                "engine": "google",
                "q": query,
                "google_domain": "google.com.br",
                "hl": "pt-br",
                "gl": "br",
                "num": max_results,
            },
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return [
            SearchHit(
                provider="serpapi",
                title=item.get("title", "") or item.get("link", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", "") or "",
            )
            for item in data.get("organic_results", []) or []
            if item.get("link")
        ][:max_results]

    def _search_searchapi(self, query: str, max_results: int) -> list[SearchHit]:
        response = requests.get(
            "https://www.searchapi.io/api/v1/search",
            params={
                "api_key": self.searchapi_key,
                "engine": "google",
                "q": query,
                "google_domain": "google.com.br",
                "hl": "pt-br",
                "gl": "br",
                "num": max_results,
            },
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return [
            SearchHit(
                provider="searchapi",
                title=item.get("title", "") or item.get("link", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", "") or "",
            )
            for item in data.get("organic_results", []) or []
            if item.get("link")
        ][:max_results]
