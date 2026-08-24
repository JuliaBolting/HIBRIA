from __future__ import annotations

import os
import requests

from .config import DEFAULT_TIMEOUT, env_flag, env_int, valid_api_key
from .quota import DailyQuota
from .models import SearchHit


class SerpSearchProvider:
    """Usa Serper, SerpApi ou SearchAPI, na ordem configurada no .env."""

    name = "serp"

    def __init__(self) -> None:
        self.serper_key = os.getenv("SERPER_API_KEY", "").strip()
        self.serpapi_key = os.getenv("SERPAPI_API_KEY", "").strip()
        self.searchapi_key = os.getenv("SEARCHAPI_API_KEY", "").strip()
        self.daily_limit = env_int("HIBRIA_SERP_SEARCH_DAILY_LIMIT", 80)

    def _selected_provider(self) -> str | None:
        reputation_auto = env_flag("HIBRIA_REPUTATION_USE_CONFIGURED_PROVIDERS", True)
        if (env_flag("HIBRIA_ENABLE_SERPER", False) or reputation_auto) and valid_api_key(self.serper_key):
            return "serper"
        if (env_flag("HIBRIA_ENABLE_SERPAPI", False) or reputation_auto) and valid_api_key(self.serpapi_key):
            return "serpapi"
        if (env_flag("HIBRIA_ENABLE_SEARCHAPI", False) or reputation_auto) and valid_api_key(self.searchapi_key):
            return "searchapi"
        return None

    def is_available(self) -> bool:
        return self._selected_provider() is not None and DailyQuota.can_use("serp", self.daily_limit)

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        provider = self._selected_provider()
        if provider == "serper":
            return self._search_serper(query, max_results)
        if provider == "serpapi":
            return self._search_serpapi(query, max_results)
        if provider == "searchapi":
            return self._search_searchapi(query, max_results)
        return []

    def _search_serper(self, query: str, max_results: int) -> list[SearchHit]:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": self.serper_key, "Content-Type": "application/json"},
            json={"q": query, "num": max_results, "gl": "br", "hl": "pt-br"},
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        DailyQuota.register("serp")
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
        DailyQuota.register("serp")
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
        DailyQuota.register("serp")
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
