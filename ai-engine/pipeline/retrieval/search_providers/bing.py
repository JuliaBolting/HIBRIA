from __future__ import annotations

import os
import requests

from .config import DEFAULT_TIMEOUT, env_flag, valid_api_key
from .quota import ProviderQuota
from .models import SearchHit


class BingSearchProvider:
    name = "bing"

    def __init__(self) -> None:
        self.api_key = (
            os.getenv("BING_SEARCH_API_KEY", "").strip()
            or os.getenv("AZURE_BING_SEARCH_KEY", "").strip()
        )
        self.endpoint = os.getenv(
            "BING_SEARCH_ENDPOINT",
            "https://api.bing.microsoft.com/v7.0/search",
        ).strip()
        # Se a chave existe, o provedor pode ser usado. A flag opcional permite desligar.
        self.enabled = env_flag("HIBRIA_ENABLE_BING_SEARCH", bool(self.api_key))

    def is_available(self) -> bool:
        return self.enabled and valid_api_key(self.api_key) and ProviderQuota.can_use(self.name)

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        if not ProviderQuota.try_register(self.name):
            return []
        response = requests.get(
            self.endpoint,
            headers={"Ocp-Apim-Subscription-Key": self.api_key},
            params={"q": query, "mkt": "pt-BR", "count": max_results},
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return [
            SearchHit(
                provider=self.name,
                title=item.get("name", "") or item.get("url", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", "") or "",
                published_at=item.get("dateLastCrawled"),
            )
            for item in data.get("webPages", {}).get("value", []) or []
            if item.get("url")
        ][:max_results]
