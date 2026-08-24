from __future__ import annotations

import os
import requests

from .config import DEFAULT_TIMEOUT, env_flag, env_int, valid_api_key
from .quota import DailyQuota
from .models import SearchHit


class BraveSearchProvider:
    name = "brave"
    API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self) -> None:
        self.api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        self.enabled = env_flag("HIBRIA_ENABLE_WEB_SEARCH", False) or env_flag("HIBRIA_REPUTATION_USE_CONFIGURED_PROVIDERS", True)
        self.daily_limit = env_int("HIBRIA_WEB_SEARCH_DAILY_LIMIT", 80)

    def is_available(self) -> bool:
        return self.enabled and valid_api_key(self.api_key) and DailyQuota.can_use(self.name, self.daily_limit)

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        response = requests.get(
            self.API_URL,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
            params={
                "q": query,
                "country": "BR",
                "search_lang": "pt-br",
                "count": min(max_results, 10),
            },
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        DailyQuota.register(self.name)
        data = response.json()

        hits: list[SearchHit] = []
        for item in data.get("web", {}).get("results", []) or []:
            url = item.get("url", "")
            if not url:
                continue
            hits.append(SearchHit(
                provider=self.name,
                title=item.get("title", "") or url,
                url=url,
                snippet=item.get("description", "") or "",
                metadata={"profile": item.get("profile")},
            ))
        return hits[:max_results]
