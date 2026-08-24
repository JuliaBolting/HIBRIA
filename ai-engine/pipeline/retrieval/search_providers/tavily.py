from __future__ import annotations

import os
import requests

from .config import DEFAULT_TIMEOUT, env_flag, env_int, valid_api_key
from .quota import DailyQuota
from .models import SearchHit


class TavilySearchProvider:
    name = "tavily"
    API_URL = "https://api.tavily.com/search"

    def __init__(self) -> None:
        self.api_key = os.getenv("TAVILY_API_KEY", "").strip()
        self.enabled = env_flag("HIBRIA_ENABLE_TAVILY", False) or env_flag("HIBRIA_REPUTATION_USE_CONFIGURED_PROVIDERS", True)
        self.daily_limit = env_int("HIBRIA_TAVILY_DAILY_LIMIT", 80)

    def is_available(self) -> bool:
        return self.enabled and valid_api_key(self.api_key) and DailyQuota.can_use(self.name, self.daily_limit)

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        response = requests.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "search_depth": os.getenv("HIBRIA_TAVILY_SEARCH_DEPTH", "basic"),
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=max(DEFAULT_TIMEOUT, 15),
        )
        response.raise_for_status()
        DailyQuota.register(self.name)
        data = response.json()

        hits: list[SearchHit] = []
        for item in data.get("results", []) or []:
            url = item.get("url", "")
            if not url:
                continue
            hits.append(SearchHit(
                provider=self.name,
                title=item.get("title", "") or url,
                url=url,
                snippet=item.get("content", "") or "",
                metadata={"score": item.get("score")},
            ))
        return hits[:max_results]
