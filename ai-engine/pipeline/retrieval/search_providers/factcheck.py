from __future__ import annotations

import os
import time
import requests

from .config import DEFAULT_TIMEOUT, env_flag, env_float, env_int, valid_api_key
from .quota import DailyQuota
from .models import SearchHit


class GoogleFactCheckProvider:
    name = "google_factcheck"
    API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(self) -> None:
        self.api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip()
        self.enabled = env_flag("HIBRIA_ENABLE_FACTCHECK", False) or env_flag("HIBRIA_REPUTATION_USE_CONFIGURED_PROVIDERS", True)
        self.daily_limit = env_int("HIBRIA_FACTCHECK_DAILY_LIMIT", 100)
        self.sleep_seconds = env_float("HIBRIA_FACTCHECK_SLEEP_SECONDS", 0.5)

    def is_available(self) -> bool:
        return self.enabled and valid_api_key(self.api_key) and DailyQuota.can_use(self.name, self.daily_limit)

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        response = requests.get(
            self.API_URL,
            params={
                "query": query,
                "languageCode": "pt",
                "pageSize": min(max_results, 10),
                "key": self.api_key,
            },
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        DailyQuota.register(self.name)
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        data = response.json()

        hits: list[SearchHit] = []
        for claim in data.get("claims", []) or []:
            claim_text = claim.get("text", "")
            claimant = claim.get("claimant", "")
            for review in claim.get("claimReview", []) or []:
                publisher = (review.get("publisher") or {}).get("name", "")
                rating = review.get("textualRating", "")
                url = review.get("url", "")
                if not url:
                    continue
                snippet = "; ".join(part for part in [
                    f"alegação: {claim_text}" if claim_text else "",
                    f"autor da alegação: {claimant}" if claimant else "",
                    f"checador: {publisher}" if publisher else "",
                    f"classificação: {rating}" if rating else "",
                ] if part)
                hits.append(SearchHit(
                    provider=self.name,
                    title=review.get("title", "") or claim_text or "Checagem encontrada",
                    url=url,
                    snippet=snippet,
                    published_at=review.get("reviewDate") or claim.get("claimDate"),
                    metadata={"publisher": publisher, "rating": rating},
                ))
        return hits[:max_results]
