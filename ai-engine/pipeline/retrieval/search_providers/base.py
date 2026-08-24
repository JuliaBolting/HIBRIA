from __future__ import annotations

from typing import Protocol

from .models import SearchHit


class SearchProvider(Protocol):
    name: str

    def is_available(self) -> bool:
        ...

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        ...
