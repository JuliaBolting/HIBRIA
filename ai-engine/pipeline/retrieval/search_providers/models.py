from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHit:
    """Resultado genérico recuperado por um provedor externo."""

    provider: str
    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "published_at": self.published_at,
            "metadata": self.metadata,
        }


@dataclass
class SearchBatch:
    """Conjunto de resultados e diagnóstico de uma consulta."""

    query: str
    hits: list[SearchHit] = field(default_factory=list)
    providers_attempted: list[str] = field(default_factory=list)
    providers_succeeded: list[str] = field(default_factory=list)
    providers_failed: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "hits": [hit.to_dict() for hit in self.hits],
            "providers_attempted": self.providers_attempted,
            "providers_succeeded": self.providers_succeeded,
            "providers_failed": self.providers_failed,
        }
