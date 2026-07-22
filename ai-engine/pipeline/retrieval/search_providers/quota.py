from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import threading


class DailyQuota:
    """Contador local simples compartilhado pelos provedores da reputação."""

    _lock = threading.Lock()
    _path = Path("data/runtime/reputation_search_quotas.json")

    @classmethod
    def can_use(cls, provider: str, limit: int) -> bool:
        if limit <= 0:
            return True
        with cls._lock:
            data = cls._read()
            return int(data.get(provider, 0)) < limit

    @classmethod
    def register(cls, provider: str) -> None:
        with cls._lock:
            data = cls._read()
            data[provider] = int(data.get(provider, 0)) + 1
            cls._path.parent.mkdir(parents=True, exist_ok=True)
            cls._path.write_text(json.dumps({"date": cls._today(), "counts": data}, indent=2), encoding="utf-8")

    @classmethod
    def _today(cls) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    @classmethod
    def _read(cls) -> dict[str, int]:
        if not cls._path.exists():
            return {}
        try:
            payload = json.loads(cls._path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if payload.get("date") != cls._today():
            return {}
        return {key: int(value) for key, value in (payload.get("counts") or {}).items()}
