from __future__ import annotations

from datetime import datetime
import json
import os
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


class ProviderQuota:
    """Contador compartilhado para provedores com franquias diferentes.

    Os contadores de Serper, SerpApi e SearchAPI são usados tanto pelo RAG
    quanto pela avaliação de reputação. Limite igual a zero significa que o
    respectivo período não será limitado localmente.
    """

    _lock = threading.Lock()
    _path = Path(
        os.getenv(
            "HIBRIA_PROVIDER_QUOTA_PATH",
            "data/runtime/search_provider_quotas.json",
        )
    )
    _default_limits = {
        "serper": {"daily": 0, "monthly": 0, "total": 2500},
        "serpapi": {"daily": 0, "monthly": 250, "total": 0},
        "searchapi": {"daily": 0, "monthly": 0, "total": 100},
    }

    @classmethod
    def can_use(cls, provider: str) -> bool:
        with cls._lock:
            data = cls._read()
            return cls._within_limits(provider, data)

    @classmethod
    def try_register(cls, provider: str) -> bool:
        """Reserva uma consulta de forma atômica antes da chamada externa."""
        with cls._lock:
            data = cls._read()
            if not cls._within_limits(provider, data):
                return False

            counts = data.setdefault("providers", {}).setdefault(
                provider,
                {"daily": 0, "monthly": 0, "total": 0},
            )
            counts["daily"] = int(counts.get("daily", 0)) + 1
            counts["monthly"] = int(counts.get("monthly", 0)) + 1
            counts["total"] = int(counts.get("total", 0)) + 1
            cls._write(data)
            return True

    @classmethod
    def usage(cls, provider: str) -> dict[str, int]:
        with cls._lock:
            data = cls._read()
            counts = data.get("providers", {}).get(provider, {})
            return {
                "daily": int(counts.get("daily", 0)),
                "monthly": int(counts.get("monthly", 0)),
                "total": int(counts.get("total", 0)),
            }

    @classmethod
    def limits(cls, provider: str) -> dict[str, int]:
        prefix = f"HIBRIA_{provider.upper()}"
        defaults = cls._default_limits.get(
            provider,
            {"daily": 0, "monthly": 0, "total": 0},
        )
        return {
            period: cls._env_int(f"{prefix}_{period.upper()}_LIMIT", default)
            for period, default in defaults.items()
        }

    @classmethod
    def _within_limits(cls, provider: str, data: dict) -> bool:
        counts = data.get("providers", {}).get(provider, {})
        limits = cls.limits(provider)
        return all(
            limit <= 0 or int(counts.get(period, 0)) < limit
            for period, limit in limits.items()
        )

    @classmethod
    def _read(cls) -> dict:
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")
        payload: dict = {}

        if cls._path.exists():
            try:
                payload = json.loads(cls._path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}

        providers = payload.get("providers") or {}
        normalized: dict[str, dict[str, int]] = {}
        for provider, counts in providers.items():
            normalized[provider] = {
                "daily": (
                    int(counts.get("daily", 0))
                    if payload.get("date") == today
                    else 0
                ),
                "monthly": (
                    int(counts.get("monthly", 0))
                    if payload.get("month") == month
                    else 0
                ),
                "total": int(counts.get("total", 0)),
            }

        return {
            "date": today,
            "month": month,
            "providers": normalized,
        }

    @classmethod
    def _write(cls, data: dict) -> None:
        cls._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cls._path.with_suffix(f"{cls._path.suffix}.tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(cls._path)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            return default
        try:
            return max(0, int(raw))
        except ValueError:
            return default
