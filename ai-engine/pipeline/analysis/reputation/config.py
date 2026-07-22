from __future__ import annotations

import os
from pathlib import Path


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "sim", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def project_root() -> Path:
    # .../HIBRIA/ai-engine/pipeline/analysis/reputation/config.py -> ai-engine
    return Path(__file__).resolve().parents[3]


METHOD_NAME = "dynamic_weighted_source_reputation_v1"
DYNAMIC_ENABLED = env_flag("HIBRIA_REPUTATION_DYNAMIC_ENABLED", True)
CACHE_TTL_DAYS = env_int("HIBRIA_REPUTATION_CACHE_TTL_DAYS", 180)
INSUFFICIENT_TTL_DAYS = env_int("HIBRIA_REPUTATION_INSUFFICIENT_TTL_DAYS", 7)
MIN_EVIDENCE_URLS = env_int("HIBRIA_REPUTATION_MIN_EVIDENCE_URLS", 4)
MIN_CONFIRMED_CRITERIA = env_int("HIBRIA_REPUTATION_MIN_CONFIRMED_CRITERIA", 4)
MAX_RESULTS_PER_QUERY = env_int("HIBRIA_REPUTATION_MAX_RESULTS_PER_QUERY", 5)
MAX_PROVIDERS_PER_QUERY = env_int("HIBRIA_REPUTATION_MAX_PROVIDERS_PER_QUERY", 2)
SEARCH_SLEEP_SECONDS = env_float("HIBRIA_REPUTATION_SEARCH_SLEEP_SECONDS", 0.5)
DIRECT_FETCH_TIMEOUT = env_float("HIBRIA_REPUTATION_DIRECT_FETCH_TIMEOUT", 10.0)
ENABLE_GEMINI_REVIEW = env_flag("HIBRIA_REPUTATION_ENABLE_GEMINI_REVIEW", False)

DATABASE_URL = os.getenv("HIBRIA_DATABASE_URL") or os.getenv("DATABASE_URL") or ""
JSON_STORAGE_PATH = Path(os.getenv(
    "HIBRIA_REPUTATION_JSON_PATH",
    str(project_root() / "data" / "reputation" / "source_reputation.json"),
))
