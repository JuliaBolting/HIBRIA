from __future__ import annotations

import os


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


def valid_api_key(value: str) -> bool:
    value = (value or "").strip()
    return bool(value and value.lower() not in {
        "sua_chave_aqui",
        "your_key_here",
        "sua_chave_serper_aqui",
    })


DEFAULT_TIMEOUT = env_float("HIBRIA_SEARCH_TIMEOUT_SECONDS", 12.0)
DEFAULT_MAX_RESULTS = env_int("HIBRIA_SEARCH_MAX_RESULTS", 5)
DEFAULT_MAX_PROVIDERS = env_int("HIBRIA_SEARCH_MAX_PROVIDERS_PER_QUERY", 2)
