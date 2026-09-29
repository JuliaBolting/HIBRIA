"""Nunca serialize mensagens brutas de exceções de serviços externos."""
import os
import re


def error_summary(exc):
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return f"Falha HTTP {status}." if isinstance(status, int) else f"Falha externa ({type(exc).__name__})."


def redact(value):
    if isinstance(value, dict):
        return {k: ("[removido]" if re.search(r"api[_-]?key|password|authorization|secret|token", str(k), re.I)
                    else {name: "Serviço indisponível." for name in v} if k in {"provider_failures", "providers_failed"} and isinstance(v, dict)
                    else redact(v)) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)([?&](?:api[_-]?key|key|token|access_token|secret)=)[^&#\s]+", r"\1[removido]", value)
        for name, secret in os.environ.items():
            if len(secret) >= 8 and re.search(r"(?:KEY|TOKEN|SECRET|PASSWORD|DATABASE_URL)", name):
                value = value.replace(secret, "[removido]")
    return value
