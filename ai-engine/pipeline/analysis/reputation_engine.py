# =============================================================================
# Avalia a reputação do domínio da notícia analisada.
#
# Nesta versão inicial, a reputação é calculada a partir de uma lista controlada
# de domínios jornalísticos confiáveis. Futuramente, essa lista poderá vir do
# PostgreSQL, pela tabela fontes_confiaveis.
#
# Entrada:
#   - URL da notícia analisada
#
# Saída:
#   - dict com domínio, categoria, score e justificativa
# =============================================================================

from __future__ import annotations

import os
from urllib.parse import urlparse

DEFAULT_TRUSTED_DOMAINS = {
    "g1.globo.com",
    "oglobo.globo.com",
    "folha.uol.com.br",
    "estadao.com.br",
    "bbc.com",
    "agenciabrasil.ebc.com.br",
}


def extract_domain(url: str) -> str:
    parsed = urlparse(url or "")
    domain = parsed.netloc.lower().strip()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def load_trusted_domains() -> set[str]:
    """
    Carrega domínios confiáveis do .env, se existirem.
    Caso contrário, usa uma lista padrão inicial.
    """
    raw = os.getenv("HIBRIA_TRUSTED_DOMAINS", "")

    env_domains = {
        item.strip().lower().replace("www.", "")
        for item in raw.split(",")
        if item.strip()
    }

    return env_domains or DEFAULT_TRUSTED_DOMAINS


class ReputationEngine:
    """
    Avalia a reputação da fonte da notícia.

    A reputação não decide sozinha se a notícia é confiável.
    Ela é apenas um dos critérios usados futuramente pelo aggregator.py.
    """

    @staticmethod
    def evaluate(url: str) -> dict:
        domain = extract_domain(url)
        trusted_domains = load_trusted_domains()

        if not domain:
            return {
                "domain": "",
                "is_trusted_domain": False,
                "source_category": "unknown",
                "score": 0.0,
                "reason": "Não foi possível identificar o domínio da URL.",
            }

        is_trusted = domain in trusted_domains

        if is_trusted:
            return {
                "domain": domain,
                "is_trusted_domain": True,
                "source_category": "trusted_news_source",
                "score": 0.9,
                "reason": (
                    "O domínio da notícia está cadastrado na base de fontes "
                    "jornalísticas confiáveis do sistema."
                ),
            }

        return {
            "domain": domain,
            "is_trusted_domain": False,
            "source_category": "unverified_source",
            "score": 0.4,
            "reason": (
                "O domínio da notícia não está cadastrado na base de fontes "
                "confiáveis do sistema."
            ),
        }
