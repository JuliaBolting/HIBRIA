from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .config import DIRECT_FETCH_TIMEOUT
from .models import SourceIdentity


def normalize_domain(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"^https?://", "", value)
    value = value.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    value = value.rsplit("@", 1)[-1].split(":", 1)[0].strip(". ")
    while value.startswith("www."):
        value = value[4:]
    return value


def ensure_url(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if value.startswith(("http://", "https://")):
        return value
    return f"https://{value}"


def domain_from_url(value: str) -> str:
    parsed = urlparse(ensure_url(value))
    return normalize_domain(parsed.netloc or parsed.path)


def _clean_source_name(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    value = re.sub(r"\s*[-|–—]\s*(notícias|noticias|news|home).*$", "", value, flags=re.I)
    return value[:160]


def _fallback_name(domain: str) -> str:
    labels = [part for part in normalize_domain(domain).split(".") if part]
    if not labels:
        return ""
    ignored = {"com", "org", "net", "gov", "edu", "br", "news", "jor"}
    meaningful = [part for part in labels if part not in ignored]
    core = meaningful[0] if meaningful else labels[0]
    return core.replace("-", " ").title()


class SourceIdentityResolver:
    """Resolve domínio canônico e nome da fonte sem listas pré-definidas."""

    USER_AGENT = "HibriaSourceReputation/1.0"

    def resolve(self, url_or_domain: str) -> SourceIdentity:
        requested_url = ensure_url(url_or_domain)
        requested_domain = domain_from_url(requested_url)

        if not requested_domain:
            return SourceIdentity(
                requested_url=requested_url,
                requested_domain="",
                canonical_url=requested_url,
                canonical_domain="",
                source_name="",
            )

        root_url = f"https://{requested_domain}/"
        response = None
        redirect_chain: list[str] = []
        aliases: set[str] = {requested_domain}

        try:
            response = requests.get(
                root_url,
                timeout=DIRECT_FETCH_TIMEOUT,
                allow_redirects=True,
                headers={"User-Agent": self.USER_AGENT},
            )
        except requests.RequestException:
            try:
                response = requests.get(
                    f"http://{requested_domain}/",
                    timeout=DIRECT_FETCH_TIMEOUT,
                    allow_redirects=True,
                    headers={"User-Agent": self.USER_AGENT},
                )
            except requests.RequestException:
                response = None

        canonical_url = root_url
        canonical_domain = requested_domain
        source_name = ""
        homepage_title = ""
        homepage_excerpt = ""
        status_code = None
        accessible = False

        if response is not None:
            status_code = response.status_code
            accessible = response.status_code < 400
            redirect_chain = [item.url for item in response.history] + [response.url]
            for item_url in redirect_chain:
                item_domain = domain_from_url(item_url)
                if item_domain:
                    aliases.add(item_domain)

            if response.url:
                canonical_url = response.url
                final_domain = domain_from_url(response.url)
                if final_domain:
                    canonical_domain = final_domain

            content_type = response.headers.get("Content-Type", "").lower()
            if "html" in content_type or response.text.lstrip().startswith("<"):
                soup = BeautifulSoup(response.text[:500_000], "html.parser")
                title_tag = soup.find("title")
                homepage_title = _clean_source_name(title_tag.get_text(" ", strip=True) if title_tag else "")

                site_name = soup.find("meta", attrs={"property": "og:site_name"})
                if site_name and site_name.get("content"):
                    source_name = _clean_source_name(site_name.get("content", ""))

                canonical_tag = soup.find("link", attrs={"rel": lambda value: value and "canonical" in value})
                if canonical_tag and canonical_tag.get("href"):
                    candidate_url = urljoin(response.url, canonical_tag.get("href"))
                    candidate_domain = domain_from_url(candidate_url)
                    if candidate_domain:
                        canonical_url = candidate_url
                        canonical_domain = candidate_domain
                        aliases.add(candidate_domain)

                text = soup.get_text(" ", strip=True)
                homepage_excerpt = re.sub(r"\s+", " ", text)[:2000]

        if not source_name:
            source_name = homepage_title or _fallback_name(canonical_domain)

        aliases.discard(canonical_domain)
        return SourceIdentity(
            requested_url=requested_url,
            requested_domain=requested_domain,
            canonical_url=canonical_url,
            canonical_domain=canonical_domain,
            source_name=source_name,
            aliases=sorted(alias for alias in aliases if alias),
            redirect_chain=redirect_chain,
            homepage_accessible=accessible,
            homepage_status_code=status_code,
            homepage_title=homepage_title,
            homepage_text_excerpt=homepage_excerpt,
        )
