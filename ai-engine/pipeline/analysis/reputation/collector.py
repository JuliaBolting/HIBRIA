from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from pipeline.retrieval.search_gateway import SearchGateway

from .config import (
    DIRECT_FETCH_TIMEOUT,
    MAX_PROVIDERS_PER_QUERY,
    MAX_RESULTS_PER_QUERY,
    SEARCH_SLEEP_SECONDS,
)
from .criteria import CRITERIA, ReputationCriterion
from .identity import normalize_domain
from .models import ReputationEvidence, SourceIdentity


@dataclass
class CollectionResult:
    evidence_by_criterion: dict[str, list[ReputationEvidence]] = field(default_factory=dict)
    providers_attempted: list[str] = field(default_factory=list)
    providers_succeeded: list[str] = field(default_factory=list)
    provider_failures: dict[str, str] = field(default_factory=dict)
    query_count: int = 0


class ReputationEvidenceCollector:
    """Coleta evidências públicas para os critérios do TCC."""

    USER_AGENT = "HibriaSourceReputation/1.0"

    def __init__(self, gateway: SearchGateway | None = None) -> None:
        self.gateway = gateway or SearchGateway()

    def collect(self, identity: SourceIdentity) -> CollectionResult:
        output = CollectionResult()

        for criterion in CRITERIA:
            evidence: list[ReputationEvidence] = []
            evidence.extend(self._direct_probe(identity, criterion))

            for template in criterion.query_templates[:1]:
                query = template.format(
                    domain=identity.canonical_domain,
                    source_name=identity.source_name or identity.canonical_domain,
                )
                batch = self.gateway.search(
                    query,
                    max_results=MAX_RESULTS_PER_QUERY,
                    min_results=2,
                    max_providers=MAX_PROVIDERS_PER_QUERY,
                )
                output.query_count += 1
                output.providers_attempted.extend(batch.providers_attempted)
                output.providers_succeeded.extend(batch.providers_succeeded)
                output.provider_failures.update(batch.providers_failed)
                evidence.extend(self._convert_hits(identity, criterion.key, batch.hits))
                if SEARCH_SLEEP_SECONDS > 0:
                    time.sleep(SEARCH_SLEEP_SECONDS)

            if criterion.use_factcheck:
                query = f'"{identity.source_name or identity.canonical_domain}" {identity.canonical_domain}'
                batch = self.gateway.search_factchecks(query, max_results=MAX_RESULTS_PER_QUERY)
                output.query_count += 1 if batch.providers_attempted else 0
                output.providers_attempted.extend(batch.providers_attempted)
                output.providers_succeeded.extend(batch.providers_succeeded)
                output.provider_failures.update(batch.providers_failed)
                evidence.extend(self._convert_hits(identity, criterion.key, batch.hits))

            output.evidence_by_criterion[criterion.key] = self._deduplicate(evidence)

        output.providers_attempted = list(dict.fromkeys(output.providers_attempted))
        output.providers_succeeded = list(dict.fromkeys(output.providers_succeeded))
        return output

    def _direct_probe(
        self,
        identity: SourceIdentity,
        criterion: ReputationCriterion,
    ) -> list[ReputationEvidence]:
        evidence: list[ReputationEvidence] = []
        base_url = f"https://{identity.canonical_domain}/"

        for path in criterion.direct_paths:
            url = urljoin(base_url, path)
            try:
                response = requests.get(
                    url,
                    timeout=DIRECT_FETCH_TIMEOUT,
                    allow_redirects=True,
                    headers={"User-Agent": self.USER_AGENT},
                )
            except requests.RequestException:
                continue

            if response.status_code >= 400:
                continue

            content_type = response.headers.get("Content-Type", "").lower()
            if "html" not in content_type and not response.text.lstrip().startswith("<"):
                continue

            soup = BeautifulSoup(response.text[:500_000], "html.parser")
            title_tag = soup.find("title")
            title = title_tag.get_text(" ", strip=True) if title_tag else url
            text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
            if len(text) < 80:
                continue

            if criterion.key != "adequacao_tecnica_sistema":
                normalized = text.lower()
                if not any(term.lower() in normalized for term in criterion.positive_terms):
                    continue

            evidence.append(ReputationEvidence(
                criterion=criterion.key,
                provider="direct_site",
                title=title[:300],
                url=response.url,
                snippet=text[:1800],
                evidence_type="direct_page",
                is_same_domain=self._same_domain(identity.canonical_domain, response.url),
                metadata={"status_code": response.status_code, "requested_path": path},
            ))
            # Uma página direta forte por critério é suficiente e reduz o tempo
            # da semeadura individual.
            break

        if criterion.key == "adequacao_tecnica_sistema" and identity.homepage_accessible:
            evidence.append(ReputationEvidence(
                criterion=criterion.key,
                provider="identity_resolver",
                title=identity.homepage_title or identity.source_name,
                url=identity.canonical_url,
                snippet=identity.homepage_text_excerpt,
                evidence_type="technical_probe",
                is_same_domain=True,
                metadata={"status_code": identity.homepage_status_code},
            ))
        return evidence

    def _convert_hits(self, identity: SourceIdentity, criterion: str, hits) -> list[ReputationEvidence]:
        return [
            ReputationEvidence(
                criterion=criterion,
                provider=hit.provider,
                title=hit.title,
                url=hit.url,
                snippet=hit.snippet,
                evidence_type="fact_check" if hit.provider == "google_factcheck" else "search_result",
                is_same_domain=self._same_domain(identity.canonical_domain, hit.url),
                metadata=hit.metadata,
            )
            for hit in hits
        ]

    @staticmethod
    def _same_domain(domain: str, url: str) -> bool:
        target = normalize_domain(urlparse(url).netloc)
        source = normalize_domain(domain)
        return bool(target and source and (target == source or target.endswith(f".{source}")))

    @staticmethod
    def _deduplicate(items: list[ReputationEvidence]) -> list[ReputationEvidence]:
        seen: set[str] = set()
        output: list[ReputationEvidence] = []
        for item in items:
            key = item.url.strip().lower().rstrip("/")
            if not key or key in seen:
                continue
            seen.add(key)
            output.append(item)
        return output
