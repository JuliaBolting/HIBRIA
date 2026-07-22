from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .criteria import weights_output


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SourceIdentity:
    requested_url: str
    requested_domain: str
    canonical_url: str
    canonical_domain: str
    source_name: str
    aliases: list[str] = field(default_factory=list)
    redirect_chain: list[str] = field(default_factory=list)
    homepage_accessible: bool = False
    homepage_status_code: int | None = None
    homepage_title: str = ""
    homepage_text_excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_url": self.requested_url,
            "requested_domain": self.requested_domain,
            "canonical_url": self.canonical_url,
            "canonical_domain": self.canonical_domain,
            "source_name": self.source_name,
            "aliases": self.aliases,
            "redirect_chain": self.redirect_chain,
            "homepage_accessible": self.homepage_accessible,
            "homepage_status_code": self.homepage_status_code,
            "homepage_title": self.homepage_title,
            "homepage_text_excerpt": self.homepage_text_excerpt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceIdentity":
        return cls(
            requested_url=data.get("requested_url", ""),
            requested_domain=data.get("requested_domain", ""),
            canonical_url=data.get("canonical_url", ""),
            canonical_domain=data.get("canonical_domain", ""),
            source_name=data.get("source_name", ""),
            aliases=list(data.get("aliases", [])),
            redirect_chain=list(data.get("redirect_chain", [])),
            homepage_accessible=bool(data.get("homepage_accessible", False)),
            homepage_status_code=data.get("homepage_status_code"),
            homepage_title=data.get("homepage_title", ""),
            homepage_text_excerpt=data.get("homepage_text_excerpt", ""),
        )


@dataclass
class ReputationEvidence:
    criterion: str
    provider: str
    title: str
    url: str
    snippet: str = ""
    evidence_type: str = "search_result"
    is_same_domain: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "provider": self.provider,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "evidence_type": self.evidence_type,
            "is_same_domain": self.is_same_domain,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReputationEvidence":
        return cls(
            criterion=data.get("criterion", ""),
            provider=data.get("provider", ""),
            title=data.get("title", ""),
            url=data.get("url", ""),
            snippet=data.get("snippet", ""),
            evidence_type=data.get("evidence_type", "search_result"),
            is_same_domain=bool(data.get("is_same_domain", False)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class CriterionEvaluation:
    key: str
    label: str
    max_points: int
    points: int | None
    status: str
    justification: str
    evidences: list[ReputationEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "max_points": self.max_points,
            "points": self.points,
            "status": self.status,
            "justification": self.justification,
            "evidence_urls": list(dict.fromkeys(e.url for e in self.evidences if e.url)),
            "evidences": [e.to_dict() for e in self.evidences],
        }

    @classmethod
    def from_dict(cls, key: str, data: dict[str, Any]) -> "CriterionEvaluation":
        return cls(
            key=key,
            label=data.get("label", key),
            max_points=int(data.get("max_points", 0)),
            points=data.get("points"),
            status=data.get("status", "unknown"),
            justification=data.get("justification", ""),
            evidences=[ReputationEvidence.from_dict(item) for item in data.get("evidences", [])],
        )


@dataclass
class SourceReputation:
    identity: SourceIdentity
    status: str
    note: int | None
    score: float
    classification: str
    criteria: dict[str, CriterionEvaluation]
    origin: str
    method: str
    needs_review: bool
    requires_external_evidence_weight: bool
    reason: str
    providers_attempted: list[str] = field(default_factory=list)
    providers_succeeded: list[str] = field(default_factory=list)
    provider_failures: dict[str, str] = field(default_factory=dict)
    query_count: int = 0
    evidence_count: int = 0
    evaluated_at: str = field(default_factory=utc_now_iso)
    expires_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "method": self.method,
            "origin": self.origin,
            "domain": self.identity.canonical_domain,
            "normalized_domain": self.identity.canonical_domain,
            "requested_domain": self.identity.requested_domain,
            "canonical_domain": self.identity.canonical_domain,
            "canonical_url": self.identity.canonical_url,
            "source_name": self.identity.source_name,
            "aliases": self.identity.aliases,
            "identity": self.identity.to_dict(),
            "is_registered_source": self.status == "evaluated",
            "is_trusted_domain": self.status == "evaluated" and self.note is not None and self.note >= 80,
            "requires_external_evidence_weight": self.requires_external_evidence_weight,
            "note": self.note,
            "note_scale": "0-100",
            "score": round(self.score, 4),
            "score_scale": "0.0-1.0",
            "classification": self.classification,
            "criteria_weights": weights_output(),
            "criteria_scores": {key: value.to_dict() for key, value in self.criteria.items()},
            "needs_review": self.needs_review,
            "providers_attempted": self.providers_attempted,
            "providers_succeeded": self.providers_succeeded,
            "provider_failures": self.provider_failures,
            "query_count": self.query_count,
            "evidence_count": self.evidence_count,
            "evaluated_at": self.evaluated_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceReputation":
        identity_data = data.get("identity") or {
            "requested_url": data.get("canonical_url", ""),
            "requested_domain": data.get("requested_domain", data.get("domain", "")),
            "canonical_url": data.get("canonical_url", ""),
            "canonical_domain": data.get("canonical_domain", data.get("domain", "")),
            "source_name": data.get("source_name", ""),
            "aliases": data.get("aliases", []),
        }
        return cls(
            identity=SourceIdentity.from_dict(identity_data),
            status=data.get("status", "unknown"),
            note=data.get("note"),
            score=float(data.get("score", 0.5)),
            classification=data.get("classification", "avaliação insuficiente"),
            criteria={
                key: CriterionEvaluation.from_dict(key, value)
                for key, value in (data.get("criteria_scores") or {}).items()
            },
            origin=data.get("origin", "stored"),
            method=data.get("method", ""),
            needs_review=bool(data.get("needs_review", False)),
            requires_external_evidence_weight=bool(data.get("requires_external_evidence_weight", True)),
            reason=data.get("reason", ""),
            providers_attempted=list(data.get("providers_attempted", [])),
            providers_succeeded=list(data.get("providers_succeeded", [])),
            provider_failures=dict(data.get("provider_failures", {})),
            query_count=int(data.get("query_count", 0)),
            evidence_count=int(data.get("evidence_count", 0)),
            evaluated_at=data.get("evaluated_at", utc_now_iso()),
            expires_at=data.get("expires_at"),
            metadata=dict(data.get("metadata", {})),
        )
