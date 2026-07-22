from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from .config import ENABLE_GEMINI_REVIEW
from .criteria import CRITERIA
from .models import ReputationEvidence, SourceIdentity


class GeminiEvidenceInterpreter:
    """Organiza evidências por critério; nunca calcula a nota final."""

    def __init__(self) -> None:
        self.enabled = ENABLE_GEMINI_REVIEW
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("HIBRIA_AI_FALLBACK_MODEL", "gemini-2.5-flash").strip()

    def is_available(self) -> bool:
        return bool(self.enabled and self.api_key)

    def interpret(
        self,
        identity: SourceIdentity,
        evidence_by_criterion: dict[str, list[ReputationEvidence]],
    ) -> dict[str, dict[str, Any]]:
        if not self.is_available():
            return {}

        compact = {
            key: [
                {
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.snippet[:600],
                    "provider": item.provider,
                }
                for item in items[:6]
            ]
            for key, items in evidence_by_criterion.items()
        }
        prompt = {
            "instruction": (
                "Leia somente as evidências fornecidas. Para cada critério, indique "
                "found, partial, negative ou inconclusive. Não invente fatos, não faça "
                "busca própria e não calcule a nota total. Retorne JSON puro."
            ),
            "source": {
                "name": identity.source_name,
                "domain": identity.canonical_domain,
            },
            "criteria": [
                {"key": item.key, "label": item.label, "max_points": item.weight}
                for item in CRITERIA
            ],
            "evidence": compact,
        }

        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                params={"key": self.api_key},
                json={"contents": [{"parts": [{"text": json.dumps(prompt, ensure_ascii=False)}]}]},
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                return {}
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
