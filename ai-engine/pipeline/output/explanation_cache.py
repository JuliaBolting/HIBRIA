"""Atualiza somente a redação de uma análise persistida, sem pesquisar."""
import hashlib
import json
import os
from types import SimpleNamespace

from .explanation_generator import ExplanationGenerator


def fingerprint():
    settings = {name: os.getenv(name, "") for name in (
        "HIBRIA_QWEN_MODEL", "HIBRIA_QWEN_NUM_CTX", "HIBRIA_QWEN_MAX_INPUT_CHARS",
        "HIBRIA_QWEN_MAX_OUTPUT_TOKENS", "HIBRIA_QWEN_TIMEOUT_SECONDS")}
    settings["version"] = ExplanationGenerator.VERSION
    settings["prompt"] = ExplanationGenerator._system_prompt()
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def refresh(data):
    analysis = data.get("analysis") or {}
    retrieval = [SimpleNamespace(
        claim=SimpleNamespace(claim_id=claim.get("claim_id", ""), text=claim.get("text", "")),
        evidences=[SimpleNamespace(**evidence) for evidence in claim.get("evidences", [])])
        for claim in (data.get("evidence") or {}).get("claims", [])]
    result = SimpleNamespace(title=analysis.get("title", ""), score_final=analysis.get("score"),
        label_final=analysis.get("label"), score_breakdown=analysis.get("score_breakdown") or {},
        retrieval_results=retrieval,
        stance_results=(data.get("transparency") or {}).get("stance_results") or [])
    report = ExplanationGenerator.generate(result)
    data["explanation"], data["details"] = report["explanation"], report["details"]
    metadata = data.setdefault("metadata", {})
    metadata.update(explanation_version=ExplanationGenerator.VERSION,
                    explanation_source=report.get("source"),
                    explanation_validation=report.get("validation", {}),
                    explanation_fingerprint=fingerprint())
    return data
