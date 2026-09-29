"""Política única usada no ranking, no cálculo e na explicação."""
import math
import os


def value(item, key, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def eligible(evidence, score=None):
    if evidence is None:
        return False
    metadata = value(evidence, "metadata", {}) or {}
    layer = value(evidence, "evidence_layer", value(evidence, "retrieval_layer", ""))
    kind = value(evidence, "source_type", "")
    source = str(value(evidence, "source", value(evidence, "evidence_source", ""))).lower()
    url = str(value(evidence, "url", value(evidence, "evidence_url", ""))).lower()
    if (layer == "wikipedia" or kind in {"encyclopedia", "labeled_dataset"}
            or source.startswith("fake.br") or url.startswith("fakebr://")
            or metadata.get("corpus") == "fake.br"):
        return False
    if score is None:
        score = value(evidence, "similarity_final", metadata.get("stance_similarity", value(evidence, "similarity", 0)))
    try:
        threshold = float(os.getenv("HIBRIA_AGGREGATOR_MIN_VALID_EVIDENCE_SCORE") or os.getenv("HIBRIA_MIN_VALID_EVIDENCE_SCORE", "0.55"))
        if not math.isfinite(float(score)) or float(score) < threshold:
            return False
    except (TypeError, ValueError):
        return False
    if not value(evidence, "is_sufficient", True):
        return False
    if layer == "factcheck" or kind == "fact_check":
        return True
    if layer == "vector_store":
        return metadata.get("approved_for_rag") is True or value(evidence, "trusted_source", False) is True
    return value(evidence, "trusted_source", False) is True
