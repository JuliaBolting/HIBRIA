from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipeline.persistence.rag_memory_repository import RagMemoryRepository


def make_evidence(**updates):
    values = {
        "retrieval_layer": "factcheck",
        "source_type": "fact_check",
        "source": "Agência de checagem",
        "url": "https://example.org/checagem/1",
        "text": "Evidência verificável com contexto suficiente. " * 4,
        "stance": "support",
        "trusted_source": False,
        "metadata": {"stance_similarity": 0.81},
    }
    values.update(updates)
    return SimpleNamespace(**values)


class RagMemoryPolicyTests(unittest.TestCase):
    def test_factcheck_with_valid_stance_is_approved(self):
        approved, _ = RagMemoryRepository._approval(make_evidence(), None)
        self.assertTrue(approved)

    def test_ai_fallback_is_never_auto_indexed(self):
        approved, _ = RagMemoryRepository._approval(
            make_evidence(retrieval_layer="ai_fallback"),
            1.0,
        )
        self.assertFalse(approved)

    def test_fakebr_is_never_used_as_factual_evidence(self):
        approved, _ = RagMemoryRepository._approval(
            make_evidence(source="Fake.Br/true"),
            1.0,
        )
        self.assertFalse(approved)

    def test_web_source_requires_reputation_by_default(self):
        evidence = make_evidence(
            retrieval_layer="web_search",
            source_type="web",
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HIBRIA_RAG_MEMORY_REQUIRE_SOURCE_REPUTATION", None)
            self.assertFalse(RagMemoryRepository._approval(evidence, None)[0])
            self.assertTrue(RagMemoryRepository._approval(evidence, 0.90)[0])

    def test_tracking_parameters_do_not_change_evidence_key(self):
        first = RagMemoryRepository._evidence_key(
            "https://example.org/a?utm_source=test",
            "texto   igual",
        )
        second = RagMemoryRepository._evidence_key(
            "https://example.org/a",
            "texto igual",
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
