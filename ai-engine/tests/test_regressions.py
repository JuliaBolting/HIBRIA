from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from pipeline.analysis.stance_model import StanceModel
from pipeline.output.aggregator import Aggregator
from pipeline.output.explanation_generator import ExplanationGenerator
from pipeline.retrieval.retriever import (
    EvidenceRetriever,
    FactCheckSource,
    NewsApiSource,
)
from pipeline.retrieval.search_providers.quota import ProviderQuota


def similarity_item(*, layer: str, source_type: str, trusted: bool, metadata=None):
    top = SimpleNamespace(
        evidence_layer=layer,
        source_type=source_type,
        trusted_source=trusted,
        is_sufficient=True,
        metadata=metadata or {},
    )
    return SimpleNamespace(
        has_evidence=True,
        has_sufficient_evidence=True,
        score=0.90,
        top_evidence=top,
    )


class RetrievalRegressionTests(unittest.TestCase):
    def test_newsapi_is_wired_into_retriever(self):
        retriever = EvidenceRetriever(vector_store=None)
        self.assertTrue(
            any(isinstance(source, NewsApiSource) for source in retriever._sources)
        )

    def test_factcheck_respects_feature_flag(self):
        with patch.dict(
            os.environ,
            {
                "GOOGLE_FACTCHECK_API_KEY": "test-key",
                "HIBRIA_ENABLE_FACTCHECK": "false",
            },
            clear=False,
        ):
            self.assertFalse(FactCheckSource().is_available())

    def test_untrusted_web_result_does_not_affect_score(self):
        item = similarity_item(
            layer="web_search",
            source_type="web",
            trusted=False,
        )
        self.assertFalse(Aggregator._is_valid_factual_evidence(item))

    def test_factcheck_is_valid_without_domain_reputation(self):
        item = similarity_item(
            layer="factcheck",
            source_type="fact_check",
            trusted=False,
        )
        self.assertTrue(Aggregator._is_valid_factual_evidence(item))

    def test_faiss_requires_approved_memory_item(self):
        unapproved = similarity_item(
            layer="vector_store",
            source_type="external_document",
            trusted=False,
        )
        approved = similarity_item(
            layer="vector_store",
            source_type="external_document",
            trusted=False,
            metadata={"approved_for_rag": True},
        )
        self.assertFalse(Aggregator._is_valid_factual_evidence(unapproved))
        self.assertTrue(Aggregator._is_valid_factual_evidence(approved))

    def test_provider_quota_enforces_each_providers_own_limit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ProviderQuota, "_path", Path(tmp) / "quotas.json"
        ), patch.dict(
            os.environ,
            {
                "HIBRIA_SERPER_DAILY_LIMIT": "0",
                "HIBRIA_SERPER_MONTHLY_LIMIT": "0",
                "HIBRIA_SERPER_TOTAL_LIMIT": "1",
                "HIBRIA_SEARCHAPI_DAILY_LIMIT": "0",
                "HIBRIA_SEARCHAPI_MONTHLY_LIMIT": "0",
                "HIBRIA_SEARCHAPI_TOTAL_LIMIT": "2",
            },
            clear=False,
        ):
            self.assertTrue(ProviderQuota.try_register("serper"))
            self.assertFalse(ProviderQuota.try_register("serper"))
            self.assertTrue(ProviderQuota.try_register("searchapi"))


class StanceRegressionTests(unittest.TestCase):
    @staticmethod
    def claim(text: str):
        return SimpleNamespace(claim_id="claim-1", text=text, normalized=text)

    @staticmethod
    def evidence(text: str):
        return SimpleNamespace(
            evidence_id="evidence-1",
            text=text,
            source="fonte.test",
            url="https://fonte.test/noticia",
            similarity=0.90,
            retrieval_layer="web_search",
            source_type="web",
            trusted_source=True,
        )

    def test_unrelated_negation_does_not_turn_confirmation_into_contradiction(self):
        claim = self.claim(
            "Cauã Reymond foi anunciado como parte do elenco da segunda temporada."
        )
        evidence = self.evidence(
            "Cauã Reymond foi confirmado como parte do elenco da segunda temporada. "
            "O papel do ator ainda não foi divulgado."
        )

        result = StanceModel.analyze_pair(claim, evidence)

        self.assertEqual(result.stance, "support")

    def test_negation_in_matching_sentence_is_contradiction(self):
        claim = self.claim(
            "Cauã Reymond foi anunciado como parte do elenco da segunda temporada."
        )
        evidence = self.evidence(
            "Cauã Reymond não foi anunciado como parte do elenco da segunda temporada."
        )

        result = StanceModel.analyze_pair(claim, evidence)

        self.assertEqual(result.stance, "contradict")



if __name__ == "__main__":
    unittest.main()
