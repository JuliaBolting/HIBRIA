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

    def test_explanation_prompt_uses_safe_verification_summary(self):
        claim = self.claim("Cauã Reymond foi anunciado para a segunda temporada.")
        evidence = self.evidence(
            "Cauã Reymond foi confirmado na segunda temporada da série."
        )
        evidence.title = "Cauã Reymond é confirmado em série"
        evidence.stance = "support"
        retrieval = SimpleNamespace(claim=claim, evidences=[evidence])
        stance = SimpleNamespace(
            to_dict=lambda: {
                "claim_id": "claim-1",
                "evidence_id": "evidence-1",
                "stance": "support",
                "confidence": 0.88,
                "source": "fonte.test",
                "url": "https://fonte.test/noticia",
                "reason": "diferença relevante de polaridade textual",
            }
        )
        result = SimpleNamespace(
            label_final="evidência insuficiente",
            score_breakdown={
                "coverage_score": 20.0,
                "evidence_score": 80.0,
                "bertimbau_score": 75.0,
                "stance_stats": {
                    "support": 1,
                    "contradict": 0,
                    "neutral": 0,
                },
            },
            reputation={},
            claims=[claim],
            stance_results=[stance],
            retrieval_results=[retrieval],
            title="Notícia de teste",
        )

        prompt = ExplanationGenerator._build_prompt(result)

        self.assertIn("verificacoes_relevantes", prompt)
        self.assertNotIn("fonte.test", prompt)
        self.assertIn("informacao_da_noticia", prompt)
        self.assertIn("resultado_calculado", prompt)
        self.assertIn("resumo_numerico", prompt)
        self.assertIn("recebeu confirmação nas verificações", prompt)
        self.assertNotIn("Cauã Reymond foi confirmado na segunda temporada", prompt)
        self.assertNotIn("diferença relevante de polaridade textual", prompt)

    def test_system_prompt_explains_result_without_numbered_claims(self):
        prompt = ExplanationGenerator._system_prompt()

        self.assertIn("pessoa leiga", prompt)
        self.assertIn("Não cite sites ou fontes", prompt)
        self.assertIn('"claim"', prompt)

    def test_prompt_excerpt_does_not_end_in_middle_of_word(self):
        excerpt = (
            "Naomi Watts e Sarah Paulson discutem a importância da "
            "independência financeira e da união entre as mulheres como "
            "pilares da trama e da vida real."
        )

        shortened = ExplanationGenerator._truncate_text(excerpt * 4, 130)

        self.assertTrue(shortened.endswith("…"))
        self.assertNotIn("como pi…", shortened)
        self.assertFalse(shortened.endswith("p…"))

    def test_report_parser_does_not_cut_detail_in_middle_of_word(self):
        long_detail = (
            "A fonte confirma a escalação do ator para a segunda temporada. "
            + "Contexto complementar " * 20
        )
        raw_report = {
            "explanation": "A cobertura foi parcial, mas há apoio para a afirmação principal.",
            "details": [
                long_detail,
                "A segunda fonte confirma a data de estreia.",
                "A terceira fonte contextualiza a história da série.",
            ],
        }

        parsed = ExplanationGenerator._parse_report(
            json.dumps(raw_report, ensure_ascii=False)
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(
            parsed["details"][0],
            "A fonte confirma a escalação do ator para a segunda temporada.",
        )

    def test_report_parser_removes_detail_numbers_and_fixes_hibria_name(self):
        raw_report = {
            "explanation": "A HÍBRA encontrou confirmação para a informação principal.",
            "details": [
                "1. A informação principal recebeu apoio externo.",
                "2) A cobertura das demais informações foi baixa.",
                "3 - A HIBRA considerou os fatores complementares.",
            ],
        }

        parsed = ExplanationGenerator._parse_report(
            json.dumps(raw_report, ensure_ascii=False)
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(
            parsed["explanation"],
            "A análise encontrou confirmação para a informação principal.",
        )
        self.assertEqual(
            parsed["details"],
            [
                "A informação principal recebeu apoio externo.",
                "A cobertura das demais informações foi baixa.",
                "A análise considerou os fatores complementares.",
            ],
        )

    def test_report_parser_replaces_repeated_details(self):
        repeated = "A informação principal recebeu apoio externo."
        raw_report = {
            "explanation": "A informação principal recebeu confirmação externa.",
            "details": [repeated, repeated, repeated],
        }
        fallback_details = [
            "A escalação do ator recebeu apoio externo.",
            "As demais informações tiveram cobertura baixa.",
            "A baixa cobertura influenciou a classificação final.",
        ]

        parsed = ExplanationGenerator._parse_report(
            json.dumps(raw_report, ensure_ascii=False),
            fallback_details=fallback_details,
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["details"], fallback_details)

    def test_report_parser_keeps_exactly_three_when_fallback_is_available(self):
        generated_details = [
            "As comparações encontraram apoio externo.",
            "A cobertura das informações foi baixa.",
            "A baixa cobertura influenciou o resultado.",
        ]
        raw_report = {
            "explanation": "O resultado reflete apoio com cobertura baixa.",
            "details": generated_details,
        }

        parsed = ExplanationGenerator._parse_report(
            json.dumps(raw_report, ensure_ascii=False),
            fallback_details=[
                "Tópico de contingência um.",
                "Tópico de contingência dois.",
                "Tópico de contingência três.",
            ],
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["details"], generated_details)

    def test_deterministic_details_explain_only_classification_factors(self):
        result = SimpleNamespace(
            label_final="evidência insuficiente",
            score_breakdown={
                "coverage_score": 20.0,
                "stance_stats": {
                    "support": 2,
                    "contradict": 0,
                    "neutral": 1,
                },
            },
        )

        details = ExplanationGenerator._deterministic_details(result)

        self.assertEqual(len(details), 3)
        self.assertIn("2 trechos receberam confirmação", details[0])
        self.assertIn("1 ficou sem confirmação direta", details[0])
        self.assertIn("Somente uma pequena parte", details[1])
        self.assertIn("reduziu a nota", details[2])
        self.assertIn("não significa", details[2])
        self.assertTrue(all("HÍBRIA" not in item for item in details))
        self.assertTrue(all("cobertura" not in item.casefold() for item in details))
        self.assertTrue(all("comparações" not in item.casefold() for item in details))

    def test_plain_explanation_rejects_false_opposition_and_jargon(self):
        self.assertFalse(
            ExplanationGenerator._is_plain_explanation(
                "A informação foi encontrada, mas há apoio externo para ela."
            )
        )
        self.assertFalse(
            ExplanationGenerator._is_plain_explanation(
                "A cobertura das comparações foi baixa."
            )
        )
        self.assertFalse(
            ExplanationGenerator._is_plain_explanation(
                "A análise encontrou que a informação está correta."
            )
        )
        self.assertTrue(
            ExplanationGenerator._is_plain_explanation(
                "Parte da notícia foi confirmada, mas outras informações não puderam ser verificadas."
            )
        )

    def test_deterministic_explanation_is_clear_for_lay_reader(self):
        result = SimpleNamespace(
            label_final="evidência insuficiente",
            score_breakdown={
                "coverage_score": 20.0,
                "stance_stats": {
                    "support": 2,
                    "contradict": 1,
                    "neutral": 0,
                },
            },
        )

        explanation = ExplanationGenerator._deterministic_explanation(result)

        self.assertIn("Algumas informações foram confirmadas", explanation)
        self.assertIn("poucas partes da notícia", explanation)
        self.assertIn("não foi suficiente para sustentar", explanation)
        self.assertNotIn("evidência insuficiente", explanation)
        self.assertNotIn("cobertura", explanation.casefold())

    def test_generate_preserves_valid_qwen_details(self):
        model_report = {
            "explanation": (
                "As verificações da pesquisa trouxeram confirmações, diferenças "
                "e pontos sem confirmação direta. Apenas uma pequena parte do "
                "conteúdo importante pôde ser verificada."
            ),
            "details": [
                "Nas verificações, 8 resultados receberam confirmação, 3 apresentaram diferenças e 2 ficaram sem confirmação direta.",
                "Somente uma pequena parte dos dados importantes pôde ser verificada, e esse alcance limitado determinou a decisão.",
                "A falta de verificação do restante reduziu a nota. Isso não significa, por si só, que a notícia seja falsa.",
            ],
        }
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "message": {
                "content": json.dumps(model_report, ensure_ascii=False),
            }
        }
        result = SimpleNamespace(
            label_final="evidência insuficiente",
            title="Quaest: disputa presidencial nos estados",
            score_breakdown={
                "coverage_score": 10.0,
                "evidence_score": 56.0,
                "bertimbau_score": 90.0,
                "stance_stats": {
                    "support": 8,
                    "contradict": 3,
                    "neutral": 2,
                },
            },
            reputation={},
            claims=[],
            stance_results=[],
            retrieval_results=[],
        )

        with patch(
            "pipeline.output.explanation_generator.requests.post",
            return_value=response,
        ):
            report = ExplanationGenerator.generate(result)

        self.assertIsNotNone(report)
        self.assertEqual(report["details"], model_report["details"])
        self.assertEqual(report["source"], "qwen")


if __name__ == "__main__":
    unittest.main()
