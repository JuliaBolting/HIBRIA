from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from pipeline.output.explanation_generator import ExplanationGenerator


def sample_result() -> SimpleNamespace:
    first_claim = SimpleNamespace(
        claim_id="claim-1",
        text="Lula lidera a disputa presidencial em três estados.",
    )
    second_claim = SimpleNamespace(
        claim_id="claim-2",
        text=(
            "Lula e Flávio Bolsonaro estão tecnicamente empatados no "
            "levantamento nacional."
        ),
    )
    first_evidence = SimpleNamespace(
        evidence_id="evidence-1",
        stance="support",
        source="fonte-que-nao-deve-ir-ao-prompt.test",
        text="trecho externo que não deve ir ao prompt",
    )
    second_evidence = SimpleNamespace(
        evidence_id="evidence-2",
        stance="contradict",
        source="outra-fonte.test",
        text="outro trecho externo que não deve ir ao prompt",
    )
    retrievals = [
        SimpleNamespace(claim=first_claim, evidences=[first_evidence]),
        SimpleNamespace(claim=second_claim, evidences=[second_evidence]),
    ]
    stances = [
        {
            "claim_id": "claim-1",
            "evidence_id": "evidence-1",
            "stance": "support",
        },
        {
            "claim_id": "claim-2",
            "evidence_id": "evidence-2",
            "stance": "contradict",
        },
    ]
    return SimpleNamespace(
        label_final="evidência insuficiente",
        title="Quaest: disputa presidencial em 7 estados e no DF",
        score_breakdown={
            "coverage_score": 10.0,
            "stance_stats": {
                "support": 8,
                "contradict": 3,
                "neutral": 2,
                "insufficient": 4,
            },
        },
        retrieval_results=retrievals,
        stance_results=stances,
    )


def valid_model_report() -> dict:
    return {
        "explanation": (
            "A liderança em três estados recebeu confirmação, enquanto o "
            "empate técnico apresentou diferenças nas verificações. Poucas "
            "informações importantes puderam ser verificadas."
        ),
        "details": [
            "A liderança de Lula em três estados recebeu confirmação; o "
            "empate técnico apresentou informações diferentes nas verificações.",
            "No conjunto, 8 trechos foram confirmados, 3 apresentaram "
            "diferenças e 6 ficaram sem confirmação direta; somente uma "
            "pequena parte do conteúdo importante pôde ser verificada.",
            "A falta de confirmação do restante reduziu a nota. Isso não "
            "significa, por si só, que a notícia seja falsa.",
        ],
    }


def ollama_response(report: dict) -> MagicMock:
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "message": {
            "content": json.dumps(report, ensure_ascii=False),
        }
    }
    return response


class ExplanationV3Tests(unittest.TestCase):
    def test_context_contains_concrete_results_without_external_sources(self):
        context = ExplanationGenerator._build_context(sample_result())
        serialized = json.dumps(context, ensure_ascii=False)

        self.assertEqual(context["resumo_numerico"]["confirmacoes"], 8)
        self.assertEqual(context["resumo_numerico"]["diferencas"], 3)
        self.assertEqual(context["resumo_numerico"]["sem_confirmacao_direta"], 6)
        self.assertEqual(len(context["verificacoes_relevantes"]), 2)
        self.assertIn("Lula lidera", serialized)
        self.assertIn("recebeu confirmação", serialized)
        self.assertIn("apresentou informações diferentes", serialized)
        self.assertNotIn("fonte-que-nao-deve", serialized)
        self.assertNotIn("trecho externo", serialized)

    def test_prompt_is_small_and_uses_validated_sheet(self):
        prompt = ExplanationGenerator._build_prompt(sample_result())

        self.assertIn("FICHA VALIDADA", prompt)
        self.assertIn("verificacoes_relevantes", prompt)
        self.assertIn("resumo_numerico", prompt)
        self.assertLessEqual(len(prompt), ExplanationGenerator.MAX_INPUT_CHARS)

    def test_valid_specific_report_is_accepted_as_qwen(self):
        with patch(
            "pipeline.output.explanation_generator.requests.post",
            return_value=ollama_response(valid_model_report()),
        ):
            report = ExplanationGenerator.generate(sample_result())

        self.assertEqual(report["source"], "qwen")
        self.assertEqual(report["explanation"], valid_model_report()["explanation"])
        self.assertEqual(report["details"], valid_model_report()["details"])

    def test_repeated_classification_rejects_only_explanation(self):
        model_report = valid_model_report()
        model_report["explanation"] = (
            "O resultado foi evidência insuficiente. Como poucas informações "
            "foram verificadas, o resultado foi evidência insuficiente."
        )
        context = ExplanationGenerator._build_context(sample_result())
        fallback = ExplanationGenerator._fallback_report_from_context(
            sample_result(),
            context,
        )

        checked = ExplanationGenerator._validate_model_report(
            model_report,
            result=sample_result(),
            context=context,
            fallback=fallback,
        )

        self.assertEqual(checked["source"], "hybrid")
        self.assertEqual(checked["explanation"], fallback["explanation"])
        self.assertEqual(checked["details"], model_report["details"])

    def test_invented_number_rejects_details(self):
        model_report = valid_model_report()
        model_report["details"][1] = (
            "No conjunto, 9 trechos foram confirmados, 3 apresentaram "
            "diferenças e 6 ficaram sem confirmação direta."
        )
        context = ExplanationGenerator._build_context(sample_result())
        fallback = ExplanationGenerator._fallback_report_from_context(
            sample_result(),
            context,
        )

        checked = ExplanationGenerator._validate_model_report(
            model_report,
            result=sample_result(),
            context=context,
            fallback=fallback,
        )

        self.assertEqual(checked["source"], "hybrid")
        self.assertEqual(checked["details"], fallback["details"])

    def test_conflicting_label_is_rejected(self):
        model_report = valid_model_report()
        model_report["explanation"] = (
            "A liderança em três estados recebeu confirmação e, por isso, a "
            "notícia foi considerada confiável."
        )
        context = ExplanationGenerator._build_context(sample_result())
        fallback = ExplanationGenerator._fallback_report_from_context(
            sample_result(),
            context,
        )

        checked = ExplanationGenerator._validate_model_report(
            model_report,
            result=sample_result(),
            context=context,
            fallback=fallback,
        )

        self.assertEqual(checked["source"], "hybrid")
        self.assertEqual(checked["explanation"], fallback["explanation"])


if __name__ == "__main__":
    unittest.main()
