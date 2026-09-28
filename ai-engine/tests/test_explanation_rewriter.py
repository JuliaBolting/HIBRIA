from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from pipeline.output.explanation_generator import ExplanationGenerator


def sample_result() -> SimpleNamespace:
    return SimpleNamespace(
        label_final="evidência insuficiente",
        title="Pesquisa em 7 estados",
        score_breakdown={
            "coverage_score": 10.0,
            "stance_stats": {
                "support": 8,
                "contradict": 3,
                "neutral": 2,
            },
        },
        reputation={"source_name": "fonte que não deve ir ao prompt"},
        claims=[SimpleNamespace(text="fato que não deve ir ao prompt")],
        stance_results=[],
        retrieval_results=[],
    )


def ollama_response(report: dict) -> MagicMock:
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "message": {
            "content": json.dumps(report, ensure_ascii=False),
        }
    }
    return response


class ExplanationRewriterTests(unittest.TestCase):
    def test_prompt_exposes_only_validated_base(self):
        prompt = ExplanationGenerator._build_prompt(sample_result())

        self.assertIn("texto_base_validado", prompt)
        self.assertIn("classificacao_final_obrigatoria", prompt)
        self.assertIn("evidência insuficiente", prompt)
        self.assertIn("8 receberam confirmação", prompt)
        self.assertIn("Não copie nenhuma frase literalmente", prompt)
        self.assertNotIn("fonte que não deve ir ao prompt", prompt)
        self.assertNotIn("fato que não deve ir ao prompt", prompt)

    def test_schema_has_room_to_finish_before_display_limit(self):
        properties = ExplanationGenerator.OUTPUT_SCHEMA["properties"]
        self.assertEqual(properties["explanation"]["maxLength"], 600)
        self.assertEqual(properties["details"]["items"]["maxLength"], 420)

    def test_valid_rewrite_keeps_qwen_source(self):
        result = sample_result()
        base = ExplanationGenerator.fallback_report(result)
        model_report = {
            "explanation": base["explanation"].replace(
                "poucas partes da notícia puderam ser verificadas",
                "somente poucas partes da notícia puderam ser verificadas",
            ),
            "details": [
                "Entre os trechos verificados, 8 tiveram confirmação e 3 "
                "mostraram informações diferentes das referências consultadas.",
                "Mesmo após as verificações, somente uma pequena parte das "
                "informações importantes pôde ser confirmada.",
                "A falta de confirmação do restante reduziu a nota e levou ao "
                'resultado "evidência insuficiente"; isso não significa que a '
                "notícia seja falsa.",
            ],
        }

        with patch(
            "pipeline.output.explanation_generator.requests.post",
            return_value=ollama_response(model_report),
        ):
            report = ExplanationGenerator.generate(result)

        self.assertEqual(report["source"], "qwen")
        self.assertEqual(report["explanation"], model_report["explanation"])
        self.assertEqual(report["details"], model_report["details"])

    def test_changed_label_falls_back_to_deterministic(self):
        result = sample_result()
        model_report = {
            "explanation": (
                "A notícia recebeu uma classificação confiável, embora poucas "
                "partes tenham sido verificadas."
            ),
            "details": [
                "O levantamento nacional confirmou a disputa presidencial.",
                "Houve apoio externo para as informações da consultoria.",
                "O resultado foi confiável, mesmo com conteúdo incompleto.",
            ],
        }
        expected = ExplanationGenerator.fallback_report(result)

        with patch(
            "pipeline.output.explanation_generator.requests.post",
            return_value=ollama_response(model_report),
        ):
            report = ExplanationGenerator.generate(result)

        self.assertEqual(report["source"], "deterministic")
        self.assertEqual(report["explanation"], expected["explanation"])
        self.assertEqual(report["details"], expected["details"])

    def test_literal_copy_is_not_attributed_to_qwen(self):
        result = sample_result()
        base = ExplanationGenerator.fallback_report(result)

        with patch(
            "pipeline.output.explanation_generator.requests.post",
            return_value=ollama_response(base),
        ):
            report = ExplanationGenerator.generate(result)

        self.assertEqual(report["source"], "deterministic")
        self.assertEqual(report["explanation"], base["explanation"])
        self.assertEqual(report["details"], base["details"])

    def test_hybrid_requires_some_accepted_qwen_text(self):
        result = sample_result()
        base = ExplanationGenerator.fallback_report(result)
        accepted_explanation = base["explanation"].replace(
            "poucas partes da notícia puderam ser verificadas",
            "somente poucas partes da notícia puderam ser verificadas",
        )
        model_report = {
            "explanation": accepted_explanation,
            "details": [
                "Um fato novo foi descoberto durante a pesquisa.",
                "A consultoria apresentou resultados nacionais.",
                "A notícia recebeu uma classificação confiável.",
            ],
        }

        with patch(
            "pipeline.output.explanation_generator.requests.post",
            return_value=ollama_response(model_report),
        ):
            report = ExplanationGenerator.generate(result)

        self.assertEqual(report["source"], "hybrid")
        self.assertEqual(report["explanation"], accepted_explanation)
        self.assertEqual(report["details"], base["details"])

    def test_changed_number_and_incomplete_sentence_are_rejected(self):
        reference = (
            'Na notícia “Pesquisa em 7 estados”, o resultado foi '
            '"evidência insuficiente".'
        )

        self.assertFalse(
            ExplanationGenerator._is_safe_rewrite(
                'Na notícia “Pesquisa em 8 estados”, o resultado foi '
                '"evidência insuficiente".',
                reference=reference,
                label="evidência insuficiente",
                require_label=True,
            )
        )
        self.assertFalse(
            ExplanationGenerator._is_safe_rewrite(
                'Na notícia “Pesquisa em 7 estados”, o resultado foi '
                '"evidência insuficiente',
                reference=reference,
                label="evidência insuficiente",
                require_label=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
