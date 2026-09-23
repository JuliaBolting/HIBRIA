# =============================================================================
# HÍBRIA — Explanation Generator
#
# Responsável por gerar a explicação curta exibida ao usuário e os tópicos de
# detalhes do resultado da análise.
#
# IMPORTANTE:
# - Não calcula o score.
# - Não altera o label.
# - Não refaz a análise.
# - Não pesquisa na web.
# - Usa Qwen localmente por meio da API HTTP do Ollama.
# - Se o Qwen/Ollama falhar, o pipeline continua normalmente.
# =============================================================================

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


class ExplanationGenerator:
    """Gera o texto final de transparência usando somente dados do pipeline."""

    DEFAULT_API_URL = "http://127.0.0.1:11434/api/chat"
    DEFAULT_MODEL = "qwen3:0.6b"
    DEFAULT_TIMEOUT = 120
    MAX_INPUT_CHARS = 4000

    OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                "maxLength": 300,
                "description": (
                    "Explicação curta do resultado em português do Brasil, "
                    "com no máximo duas frases."
                ),
            },
            "details": {
                "type": "array",
                "items": {
                    "type": "string",
                    "maxLength": 160,
                },
                "minItems": 3,
                "maxItems": 3,
                "description": (
                    "Exatamente três tópicos curtos que explicam os principais "
                    "fatores do resultado."
                ),
            },
        },
        "required": ["explanation", "details"],
        "additionalProperties": False,
    }

    @staticmethod
    def fallback_report(result: Any) -> dict[str, Any]:
        """Explicação determinística quando o serviço local não responde."""
        label = str(getattr(result, "label_final", None) or "não verificado")
        breakdown = getattr(result, "score_breakdown", None) or {}
        coverage = breakdown.get("coverage_score")
        stance = breakdown.get("stance_stats") or {}

        try:
            coverage_value = max(0.0, min(1.0, float(coverage)))
        except (TypeError, ValueError):
            coverage_value = 0.0

        if coverage_value >= 0.60:
            coverage_text = "A maior parte das afirmações recebeu evidência suficiente."
        elif coverage_value >= 0.30:
            coverage_text = "Apenas parte das afirmações recebeu evidência suficiente."
        else:
            coverage_text = "Poucas afirmações receberam evidência suficiente."

        contradictions = int(stance.get("contradict", 0) or 0)
        supports = int(stance.get("support", 0) or 0)
        relation_text = (
            f"Foram identificados {supports} apoios e {contradictions} contradições nas comparações válidas."
        )

        reputation = getattr(result, "reputation", None) or {}
        if reputation.get("status") == "evaluated":
            reputation_text = "A reputação dinâmica da fonte também foi considerada."
        else:
            reputation_text = "A fonte ainda não tinha reputação dinâmica completa."

        return {
            "explanation": (
                f'A HÍBRIA classificou a notícia como "{label}". {coverage_text}'
            )[:300],
            "details": [coverage_text, relation_text[:160], reputation_text],
        }

    @classmethod
    def generate(cls, result: Any) -> dict[str, Any] | None:
        """
        Retorna:
            {
                "explanation": str,
                "details": list[str],
            }

        Retorna None quando o serviço local estiver indisponível ou quando a
        resposta não puder ser validada. A falha não interrompe o pipeline.
        """

        api_url = os.getenv(
            "HIBRIA_QWEN_API_URL",
            cls.DEFAULT_API_URL,
        ).strip() or cls.DEFAULT_API_URL

        model = os.getenv(
            "HIBRIA_QWEN_MODEL",
            cls.DEFAULT_MODEL,
        ).strip() or cls.DEFAULT_MODEL

        timeout = cls._env_int(
            "HIBRIA_QWEN_TIMEOUT_SECONDS",
            cls.DEFAULT_TIMEOUT,
        )

        try:
            prompt = cls._build_prompt(result)

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": cls._system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "stream": False,
                "think": False,
                "format": cls.OUTPUT_SCHEMA,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 20,
                    "num_ctx": 4096,
                    "num_predict": 220,
                },
                "keep_alive": "10m",
            }

            logger.info(
                "[explanation_generator] enviando solicitação ao Qwen local "
                f"(model={model}, timeout={timeout}s, prompt={len(prompt)} chars)"
            )

            response = requests.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )

            if response.status_code >= 400:
                logger.warning(
                    "[explanation_generator] erro HTTP do Ollama/Qwen "
                    f"({response.status_code}): {response.text[:1000]}"
                )
                return None

            try:
                data = response.json()
            except ValueError:
                logger.warning(
                    "[explanation_generator] Ollama retornou resposta "
                    "que não é JSON válido"
                )
                return None

            raw_content = (
                (data.get("message") or {}).get("content")
                if isinstance(data, dict)
                else None
            )

            report = cls._parse_report(raw_content)

            if report is None:
                logger.warning(
                    "[explanation_generator] Qwen retornou saída inválida. "
                    f"Conteúdo recebido: {str(raw_content)[:1500]}"
                )
                return None

            logger.info(
                "[explanation_generator] relatório gerado com sucesso "
                f"({len(report['explanation'])} caracteres, "
                f"{len(report['details'])} detalhes)"
            )

            return report

        except requests.Timeout:
            logger.warning(
                "[explanation_generator] timeout ao consultar Qwen local "
                f"(>{timeout}s)"
            )
            return None

        except requests.RequestException as exc:
            logger.warning(
                "[explanation_generator] erro de comunicação com Ollama/Qwen: "
                f"{exc}"
            )
            return None

        except Exception as exc:
            logger.warning(
                "[explanation_generator] erro inesperado: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

    # =========================================================================
    # Prompt
    # =========================================================================

    @staticmethod
    def _system_prompt() -> str:
        return """
/no_think

Você é o módulo de redação final da HÍBRIA.

O resultado da análise já foi calculado pelo sistema.
Sua única função é explicar esse resultado de maneira simples e curta.

REGRAS OBRIGATÓRIAS:
- Não altere a classificação final.
- Não recalcule o resultado.
- Não pesquise e não utilize conhecimento externo.
- Use somente os dados fornecidos.
- Não invente fatos, fontes ou evidências.
- Não apresente scores, pesos, percentuais ou valores internos.
- Não mencione Qwen, BERTimbau, modelos, classificadores, pipeline, aggregator,
  stance ou nomes de componentes internos.
- Não mencione divergências entre classificadores internos.
- Considere somente a classificação final como o resultado oficial.
- A reputação da fonte é apenas um fator complementar.
- Ausência de evidência não significa falsidade.
- Só diga que há contradição quando as evidências fornecidas realmente
  indicarem contradição.
- Escreva para uma pessoa comum, sem linguagem técnica.
- Não use Markdown.
- Não explique seu raciocínio.
- Responda em português do Brasil.

A saída deve conter:
- "explanation": uma ou duas frases curtas, com no máximo 300 caracteres.
- "details": exatamente três frases curtas, cada uma com no máximo 160 caracteres.

Não repita nos detalhes exatamente a mesma informação da explicação.
""".strip()

    @classmethod
    def _build_prompt(cls, result: Any) -> str:
        label = getattr(result, "label_final", None)
        breakdown = getattr(result, "score_breakdown", None)
        reputation = getattr(result, "reputation", None)
        claims = getattr(result, "claims", None)
        stance_results = getattr(result, "stance_results", None)
        retrieval_results = getattr(result, "retrieval_results", None)
        title = getattr(result, "title", "") or ""

        claim_texts: list[str] = []
        for claim in (claims or [])[:3]:
            text = getattr(claim, "text", None)
            if text:
                claim_texts.append(str(text)[:220])

        stance_summary: list[dict[str, Any]] = []
        for item in (stance_results or [])[:5]:
            try:
                if hasattr(item, "to_dict"):
                    item = item.to_dict()

                if not isinstance(item, dict):
                    continue

                stance_summary.append(
                    {
                        "stance": item.get("stance"),
                        "confidence": item.get("confidence"),
                        "reason": str(item.get("reason") or "")[:160],
                        "source": str(item.get("source") or "")[:100],
                    }
                )

            except Exception:
                continue

        evidence_summary: list[dict[str, Any]] = []

        for retrieval in (retrieval_results or [])[:3]:
            claim = getattr(retrieval, "claim", None)
            claim_text = str(
                getattr(claim, "text", "") or ""
            )[:220]

            for evidence in list(
                getattr(retrieval, "evidences", []) or []
            )[:1]:

                evidence_summary.append(
                    {
                        "claim": claim_text,
                        "source": str(
                            getattr(evidence, "source", "") or ""
                        )[:100],
                        "title": str(
                            getattr(evidence, "title", "") or ""
                        )[:140],
                        "stance": getattr(
                            evidence,
                            "stance",
                            None,
                        ),
                        "trusted_source": getattr(
                            evidence,
                            "trusted_source",
                            None,
                        ),
                        "excerpt": str(
                            getattr(evidence, "text", "") or ""
                        )[:200],
                    }
                )

        # Mantemos apenas os dados necessários para explicar
        # cobertura e relação entre afirmações e evidências.
        safe_breakdown: dict[str, Any] = {}

        if isinstance(breakdown, dict):
            for key in (
                "evidence_score",
                "coverage_score",
                "reputation_status",
                "stance_stats",
            ):
                if key in breakdown:
                    safe_breakdown[key] = breakdown.get(key)

        reputation_summary: dict[str, Any] = {}

        if isinstance(reputation, dict):
            for key in (
                "status",
                "source_name",
                "note",
                "classification",
            ):
                value = reputation.get(key)

                if value is not None:
                    reputation_summary[key] = value

        context = {
            "titulo": title[:300],
            "resultado_calculado": {
                "label": label,
            },
            "componentes": safe_breakdown,
            "reputacao_fonte": reputation_summary,
            "claims_principais": claim_texts,
            "relacoes_claim_evidencia": stance_summary,
            "amostra_evidencias_recuperadas": evidence_summary,
        }

        prompt = cls._render_prompt(context)

        # Evita cortar JSON no meio.
        # Se ficar grande, reduz progressivamente as amostras.
        if len(prompt) > cls.MAX_INPUT_CHARS:
            context["amostra_evidencias_recuperadas"] = (
                evidence_summary[:1]
            )

            context["relacoes_claim_evidencia"] = (
                stance_summary[:3]
            )

            context["claims_principais"] = (
                claim_texts[:2]
            )

            prompt = cls._render_prompt(context)

        if len(prompt) > cls.MAX_INPUT_CHARS:
            context["amostra_evidencias_recuperadas"] = []

            context["relacoes_claim_evidencia"] = (
                stance_summary[:2]
            )

            context["titulo"] = str(
                context.get("titulo") or ""
            )[:180]

            prompt = cls._render_prompt(context)

        if len(prompt) > cls.MAX_INPUT_CHARS:
            context["relacoes_claim_evidencia"] = []

            context["claims_principais"] = (
                claim_texts[:1]
            )

            context["reputacao_fonte"] = {
                key: value
                for key, value in reputation_summary.items()
                if key in {"status", "classification"}
            }

            prompt = cls._render_prompt(context)

        logger.debug(
            "[explanation_generator] prompt final com %s caracteres",
            len(prompt),
        )

        return prompt

    @staticmethod
    def _render_prompt(
        context: dict[str, Any],
    ) -> str:

        context_json = json.dumps(
            context,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return f"""
/no_think

Escreva a explicação final da análise usando exclusivamente o contexto abaixo.

Retorne somente o objeto JSON solicitado pelo schema.

A "explanation" deve explicar de forma simples o principal motivo da
classificação final.

Os três itens de "details" devem destacar fatores concretos encontrados nas
evidências, sem repetir a explanation.

CONTEXTO:
{context_json}
""".strip()

    # =========================================================================
    # Validação da resposta
    # =========================================================================

    @classmethod
    def _parse_report(
        cls,
        raw_content: Any,
    ) -> dict[str, Any] | None:

        if not isinstance(
            raw_content,
            str,
        ) or not raw_content.strip():

            return None

        content = raw_content.strip()

        # Remove bloco Markdown se o modelo insistir.
        if content.startswith("```"):
            lines = content.splitlines()

            if lines:
                lines = lines[1:]

            if (
                lines
                and lines[-1].strip() == "```"
            ):
                lines = lines[:-1]

            content = "\n".join(
                lines
            ).strip()

        # Caso exista algum texto antes ou depois,
        # aproveita somente o objeto JSON.
        start = content.find("{")
        end = content.rfind("}")

        if (
            start >= 0
            and end > start
        ):
            content = content[
                start : end + 1
            ]

        try:
            parsed = json.loads(content)

        except json.JSONDecodeError:
            return None

        if not isinstance(
            parsed,
            dict,
        ):
            return None

        explanation = str(
            parsed.get("explanation")
            or parsed.get("evidence")
            or parsed.get("evidencia")
            or ""
        ).strip()

        raw_details = (
            parsed.get("details")
            or parsed.get("detalhes")
            or []
        )

        if isinstance(
            raw_details,
            str,
        ):
            raw_details = [
                raw_details
            ]

        if (
            not explanation
            or not isinstance(
                raw_details,
                list,
            )
        ):
            return None

        details: list[str] = []

        for item in raw_details:
            text = str(
                item or ""
            ).strip()

            if not text:
                continue

            if text not in details:
                details.append(
                    text[:160]
                )

            if len(details) >= 3:
                break

        if not details:
            return None

        return {
            "explanation": explanation[:300],
            "details": details,
        }

    @staticmethod
    def _env_int(
        name: str,
        default: int,
    ) -> int:

        value = os.getenv(name)

        if (
            value is None
            or not value.strip()
        ):
            return default

        try:
            return max(
                1,
                int(value),
            )

        except ValueError:
            return default
