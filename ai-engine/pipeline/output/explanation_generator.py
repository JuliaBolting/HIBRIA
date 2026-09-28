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
import re
from typing import Any

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


class ExplanationGenerator:
    """Gera o texto final de transparência usando somente dados do pipeline."""

    DEFAULT_API_URL = "http://127.0.0.1:11434/api/chat"
    DEFAULT_MODEL = "qwen3:1.7b"
    DEFAULT_TIMEOUT = 120
    MAX_INPUT_CHARS = 8000
    RELATION_LABELS = {
        "confirmed": "recebeu confirmação nas verificações",
        "divergent": "apresentou informações diferentes nas verificações",
        "mixed": "recebeu confirmações e também apresentou diferenças",
        "not_confirmed": "não recebeu confirmação direta",
    }

    OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                # O schema deixa espaço para o modelo concluir a frase; o
                # parser aplica o limite visual de 300 caracteres depois.
                "maxLength": 600,
                "description": (
                    "Explicação objetiva do resultado em português do Brasil, "
                    "com uma ou duas frases completas e no máximo 300 caracteres."
                ),
            },
            "details": {
                "type": "array",
                "items": {
                    "type": "string",
                    "maxLength": 420,
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

    @classmethod
    def fallback_report(cls, result: Any) -> dict[str, Any]:
        """Explicação determinística quando o serviço local não responde."""
        context = cls._build_context(result)
        return cls._fallback_report_from_context(result, context)

    @classmethod
    def _fallback_report_from_context(
        cls,
        result: Any,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "explanation": cls._deterministic_explanation(
                result,
                context=context,
            ),
            "details": cls._deterministic_details(
                result,
                context=context,
            ),
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
            context = cls._build_context(result)
            fallback = cls._fallback_report_from_context(result, context)
            prompt = cls._build_prompt(result, context=context)

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
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "top_k": 20,
                    "num_ctx": 4096,
                    "num_predict": cls._env_int(
                        "HIBRIA_QWEN_MAX_OUTPUT_TOKENS",
                        420,
                    ),
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
                return {
                    **fallback,
                    "source": "deterministic",
                }

            report = cls._validate_model_report(
                report,
                result=result,
                context=context,
                fallback=fallback,
            )

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

Você é o redator final de uma verificação de notícia. A classificação,
as contagens e as notas já foram calculadas. Sua tarefa é responder, em linguagem
simples: "por que esta notícia recebeu esse resultado?". Não refaça a análise.

Construa uma explicação equilibrada usando os fatores que realmente formaram a
nota. Mostre primeiro o que contribuiu positivamente e depois a limitação que
foi decisiva. A boa reputação da fonte e o sinal textual são fatores auxiliares:
eles podem aumentar a nota, mas não comprovam os fatos da notícia.

Escreva para uma pessoa leiga:
- "explanation": duas ou três frases naturais que contrastem os sinais
  favoráveis com o fator decisivo, expliquem o efeito no índice final e mencionem
  a classificação uma única vez. Use os números da ficha quando eles tornarem a
  explicação mais clara.
- detalhe 1: informe corretamente quantas confirmações, diferenças e casos sem
  confirmação direta ocorreram.
- detalhe 2: explique quais fatores auxiliares contribuíram positivamente e
  deixe claro que eles não confirmam os fatos por conta própria.
- detalhe 3: explique quanto do conteúdo importante recebeu evidência válida,
  por que isso foi decisivo e como afetou a nota.

Não troque "diferença" por "sem confirmação": são resultados distintos.
Não use as palavras "afirmação", "claim", "cobertura", "comparações", "score",
"pipeline", "modelo", "classificador", "BERTimbau" ou "similaridade".
Não cite sites ou nomes de veículos. Não diga "a HÍBRIA encontrou". Não
resuma o assunto da notícia nem transforme as informações verificadas em uma
lista. Não repita a classificação mais de uma vez em toda a resposta. Não
repita ideias entre os quatro textos. Use português do Brasil, frases completas
e pontuação final.

Não invente fatos ou números. Retorne somente o JSON do schema, com no máximo
300 caracteres na explicação e 260 em cada um dos três detalhes.
""".strip()

    @classmethod
    def _build_prompt(
        cls,
        result: Any,
        *,
        context: dict[str, Any] | None = None,
    ) -> str:
        safe_context = context or cls._build_context(result)
        prompt = cls._render_prompt(safe_context)

        # Os textos concretos já chegam truncados, mas esta proteção mantém o
        # JSON completo mesmo se um objeto externo trouxer campos inesperados.
        if len(prompt) > cls.MAX_INPUT_CHARS:
            reduced = dict(safe_context)
            reduced["verificacoes_relevantes"] = list(
                safe_context.get("verificacoes_relevantes") or []
            )[:2]
            prompt = cls._render_prompt(reduced)

        if len(prompt) > cls.MAX_INPUT_CHARS:
            reduced["verificacoes_relevantes"] = list(
                safe_context.get("verificacoes_relevantes") or []
            )[:1]
            reduced["assunto_da_noticia"] = cls._truncate_text(
                safe_context.get("assunto_da_noticia"),
                120,
            )
            prompt = cls._render_prompt(reduced)

        logger.debug(
            "[explanation_generator] prompt final com %s caracteres",
            len(prompt),
        )
        return prompt

    @classmethod
    def _build_context(cls, result: Any) -> dict[str, Any]:
        """Monta a ficha completa, mas sem páginas ou fontes externas brutas."""
        breakdown = getattr(result, "score_breakdown", None) or {}
        stance = breakdown.get("stance_stats") or {}
        counts = {
            "confirmacoes": int(stance.get("support", 0) or 0),
            "diferencas": int(stance.get("contradict", 0) or 0),
            "sem_confirmacao_direta": int(stance.get("neutral", 0) or 0)
            + int(stance.get("insufficient", 0) or 0),
        }

        factors = {
            "indice_final": getattr(result, "score_final", None),
            "forca_das_evidencias": breakdown.get("evidence_score"),
            "parte_do_conteudo_com_evidencia_valida": breakdown.get(
                "coverage_score"
            ),
            "reputacao_da_fonte": breakdown.get("reputation_score"),
            "sinal_textual_auxiliar": breakdown.get("bertimbau_score"),
        }

        return {
            "classificacao_final_apenas_para_consistencia": str(
                getattr(result, "label_final", None) or "não verificado"
            ),
            "assunto_da_noticia": cls._truncate_text(
                getattr(result, "title", "") or "",
                180,
            ),
            "resumo_numerico": counts,
            "fatores_que_formaram_a_nota": factors,
            "motivo_determinante": cls._decision_reason(result),
            "quanto_do_conteudo_importante_foi_verificado": (
                cls._coverage_plain_description(
                    breakdown.get("coverage_score")
                )
            ),
            "verificacoes_relevantes": cls._build_verification_points(
                getattr(result, "retrieval_results", None) or [],
                getattr(result, "stance_results", None) or [],
            ),
            "limite_de_interpretacao": (
                "A falta de confirmação, sozinha, não prova que a notícia "
                "seja falsa."
            ),
        }

    @classmethod
    def _decision_reason(cls, result: Any) -> str:
        """Explica a regra decisiva sem pedir ao Qwen que a deduza."""
        breakdown = getattr(result, "score_breakdown", None) or {}
        label = str(getattr(result, "label_final", None) or "").casefold()
        stance = breakdown.get("stance_stats") or {}

        try:
            coverage = float(breakdown.get("coverage_score") or 0.0)
            if coverage > 1:
                coverage /= 100.0
        except (TypeError, ValueError):
            coverage = 0.0

        try:
            contradiction_rate = float(
                stance.get("contradiction_rate") or 0.0
            )
        except (TypeError, ValueError):
            contradiction_rate = 0.0

        min_partial = float(
            breakdown.get("min_coverage_for_partial") or 0.20
        )
        min_unreliable = float(
            breakdown.get("min_contradiction_rate_for_unreliable") or 0.40
        )

        if label == "não confiável" and contradiction_rate >= min_unreliable:
            return (
                "A quantidade proporcional de diferenças atingiu o limite "
                "usado para reduzir a classificação."
            )
        if label in {"evidência insuficiente", "não verificado"}:
            if coverage == 0:
                return (
                    "Não houve evidência válida suficiente para verificar os "
                    "pontos principais da notícia."
                )
            if coverage < min_partial:
                return (
                    "A pequena parte do conteúdo com evidência válida foi o "
                    "principal motivo da classificação."
                )
        if label == "confiável":
            return (
                "A quantidade de confirmações, o alcance da verificação e a "
                "nota conjunta atenderam aos critérios da classificação."
            )
        if label == "parcialmente confiável":
            return (
                "As verificações sustentaram parte do conteúdo, mas não em "
                "quantidade suficiente para a classificação mais alta."
            )
        return "O conjunto dos fatores disponíveis determinou a classificação."

    @classmethod
    def _build_verification_points(
        cls,
        retrieval_results: list[Any],
        stance_results: list[Any],
    ) -> list[dict[str, Any]]:
        """Resume por informação apenas relações já calculadas pelo pipeline."""
        stances_by_claim: dict[str, list[dict[str, Any]]] = {}
        seen_stances: set[tuple[str, str, str]] = set()

        for raw_item in stance_results:
            item = raw_item.to_dict() if hasattr(raw_item, "to_dict") else raw_item
            if not isinstance(item, dict):
                continue
            claim_id = str(item.get("claim_id") or "")
            relation = str(item.get("stance") or "")
            evidence_id = str(item.get("evidence_id") or "")
            key = (claim_id, evidence_id, relation)
            if not claim_id or key in seen_stances:
                continue
            seen_stances.add(key)
            stances_by_claim.setdefault(claim_id, []).append(item)

        points: list[dict[str, Any]] = []
        for retrieval in retrieval_results:
            claim = getattr(retrieval, "claim", None)
            claim_id = str(getattr(claim, "claim_id", "") or "")
            claim_text = cls._truncate_text(
                getattr(claim, "text", "") or "",
                190,
            ).rstrip("… .!?;:")
            if not claim_text:
                continue

            relations = [
                str(item.get("stance") or "")
                for item in stances_by_claim.get(claim_id, [])
            ]
            if not relations:
                relations = [
                    str(getattr(evidence, "stance", None) or "")
                    for evidence in (
                        getattr(retrieval, "evidences", None) or []
                    )
                ]

            supports = relations.count("support")
            contradictions = relations.count("contradict")
            if supports and contradictions:
                status = "mixed"
            elif supports:
                status = "confirmed"
            elif contradictions:
                status = "divergent"
            else:
                status = "not_confirmed"

            points.append(
                {
                    "informacao_da_noticia": claim_text,
                    "resultado_calculado": cls.RELATION_LABELS[status],
                    "tipo_resultado": status,
                }
            )

        # O pipeline limita normalmente a análise a dez informações. Todas são
        # enviadas para que o modelo compreenda o conjunto, não para enumerá-las.
        return points[:10]

    @classmethod
    def _deterministic_details(
        cls,
        result: Any,
        *,
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        """Explica os três fatores com um ponto concreto quando disponível."""
        safe_context = context or cls._build_context(result)
        counts = safe_context.get("resumo_numerico") or {}
        supports = int(counts.get("confirmacoes", 0) or 0)
        contradictions = int(counts.get("diferencas", 0) or 0)
        unconfirmed = int(counts.get("sem_confirmacao_direta", 0) or 0)

        count_parts: list[str] = []
        if supports:
            count_parts.append(
                f"{supports} "
                f"{cls._plural(supports, 'trecho recebeu', 'trechos receberam')} "
                "confirmação"
            )
        if contradictions:
            count_parts.append(
                f"{contradictions} "
                f"{cls._plural(contradictions, 'apresentou', 'apresentaram')} "
                "diferenças"
            )
        if unconfirmed:
            count_parts.append(
                f"{unconfirmed} "
                f"{cls._plural(unconfirmed, 'ficou', 'ficaram')} sem confirmação direta"
            )

        if count_parts:
            first_detail = f"Nas verificações, {', '.join(count_parts)}."
        else:
            first_detail = (
                "Não foi possível confirmar diretamente as informações "
                "principais da notícia."
            )

        factors = safe_context.get("fatores_que_formaram_a_nota") or {}
        reputation = cls._metric_text(factors.get("reputacao_da_fonte"))
        textual = cls._metric_text(factors.get("sinal_textual_auxiliar"))
        evidence = cls._metric_text(factors.get("forca_das_evidencias"))
        coverage = cls._metric_text(
            factors.get("parte_do_conteudo_com_evidencia_valida")
        )

        positive_parts: list[str] = []
        if reputation:
            positive_parts.append(
                f"a reputação da fonte foi {reputation} de 100"
            )
        if textual:
            positive_parts.append(
                f"o sinal auxiliar do texto foi {textual} de 100"
            )
        if positive_parts:
            second_detail = (
                f"Como fatores favoráveis, {' e '.join(positive_parts)}. "
                "Esses sinais ajudam na nota, mas não confirmam os fatos sozinhos."
            )
        else:
            second_detail = (
                "Os fatores auxiliares foram considerados na nota, mas não "
                "confirmam os fatos da notícia por conta própria."
            )

        reach = str(
            safe_context.get("quanto_do_conteudo_importante_foi_verificado")
            or "não foi possível medir quanto do conteúdo pôde ser verificado"
        )
        reach = reach[:1].upper() + reach[1:]
        metric_parts: list[str] = []
        if evidence:
            metric_parts.append(
                f"as evidências localizadas tiveram força {evidence} de 100"
            )
        if coverage:
            metric_parts.append(
                f"somente {coverage}% do conteúdo importante teve evidência válida"
            )
        measured = ", mas ".join(metric_parts)
        if measured:
            third_detail = (
                f"{measured[:1].upper() + measured[1:]}. Esse alcance limitado "
                "foi decisivo e reduziu a nota."
            )
        else:
            third_detail = (
                f"{reach}. Esse alcance foi decisivo para a nota final."
            )

        return [
            cls._limit_output_text(first_detail, 260),
            cls._limit_output_text(second_detail, 260),
            cls._limit_output_text(third_detail, 260),
        ]

    @staticmethod
    def _plural(amount: int, singular: str, plural: str) -> str:
        return singular if amount == 1 else plural

    @classmethod
    def _deterministic_explanation(
        cls,
        result: Any,
        *,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Resume a decisão sem vocabulário técnico ou oposição falsa."""
        safe_context = context or cls._build_context(result)
        factors = safe_context.get("fatores_que_formaram_a_nota") or {}
        label = str(getattr(result, "label_final", None) or "não verificado")
        final_score = cls._metric_text(factors.get("indice_final"))
        coverage = cls._metric_text(
            factors.get("parte_do_conteudo_com_evidencia_valida")
        )
        reputation = cls._metric_text(factors.get("reputacao_da_fonte"))
        textual = cls._metric_text(factors.get("sinal_textual_auxiliar"))

        favorable: list[str] = []
        if reputation:
            favorable.append(f"reputação da fonte de {reputation} em 100")
        if textual:
            favorable.append(f"sinal textual de {textual} em 100")
        favorable_text = " e ".join(favorable)

        if favorable_text and coverage:
            opening = (
                f"Mesmo com {favorable_text}, somente {coverage}% do conteúdo "
                "importante recebeu evidência válida"
            )
        elif coverage:
            opening = (
                f"Somente {coverage}% do conteúdo importante recebeu evidência "
                "válida"
            )
        else:
            opening = str(
                safe_context.get("motivo_determinante")
                or "O conjunto das verificações determinou o resultado"
            ).rstrip(".")

        if final_score:
            consequence = (
                f"Essa limitação deixou o índice em {final_score} de 100 e "
                f"levou ao resultado \"{label}\""
            )
        else:
            consequence = f"Essa limitação levou ao resultado \"{label}\""

        if label.casefold() in {"evidência insuficiente", "não verificado"}:
            consequence += "; a falta de confirmação não prova que a notícia seja falsa"

        explanation = f"{opening}. {consequence}."

        return cls._limit_output_text(explanation, 300)

    @staticmethod
    def _metric_text(value: Any) -> str:
        """Formata uma métrica para leitura humana sem casas desnecessárias."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        if number.is_integer():
            return str(int(number))
        return f"{number:.1f}".replace(".", ",")

    @staticmethod
    def _is_plain_explanation(value: Any) -> bool:
        """Rejeita contradições recorrentes e jargão pouco útil ao leitor."""
        original = " ".join(str(value or "").split())
        text = original.casefold()
        if not text:
            return False
        if original.endswith("…") or not re.search(r"[.!?][\"'”’]?$", original):
            return False

        forbidden = (
            "mas há apoio",
            "porém há apoio",
            "cobertura",
            "comparações",
            "apoio externo",
            "evidências externas",
            "claim",
            "afirmação",
            "afirmações",
            "afirmacao",
            "afirmacoes",
            "primeira afirmação",
            "segunda afirmação",
            "terceira afirmação",
            "a análise encontrou que",
            "resultado foi calculado com base",
            "complexidade do tema",
            "avaliação equilibrada",
        )
        return not any(term in text for term in forbidden)

    @classmethod
    def _are_plain_details(
        cls,
        details: Any,
        *,
        explanation: str,
    ) -> bool:
        """Aceita somente três tópicos legíveis, distintos e não repetitivos."""
        if not isinstance(details, list) or len(details) != 3:
            return False

        normalized = [" ".join(str(item or "").split()) for item in details]
        if any(not cls._is_plain_explanation(item) for item in normalized):
            return False
        if len({item.casefold() for item in normalized}) != 3:
            return False

        for index, item in enumerate(normalized):
            if cls._text_overlap(item, explanation) >= 0.86:
                return False
            for other in normalized[index + 1 :]:
                if cls._text_overlap(item, other) >= 0.72:
                    return False

        return True

    @classmethod
    def _validate_model_report(
        cls,
        report: dict[str, Any],
        *,
        result: Any,
        context: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        """Aceita apenas blocos que preservem a ficha calculada."""
        explanation_accepted = cls._is_safe_explanation(
            report.get("explanation"),
            result=result,
            context=context,
        )
        explanation = (
            report["explanation"]
            if explanation_accepted
            else fallback["explanation"]
        )
        details_accepted = cls._are_safe_details(
            report.get("details"),
            result=result,
            context=context,
            explanation=explanation,
        )

        final_report = {
            "explanation": explanation,
            "details": (
                report["details"]
                if details_accepted
                else fallback["details"]
            ),
        }
        if explanation_accepted and details_accepted:
            final_report["source"] = "qwen"
        elif explanation_accepted or details_accepted:
            final_report["source"] = "hybrid"
        else:
            final_report["source"] = "deterministic"
        return final_report

    @classmethod
    def _is_safe_explanation(
        cls,
        value: Any,
        *,
        result: Any,
        context: dict[str, Any],
    ) -> bool:
        text = " ".join(str(value or "").split())
        label = str(getattr(result, "label_final", None) or "")
        if not cls._is_plain_explanation(text):
            return False
        if cls._has_conflicting_label(text, label):
            return False
        if cls._label_occurrences(text, label) > 1:
            return False
        if not cls._numbers_are_allowed(text, context):
            return False

        points = list(context.get("verificacoes_relevantes") or [])
        if any(
            cls._text_overlap(
                text,
                str(point.get("informacao_da_noticia") or ""),
            ) >= 0.55
            for point in points
        ):
            return False

        return True

    @classmethod
    def _are_safe_details(
        cls,
        details: Any,
        *,
        result: Any,
        context: dict[str, Any],
        explanation: str,
    ) -> bool:
        if not cls._are_plain_details(details, explanation=explanation):
            return False

        normalized = [" ".join(str(item or "").split()) for item in details]
        combined = " ".join(normalized)
        label = str(getattr(result, "label_final", None) or "")
        if cls._has_conflicting_label(combined, label):
            return False
        if cls._label_occurrences(
            f"{explanation} {combined}",
            label,
        ) > 1:
            return False
        if not cls._numbers_are_allowed(combined, context):
            return False

        required = cls._required_count_tokens(context)
        if not required.issubset(set(cls._number_tokens(combined))):
            return False
        if not cls._counts_keep_their_meaning(normalized[0], context):
            return False

        consequence_terms = (
            "nota",
            "resultado",
            "classificação",
            "notícia seja falsa",
            "notícia é falsa",
        )
        if not any(
            term in normalized[2].casefold()
            for term in consequence_terms
        ):
            return False
        if re.search(
            r"(?:classifica(?:ção|cao)|resultado).{0,80}(?:porque|pois)"
            r".{0,100}não (?:prova|significa).{0,80}fals",
            normalized[2].casefold(),
        ):
            return False
        return True

    @classmethod
    def _counts_keep_their_meaning(
        cls,
        text: str,
        context: dict[str, Any],
    ) -> bool:
        counts = context.get("resumo_numerico") or {}
        expected_terms = {
            "confirmacoes": ("confirm",),
            "diferencas": ("diferen", "diverg"),
            "sem_confirmacao_direta": (
                "sem confirmação",
                "sem confirmacao",
                "não receberam confirmação",
                "nao receberam confirmacao",
                "não tiveram confirmação",
                "nao tiveram confirmacao",
                "não foram confirm",
                "nao foram confirm",
            ),
        }
        normalized = " ".join(str(text or "").casefold().split())

        for key, terms in expected_terms.items():
            amount = int(counts.get(key, 0) or 0)
            if amount <= 0:
                continue
            matches = list(
                re.finditer(rf"(?<!\w){amount}(?!\w)", normalized)
            )
            if not matches:
                return False
            if not any(
                any(
                    term in normalized[
                        max(0, match.start() - 25) : match.end() + 80
                    ]
                    for term in terms
                )
                for match in matches
            ):
                return False
        return True

    @classmethod
    def _numbers_are_allowed(
        cls,
        value: Any,
        context: dict[str, Any],
    ) -> bool:
        # As métricas públicas usam escala de 0 a 100. O limite da escala pode
        # aparecer como "de 100" mesmo quando o JSON contém apenas o valor.
        allowed: set[str] = {"100"}
        for token in cls._number_tokens(
            json.dumps(context, ensure_ascii=False)
        ):
            normalized = token.replace(",", ".")
            allowed.add(normalized)
            try:
                number = float(normalized)
            except ValueError:
                continue
            allowed.add(f"{number:.1f}")
            allowed.add(f"{number:.2f}")
            if number.is_integer():
                allowed.add(str(int(number)))

        received = {
            token.replace(",", ".")
            for token in cls._number_tokens(value)
        }
        return received.issubset(allowed)

    @classmethod
    def _required_count_tokens(cls, context: dict[str, Any]) -> set[str]:
        counts = context.get("resumo_numerico") or {}
        return {
            str(int(value))
            for value in counts.values()
            if int(value or 0) > 0
        }

    @staticmethod
    def _number_tokens(value: Any) -> list[str]:
        return re.findall(
            r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)",
            str(value or ""),
        )

    @staticmethod
    def _label_occurrences(value: Any, label: str) -> int:
        expected = " ".join(str(label or "").casefold().split())
        text = " ".join(str(value or "").casefold().split())
        return text.count(expected) if expected else 0

    @classmethod
    def _has_conflicting_label(cls, value: Any, label: str) -> bool:
        expected = " ".join(str(label or "").casefold().split())
        text = " ".join(str(value or "").casefold().split())
        if expected:
            text = text.replace(expected, " ")

        labels = (
            "parcialmente confiável",
            "evidência insuficiente",
            "não verificado",
            "não confiável",
            "confiável",
        )
        return any(
            candidate != expected
            and re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", text)
            for candidate in labels
        )

    @staticmethod
    def _text_overlap(first: str, second: str) -> float:
        """Mede repetição lexical ignorando conectivos comuns."""
        ignored = {
            "a", "as", "o", "os", "um", "uma", "de", "da", "das", "do",
            "dos", "e", "em", "na", "nas", "no", "nos", "para", "por",
            "que", "com", "como", "foi", "foram", "isso", "esta", "este",
        }

        def tokens(value: str) -> set[str]:
            return {
                token
                for token in re.findall(r"[a-záàâãéêíóôõúç]+", value.casefold())
                if len(token) >= 3 and token not in ignored
            }

        first_tokens = tokens(first)
        second_tokens = tokens(second)
        smaller = min(len(first_tokens), len(second_tokens))
        if smaller == 0:
            return 0.0
        return len(first_tokens & second_tokens) / smaller

    @staticmethod
    def _truncate_text(value: Any, max_chars: int) -> str:
        """Reduz um texto sem deixar fragmentos de palavras como "pi"."""
        text = " ".join(str(value or "").split())
        if len(text) <= max_chars:
            return text

        shortened = text[: max_chars + 1]
        last_space = shortened.rfind(" ")
        if last_space > 0:
            shortened = shortened[:last_space]
        else:
            shortened = shortened[:max_chars]

        return shortened.rstrip(" ,;:-") + "…"

    @classmethod
    def _limit_output_text(cls, value: Any, max_chars: int) -> str:
        """Limita saída extensa preferindo encerrar em uma frase completa."""
        text = " ".join(str(value or "").split())
        if len(text) <= max_chars:
            return text

        candidate = text[: max_chars + 1]
        sentence_end = max(
            candidate.rfind("."),
            candidate.rfind("!"),
            candidate.rfind("?"),
        )
        if sentence_end >= max_chars // 5:
            return candidate[: sentence_end + 1].strip()

        return cls._truncate_text(text, max_chars)

    @staticmethod
    def _coverage_description(value: Any) -> str:
        try:
            coverage = float(value)
        except (TypeError, ValueError):
            return "não informada"

        if coverage > 1.0:
            coverage /= 100.0

        if coverage >= 0.60:
            return "ampla"
        if coverage >= 0.30:
            return "parcial"
        if coverage > 0:
            return "baixa"
        return "nenhuma"

    @classmethod
    def _coverage_plain_description(cls, value: Any) -> str:
        coverage = cls._coverage_description(value)
        if coverage == "ampla":
            return "a maior parte das informações importantes pôde ser verificada"
        if coverage == "parcial":
            return "apenas parte das informações importantes pôde ser verificada"
        if coverage == "baixa":
            return "somente uma pequena parte das informações importantes pôde ser verificada"
        return "não houve informação suficiente para verificar os pontos principais"

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

Use somente a ficha abaixo. Ela contém tudo o que foi calculado. As
"verificacoes_relevantes" servem apenas para você compreender o conjunto; não
as copie, não as enumere e não faça um resumo do tema da notícia.

A resposta deve explicar a formação da nota:
1. reconheça os fatores favoráveis de "fatores_que_formaram_a_nota";
2. contraste-os com "quanto_do_conteudo_importante_foi_verificado" e
   "motivo_determinante";
3. relacione essa limitação ao índice e à classificação final;
4. não trate reputação da fonte nem sinal textual como prova factual.

O primeiro detalhe deve manter em algarismos todas as contagens maiores que
zero de "resumo_numerico" e usar cada uma com seu significado correto. O
segundo deve explicar os fatores auxiliares favoráveis. O terceiro deve
explicar o alcance limitado das evidências e seu efeito na nota. Retorne
somente o objeto JSON solicitado, sem marcadores ou numeração.

FICHA VALIDADA:
{context_json}
""".strip()

    # =========================================================================
    # Validação da resposta
    # =========================================================================

    @classmethod
    def _parse_report(
        cls,
        raw_content: Any,
        fallback_details: list[str] | None = None,
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

        explanation = cls._normalize_model_text(
            parsed.get("explanation")
            or parsed.get("evidence")
            or parsed.get("evidencia")
            or ""
        )

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

        raw_clean_details: list[str] = []

        for item in raw_details:
            text = cls._normalize_model_text(item)
            text = re.sub(
                r"^(?:\d{1,2}\s*[.)\-:]|[-•])\s*",
                "",
                text,
            )

            if not text:
                continue

            raw_clean_details.append(
                cls._limit_output_text(text, 260)
            )

            if len(raw_clean_details) >= 3:
                break

        normalized_unique = {
            item.casefold()
            for item in raw_clean_details
        }
        repeated_details = len(normalized_unique) != len(raw_clean_details)

        details: list[str] = [] if repeated_details else raw_clean_details

        for item in fallback_details or []:
            if len(details) >= 3:
                break
            text = cls._normalize_model_text(item)
            if not text:
                continue
            if text.casefold() not in {detail.casefold() for detail in details}:
                details.append(cls._limit_output_text(text, 260))

        if len(details) != 3:
            return None

        return {
            "explanation": cls._limit_output_text(explanation, 300),
            "details": details,
        }

    @staticmethod
    def _normalize_model_text(value: Any) -> str:
        """Normaliza espaços e pequenos erros recorrentes do modelo local."""
        text = " ".join(str(value or "").split())
        substitutions = (
            (r"\bconfirmacoes\b", "confirmações"),
            (r"\bdiferencas\b", "diferenças"),
            (r"\bevidencias\b", "evidências"),
            (r"\bempatados? técnico\b", "tecnicamente empatados"),
            (r"\bempatadas? técnico\b", "tecnicamente empatadas"),
        )
        for pattern, replacement in substitutions:
            text = re.sub(
                pattern,
                replacement,
                text,
                flags=re.IGNORECASE,
            )
        text = re.sub(
            r"\bA\s+H[IÍ](?:BRIA|BIRA|BRA)\b",
            "A análise",
            text,
            flags=re.IGNORECASE,
        )
        return re.sub(
            r"\bH[IÍ](?:BRIA|BIRA|BRA)\b",
            "HÍBRIA",
            text,
            flags=re.IGNORECASE,
        )

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
