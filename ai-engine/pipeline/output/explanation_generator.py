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
    MAX_INPUT_CHARS = 4200
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

Você é somente o redator final de uma verificação de notícia. A classificação,
as contagens e o resultado de cada verificação já foram calculados. Não refaça
a análise e não decida se a notícia é verdadeira ou falsa.

Use exclusivamente a ficha recebida. Ela contém um assunto, um resumo numérico,
o alcance da verificação e até três informações concretas da notícia. Não use
conhecimento próprio e não acrescente fatos, pessoas, datas, números, pesquisas,
fontes ou conclusões que não estejam nessa ficha.

A interface já mostra a classificação em destaque. Explique os motivos sem
repetir a classificação. Se for indispensável citá-la, faça isso uma única vez
em toda a resposta.

Escreva para uma pessoa leiga:
- "explanation": uma ou duas frases que mencionem ao menos uma informação
  concreta da ficha e expliquem, de forma direta, o que foi possível verificar.
- primeiro detalhe: explique uma informação concreta e o resultado de sua
  verificação; não escreva "primeira afirmação" ou "claim".
- segundo detalhe: apresente exatamente as contagens numéricas recebidas e diga
  quanto do conteúdo importante pôde ser verificado.
- terceiro detalhe: explique como isso afetou a nota e inclua o cuidado de que
  falta de confirmação, sozinha, não prova que uma notícia seja falsa.

Regras obrigatórias:
- Preserve exatamente todos os algarismos que usar e a classificação calculada.
- Não invente ligação entre duas informações da ficha.
- Não diga que algo foi confirmado se o status recebido indicar diferença,
  resultado misto ou ausência de confirmação.
- Não cite nomes de sites, veículos ou fontes.
- Não use "apoio externo", "evidência externa", "cobertura", "comparações",
  "claim", "pipeline", "stance", "score", "modelo", "classificador",
  "BERTimbau", "similaridade", "polaridade" ou "base de dados".
- Não escreva "a HÍBRIA encontrou", "a HÍBRIA classificou" ou "a HÍBRIA
  atribuiu". Use "as verificações" ou uma construção impessoal.
- Não repita a mesma ideia ou a mesma frase.
- Não copie o título completo da notícia.
- Não use Markdown, numeração ou marcadores nos textos.
- Termine a explicação e cada detalhe com pontuação.

Retorne somente o JSON do schema. A explicação deve ter no máximo 300
caracteres e cada um dos três detalhes, no máximo 260 caracteres.
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
        """Monta a ficha factual mínima que o Qwen pode apenas redigir."""
        breakdown = getattr(result, "score_breakdown", None) or {}
        stance = breakdown.get("stance_stats") or {}
        counts = {
            "confirmacoes": int(stance.get("support", 0) or 0),
            "diferencas": int(stance.get("contradict", 0) or 0),
            "sem_confirmacao_direta": int(stance.get("neutral", 0) or 0)
            + int(stance.get("insufficient", 0) or 0),
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

        if not points:
            return []

        # Mantém a primeira informação (normalmente a mais relevante) e tenta
        # incluir um resultado diferente para explicar situações mistas.
        selected = [points[0]]
        first_status = points[0]["tipo_resultado"]
        different = next(
            (
                point
                for point in points[1:]
                if point["tipo_resultado"] != first_status
            ),
            None,
        )
        if different is not None:
            selected.append(different)

        for point in points[1:]:
            if len(selected) >= 3:
                break
            if point not in selected:
                selected.append(point)

        return selected[:3]

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

        points = list(safe_context.get("verificacoes_relevantes") or [])
        if points:
            point = points[0]
            information = cls._truncate_text(
                point.get("informacao_da_noticia"),
                145,
            ).rstrip("… .!?;:")
            status = str(point.get("resultado_calculado") or "")
            first_detail = cls._limit_output_text(
                f'Ao verificar “{information}”, essa informação {status}.',
                260,
            )
        elif supports or contradictions:
            first_detail = (
                "As verificações apresentaram confirmações e diferenças nas "
                "informações da notícia."
            )
        else:
            first_detail = (
                "Não foi possível confirmar diretamente as informações "
                "principais da notícia."
            )

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

        reach = str(
            safe_context.get("quanto_do_conteudo_importante_foi_verificado")
            or "não foi possível medir quanto do conteúdo pôde ser verificado"
        )
        if count_parts:
            second_detail = (
                f"No conjunto, {', '.join(count_parts)}; {reach}."
            )
        else:
            second_detail = reach[:1].upper() + reach[1:] + "."

        label = str(getattr(result, "label_final", None) or "").casefold()
        if label in {"evidência insuficiente", "não verificado"}:
            third_detail = (
                "A falta de confirmação para o restante reduziu a nota. Isso "
                "não significa, por si só, que a notícia seja falsa."
            )
        elif label == "não confiável":
            third_detail = (
                "As diferenças encontradas reduziram a nota. É recomendável "
                "conferir o conteúdo em outras fontes."
            )
        elif label == "confiável":
            third_detail = (
                "O conjunto de confirmações elevou a nota, sem garantir que "
                "todas as informações estejam corretas."
            )
        else:
            third_detail = (
                "Como apenas parte do conteúdo pôde ser confirmada, a nota "
                "permaneceu intermediária."
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
        breakdown = getattr(result, "score_breakdown", None) or {}
        stance = breakdown.get("stance_stats") or {}
        supports = int(stance.get("support", 0) or 0)
        contradictions = int(stance.get("contradict", 0) or 0)
        coverage = cls._coverage_description(breakdown.get("coverage_score"))
        label = str(getattr(result, "label_final", None) or "não verificado")

        points = list(safe_context.get("verificacoes_relevantes") or [])
        if points:
            point = points[0]
            information = cls._truncate_text(
                point.get("informacao_da_noticia"),
                125,
            ).rstrip("… .!?;:")
            status = str(point.get("resultado_calculado") or "")
            finding = f'Ao verificar “{information}”, a informação {status}'
        elif supports and contradictions:
            finding = (
                "Algumas informações foram confirmadas, enquanto outras não "
                "coincidiram com o que foi encontrado"
            )
        elif contradictions:
            finding = (
                "Algumas informações não coincidiram com o que foi encontrado"
            )
        elif supports:
            finding = "As informações verificadas receberam confirmação"
        else:
            finding = "Não foi possível confirmar diretamente as informações principais"

        if coverage == "ampla":
            reach = "a maior parte da notícia pôde ser verificada"
        elif coverage == "parcial":
            reach = "apenas parte da notícia pôde ser verificada"
        elif coverage == "baixa":
            reach = "poucas partes da notícia puderam ser verificadas"
        else:
            reach = "não houve informações suficientes para verificar a notícia"

        if label.casefold() in {"evidência insuficiente", "não verificado"}:
            consequence = (
                "o conjunto das verificações não foi suficiente para sustentar "
                "todo o conteúdo"
            )
        elif label.casefold() == "não confiável":
            consequence = "as diferenças encontradas reduziram a nota"
        elif label.casefold() == "confiável":
            consequence = "o conjunto das verificações sustentou o resultado"
        else:
            consequence = (
                "o conjunto das verificações sustentou apenas parte do conteúdo"
            )

        explanation = f"{finding}. Como {reach}, {consequence}."

        return cls._limit_output_text(explanation, 300)

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
        subject = str(context.get("assunto_da_noticia") or "")
        if points:
            factual_overlap = max(
                cls._text_overlap(
                    text,
                    str(point.get("informacao_da_noticia") or ""),
                )
                for point in points
            )
            if factual_overlap < 0.16 and cls._text_overlap(text, subject) < 0.20:
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

        points = list(context.get("verificacoes_relevantes") or [])
        if points and max(
            cls._text_overlap(
                normalized[0],
                str(point.get("informacao_da_noticia") or ""),
            )
            for point in points
        ) < 0.16:
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
        return True

    @classmethod
    def _numbers_are_allowed(
        cls,
        value: Any,
        context: dict[str, Any],
    ) -> bool:
        allowed = set(
            cls._number_tokens(
                json.dumps(context, ensure_ascii=False)
            )
        )
        return set(cls._number_tokens(value)).issubset(allowed)

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

Redija a explicação usando somente a ficha abaixo. A classificação já aparece
na interface: não a repita para preencher espaço. Use uma informação concreta
em vez de frases vagas como "o resultado foi calculado".

O segundo detalhe deve manter em algarismos todas as contagens maiores que zero
do campo "resumo_numerico". Não use marcadores ou numeração. Retorne somente o
objeto JSON solicitado.

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
