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

    OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                # O limite do schema é deliberadamente maior que o limite de
                # exibição. Alguns modelos completam o campo até maxLength e
                # cortavam a última palavra. O parser reduz o texto somente
                # depois, preservando uma frase completa.
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
        return {
            "explanation": cls._deterministic_explanation(result),
            "details": cls._deterministic_details(result),
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
            # A HÍBRIA calcula primeiro uma explicação segura e factual. O
            # modelo local recebe somente esse texto para revisar a linguagem;
            # ele não volta a interpretar claims, fontes ou evidências.
            base_report = cls.fallback_report(result)
            prompt = cls._build_prompt(
                result,
                base_report=base_report,
            )

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
                    **base_report,
                    "source": "deterministic",
                }

            label = str(getattr(result, "label_final", None) or "")
            explanation_accepted = cls._is_safe_rewrite(
                report["explanation"],
                reference=base_report["explanation"],
                label=label,
                require_label=True,
            )
            details_accepted = cls._are_safe_rewritten_details(
                report["details"],
                reference=base_report["details"],
                explanation=(
                    report["explanation"]
                    if explanation_accepted
                    else base_report["explanation"]
                ),
                label=label,
            )

            if not explanation_accepted:
                report["explanation"] = base_report["explanation"]
            if not details_accepted:
                report["details"] = base_report["details"]

            if explanation_accepted and details_accepted:
                report["source"] = "qwen"
            elif explanation_accepted or details_accepted:
                report["source"] = "hybrid"
            else:
                report["source"] = "deterministic"

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

Você é um revisor de linguagem. Receberá uma explicação-base já calculada e
validada pela HÍBRIA. Sua única função é deixá-la mais natural e fácil para uma
pessoa sem conhecimento técnico.

Regras obrigatórias:
- Preserve exatamente a classificação final e todos os números recebidos.
- Preserve o sentido de cada frase. Não acrescente fatos, interpretações,
  nomes, fontes, pesquisas, datas, resultados ou conclusões.
- Não transforme “evidência insuficiente” em “confiável”, “não confiável” ou
  qualquer outra classificação.
- Não use conhecimento externo e não volte a analisar a notícia.
- Não use “apoio externo”, “claim”, “pipeline”, “stance”, “score”, “cobertura”,
  “modelo”, “classificador”, “BERTimbau”, “similaridade” ou “base de dados”.
- Não escreva “a HÍBRIA encontrou” ou “a HÍBRIA classificou”.
- Não mencione sites, veículos, consultorias ou fontes.
- Mantenha os três detalhes na mesma ordem do texto-base.
- Não repita ideias. Termine todas as frases com pontuação.
- Se não conseguir melhorar uma frase sem mudar seu sentido, copie a frase-base.

Retorne somente o JSON solicitado:
- “explanation”: uma ou duas frases, no máximo 300 caracteres.
- “details”: exatamente três frases diferentes, no máximo 260 caracteres cada.
""".strip()

    @classmethod
    def _build_prompt(
        cls,
        result: Any,
        *,
        base_report: dict[str, Any] | None = None,
    ) -> str:
        base = base_report or cls.fallback_report(result)
        context = {
            "classificacao_final_obrigatoria": str(
                getattr(result, "label_final", None) or "não verificado"
            ),
            "texto_base_validado": {
                "explanation": str(base.get("explanation") or ""),
                "details": [str(item) for item in base.get("details") or []],
            },
        }

        prompt = cls._render_prompt(context)

        logger.debug(
            "[explanation_generator] prompt final com %s caracteres",
            len(prompt),
        )

        return prompt

    @classmethod
    def _deterministic_details(cls, result: Any) -> list[str]:
        """Explica os três fatores do resultado em linguagem cotidiana."""
        breakdown = getattr(result, "score_breakdown", None) or {}
        stance = breakdown.get("stance_stats") or {}
        supports = int(stance.get("support", 0) or 0)
        contradictions = int(stance.get("contradict", 0) or 0)
        neutral = int(stance.get("neutral", 0) or 0)

        if supports and contradictions:
            comparison_text = (
                f"Entre os trechos verificados, {supports} "
                f"{cls._plural(supports, 'recebeu', 'receberam')} confirmação e "
                f"{contradictions} "
                f"{cls._plural(contradictions, 'apresentou', 'apresentaram')} "
                "informações diferentes das referências consultadas."
            )
        elif contradictions:
            comparison_text = (
                f"Entre os trechos verificados, {contradictions} "
                f"{cls._plural(contradictions, 'apresentou', 'apresentaram')} "
                "informações diferentes das referências consultadas."
            )
        elif supports:
            comparison_text = (
                f"Entre os trechos verificados, {supports} "
                f"{cls._plural(supports, 'recebeu', 'receberam')} confirmação, "
                "sem diferenças importantes nas referências consultadas."
            )
        elif neutral:
            comparison_text = (
                f"Foram encontrados {neutral} "
                f"{cls._plural(neutral, 'texto relacionado', 'textos relacionados')} "
                "ao assunto, sem confirmação direta das informações principais."
            )
        else:
            comparison_text = (
                "Não foi possível confirmar nem contestar as principais "
                "informações da notícia."
            )

        coverage = cls._coverage_description(breakdown.get("coverage_score"))
        if coverage == "ampla":
            coverage_text = (
                "Foi possível verificar a maior parte das informações "
                "importantes da notícia."
            )
        elif coverage == "parcial":
            coverage_text = (
                "Foi possível verificar apenas parte das informações "
                "importantes da notícia."
            )
        elif coverage == "baixa":
            coverage_text = (
                "Apesar das verificações realizadas, apenas uma pequena parte "
                "das informações importantes pôde ser confirmada."
            )
        else:
            coverage_text = (
                "Não houve informações suficientes para verificar os pontos "
                "principais da notícia."
            )

        label = str(getattr(result, "label_final", None) or "").casefold()
        if label in {"evidência insuficiente", "não verificado"}:
            meaning_text = (
                "A falta de confirmação para o restante do conteúdo reduziu a "
                f'nota e levou ao resultado "{label}"; isso não indica, por si '
                "só, que a notícia seja falsa."
            )
        elif label == "não confiável":
            meaning_text = (
                'As diferenças encontradas reduziram a nota e levaram ao resultado '
                '"não confiável". É recomendável conferir o conteúdo em outras fontes.'
            )
        elif label == "confiável":
            meaning_text = (
                'A quantidade de confirmações elevou a nota e levou ao resultado '
                '"confiável". Isso não é uma garantia absoluta de que tudo esteja correto.'
            )
        else:
            meaning_text = (
                f'Como apenas parte do conteúdo pôde ser confirmada, o resultado '
                f'foi "{label or "parcialmente confiável"}".'
            )

        return [comparison_text, coverage_text, meaning_text]

    @staticmethod
    def _plural(amount: int, singular: str, plural: str) -> str:
        return singular if amount == 1 else plural

    @classmethod
    def _deterministic_explanation(cls, result: Any) -> str:
        """Resume a decisão sem vocabulário técnico ou oposição falsa."""
        breakdown = getattr(result, "score_breakdown", None) or {}
        stance = breakdown.get("stance_stats") or {}
        supports = int(stance.get("support", 0) or 0)
        contradictions = int(stance.get("contradict", 0) or 0)
        coverage = cls._coverage_description(breakdown.get("coverage_score"))
        label = str(getattr(result, "label_final", None) or "não verificado")

        title = cls._truncate_text(
            getattr(result, "title", "") or "",
            105,
        ).rstrip("…")
        prefix = f'Na notícia “{title}”, ' if title else "Na notícia, "

        if supports and contradictions:
            finding = (
                "algumas informações foram confirmadas, enquanto outras não "
                "coincidiram com o que foi encontrado"
            )
        elif contradictions:
            finding = (
                "algumas informações não coincidiram com o que foi encontrado"
            )
        elif supports:
            finding = "as informações verificadas receberam confirmação"
        else:
            finding = "não foi possível confirmar diretamente as informações principais"

        if coverage == "ampla":
            reach = "a maior parte da notícia pôde ser verificada"
        elif coverage == "parcial":
            reach = "apenas parte da notícia pôde ser verificada"
        elif coverage == "baixa":
            reach = "poucas partes da notícia puderam ser verificadas"
        else:
            reach = "não houve informações suficientes para verificar a notícia"

        explanation = (
            f"{prefix}{finding}. Como {reach}, o resultado foi \"{label}\"."
        )
        if label.casefold() in {"evidência insuficiente", "não verificado"}:
            explanation += " Isso não significa que a notícia seja falsa."

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
            "apoios externos",
            "evidências externas",
            "claim",
            "primeira afirmação",
            "segunda afirmação",
            "terceira afirmação",
            "a análise encontrou que",
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
    def _is_safe_rewrite(
        cls,
        value: Any,
        *,
        reference: str,
        label: str,
        require_label: bool,
    ) -> bool:
        """Confirma que a revisão preservou os dados do texto-base."""
        text = " ".join(str(value or "").split())
        if not cls._is_plain_explanation(text):
            return False
        if cls._number_tokens(text) != cls._number_tokens(reference):
            return False
        if require_label and not cls._contains_label(text, label):
            return False
        if cls._has_conflicting_label(text, label):
            return False
        if cls._text_overlap(text, reference) < 0.25:
            return False
        return True

    @classmethod
    def _are_safe_rewritten_details(
        cls,
        details: Any,
        *,
        reference: list[str],
        explanation: str,
        label: str,
    ) -> bool:
        """Valida os três detalhes reescritos contra os tópicos seguros."""
        if not cls._are_plain_details(details, explanation=explanation):
            return False

        normalized = [" ".join(str(item or "").split()) for item in details]
        if len(reference) != 3:
            return False
        if cls._number_tokens(" ".join(normalized)) != cls._number_tokens(
            " ".join(reference)
        ):
            return False
        if not cls._contains_label(" ".join(normalized), label):
            return False
        if cls._has_conflicting_label(" ".join(normalized), label):
            return False

        return all(
            cls._text_overlap(candidate, original) >= 0.25
            for candidate, original in zip(normalized, reference)
        )

    @staticmethod
    def _number_tokens(value: Any) -> list[str]:
        """Extrai números para impedir alterações silenciosas pelo modelo."""
        return sorted(
            re.findall(
                r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)",
                str(value or ""),
            )
        )

    @staticmethod
    def _contains_label(value: Any, label: str) -> bool:
        expected = " ".join(str(label or "").casefold().split())
        text = " ".join(str(value or "").casefold().split())
        return bool(expected) and expected in text

    @classmethod
    def _has_conflicting_label(cls, value: Any, label: str) -> bool:
        """Rejeita uma segunda classificação diferente da calculada."""
        expected = " ".join(str(label or "").casefold().split())
        text = " ".join(str(value or "").casefold().split())
        if expected:
            text = text.replace(expected, " ")

        known_labels = (
            "parcialmente confiável",
            "evidência insuficiente",
            "não verificado",
            "não confiável",
            "confiável",
        )
        return any(
            candidate != expected
            and re.search(
                rf"(?<!\w){re.escape(candidate)}(?!\w)",
                text,
            )
            for candidate in known_labels
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

Reescreva somente o texto-base validado para deixá-lo mais natural.
Não acrescente nem retire informações. Preserve literalmente todos os números
e a classificação final. Retorne somente o objeto JSON solicitado.

DADOS IMUTÁVEIS:
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
