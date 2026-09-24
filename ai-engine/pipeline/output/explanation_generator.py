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
    DEFAULT_MODEL = "qwen3:0.6b"
    DEFAULT_TIMEOUT = 120
    MAX_INPUT_CHARS = 4000
    RELATION_LABELS = {
        "support": "há apoio externo para esta informação",
        "contradict": "foi encontrada informação divergente",
        "neutral": "há apenas contexto, sem confirmação direta",
        "insufficient": "não houve confirmação externa suficiente",
    }

    OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                "maxLength": 360,
                "description": (
                    "Explicação objetiva do resultado em português do Brasil, "
                    "com uma ou duas frases completas e no máximo 300 caracteres."
                ),
            },
            "details": {
                "type": "array",
                "items": {
                    "type": "string",
                    "maxLength": 280,
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

            classification_details = cls._deterministic_details(result)
            report = cls._parse_report(raw_content)

            if report is None:
                logger.warning(
                    "[explanation_generator] Qwen retornou saída inválida. "
                    f"Conteúdo recebido: {str(raw_content)[:1500]}"
                )
                return None

            used_deterministic_text = False

            # Modelos pequenos às vezes criam uma oposição inexistente ou
            # devolvem termos internos. A substituição é registrada como
            # híbrida, para não identificar como Qwen um texto determinístico.
            if not cls._is_plain_explanation(report["explanation"]):
                report["explanation"] = cls._deterministic_explanation(result)
                used_deterministic_text = True

            if not cls._are_plain_details(
                report["details"],
                explanation=report["explanation"],
            ):
                report["details"] = classification_details
                used_deterministic_text = True

            report["source"] = (
                "hybrid" if used_deterministic_text else "qwen"
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

Você redige a explicação final de uma verificação de notícia.
O resultado já foi calculado. Apenas explique por que ele ocorreu para uma
pessoa sem conhecimento técnico.

Use exclusivamente os dados recebidos. Não pesquise, não invente e não altere
a classificação.

Escreva assim:
- Comece pelo assunto concreto da notícia, usando o título e as informações
  verificáveis para o leitor reconhecer o que foi analisado.
- Diga com clareza o que recebeu confirmação, o que apresentou diferença ou o
  que não pôde ser verificado.
- Explique como o conjunto dessas verificações levou à classificação final.
- Quando houver confirmações e pouca abrangência, deixe claro que algumas
  partes foram confirmadas, mas poucas partes do conteúdo completo puderam ser
  verificadas. Isso não é contradição.

Regras:
- Use português do Brasil, frases simples e completas.
- Não escreva “a HÍBRIA encontrou”, “a HÍBRIA classificou” ou “a HÍBRIA
  atribuiu”. Prefira construções impessoais como “foi possível verificar”.
- Não use termos internos: claim, primeira afirmação, segunda afirmação,
  pipeline, modelo, classificador, BERTimbau, stance, score, peso, polaridade,
  similaridade, cobertura, sobreposição ou base de dados.
- Não transforme falta de confirmação em falsidade. “Evidência insuficiente”
  significa que não foi possível verificar uma parte suficiente do conteúdo.
- Não atribua o veredito a um site. Evite nomes de veículos e fontes.
- Não apresente percentuais ou notas internas. Contagens de confirmações e
  divergências podem ser usadas nos detalhes quando ajudarem a explicar.
- Não conte curiosidades descobertas na pesquisa. Fale somente dos motivos da
  classificação.
- Não repita a mesma frase na explicação e nos detalhes.
- Não use Markdown, listas numeradas ou marcadores no texto dos detalhes.
- Termine todas as frases com pontuação.

Retorne somente o JSON do schema:
- “explanation”: uma ou duas frases, no máximo 300 caracteres.
- “details”: exatamente três frases diferentes, no máximo 260 caracteres cada.

Os detalhes devem cumprir funções diferentes:
1. explicar o equilíbrio entre confirmações, diferenças e ausências;
2. explicar quanto do conteúdo importante pôde ser verificado;
3. explicar por que isso resultou na classificação final.
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
        for claim in (claims or [])[:5]:
            text = getattr(claim, "text", None)
            if text:
                claim_texts.append(cls._truncate_text(text, 240))

        evidence_pairs = cls._build_evidence_pairs(
            retrieval_results or [],
            stance_results or [],
        )

        coverage_description = cls._coverage_description(
            (breakdown or {}).get("coverage_score")
            if isinstance(breakdown, dict)
            else None
        )

        result_factors: dict[str, str] = {
            "quanto_do_conteudo_foi_verificado": coverage_description,
        }
        if isinstance(breakdown, dict):
            result_factors.update(
                {
                    "forca_das_confirmacoes": cls._score_description(
                        breakdown.get("evidence_score")
                    ),
                    "resumo_das_verificacoes": cls._stance_description(
                        breakdown.get("stance_stats")
                    ),
                    "sinal_complementar_do_texto": (
                        cls._text_signal_description(
                            breakdown.get("bertimbau_score")
                        )
                    ),
                }
            )

        reputation_summary: dict[str, Any] = {}

        if isinstance(reputation, dict):
            for key in (
                "status",
                "source_name",
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
            "fatores_que_formaram_o_resultado": result_factors,
            "reputacao_fonte": reputation_summary,
            "informacoes_verificaveis": claim_texts,
            "verificacoes_realizadas": evidence_pairs,
        }

        prompt = cls._render_prompt(context)

        # Evita cortar JSON no meio.
        # Se ficar grande, reduz progressivamente as amostras.
        if len(prompt) > cls.MAX_INPUT_CHARS:
            context["verificacoes_realizadas"] = evidence_pairs[:2]
            context["informacoes_verificaveis"] = claim_texts[:3]

            prompt = cls._render_prompt(context)

        if len(prompt) > cls.MAX_INPUT_CHARS:
            for pair in context["verificacoes_realizadas"]:
                pair["conteudo_consultado"] = cls._truncate_text(
                    pair.get("conteudo_consultado"),
                    180,
                )
            context["titulo"] = cls._truncate_text(
                context.get("titulo"),
                180,
            )

            prompt = cls._render_prompt(context)

        if len(prompt) > cls.MAX_INPUT_CHARS:
            context["verificacoes_realizadas"] = evidence_pairs[:1]
            context["informacoes_verificaveis"] = claim_texts[:2]

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

    @classmethod
    def _build_evidence_pairs(
        cls,
        retrieval_results: list[Any],
        stance_results: list[Any],
    ) -> list[dict[str, Any]]:
        """Liga cada claim à sua melhor evidência e ao stance correspondente."""
        stances_by_id: dict[str, dict[str, Any]] = {}
        stances_by_claim: dict[str, list[dict[str, Any]]] = {}

        for raw_item in stance_results:
            item = raw_item.to_dict() if hasattr(raw_item, "to_dict") else raw_item
            if not isinstance(item, dict):
                continue

            evidence_id = str(item.get("evidence_id") or "")
            claim_id = str(item.get("claim_id") or "")
            if evidence_id:
                stances_by_id[evidence_id] = item
            if claim_id:
                stances_by_claim.setdefault(claim_id, []).append(item)

        selected: list[dict[str, Any]] = []
        relation_priority = {
            "support": 4,
            "contradict": 4,
            "neutral": 2,
            "insufficient": 1,
            None: 0,
        }

        for retrieval in retrieval_results:
            claim = getattr(retrieval, "claim", None)
            claim_id = str(getattr(claim, "claim_id", "") or "")
            claim_text = cls._truncate_text(
                getattr(claim, "text", ""),
                260,
            )
            candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

            for evidence in list(getattr(retrieval, "evidences", []) or []):
                evidence_id = str(getattr(evidence, "evidence_id", "") or "")
                stance_item = stances_by_id.get(evidence_id)

                if stance_item is None:
                    source = str(getattr(evidence, "source", "") or "")
                    url = str(getattr(evidence, "url", "") or "")
                    stance_item = next(
                        (
                            item
                            for item in stances_by_claim.get(claim_id, [])
                            if (
                                (url and str(item.get("url") or "") == url)
                                or (
                                    source
                                    and str(item.get("source") or "") == source
                                )
                            )
                        ),
                        {},
                    )

                relation = (
                    stance_item.get("stance")
                    or getattr(evidence, "stance", None)
                )
                confidence = float(stance_item.get("confidence") or 0.0)
                similarity = float(getattr(evidence, "similarity", 0.0) or 0.0)
                trusted = bool(getattr(evidence, "trusted_source", False))

                pair = {
                    "informacao_da_noticia": claim_text,
                    "referencia_externa_opcional": cls._truncate_text(
                        getattr(evidence, "source", ""),
                        100,
                    ),
                    "titulo_da_referencia": cls._truncate_text(
                        getattr(evidence, "title", ""),
                        180,
                    ),
                    "resultado_da_verificacao": cls.RELATION_LABELS.get(
                        relation,
                        "relação não determinada",
                    ),
                    "conteudo_consultado": cls._truncate_text(
                        getattr(evidence, "text", ""),
                        340,
                    ),
                }
                rank = (
                    relation_priority.get(relation, 0),
                    trusted,
                    confidence,
                    similarity,
                )
                candidates.append((rank, pair))

            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                selected.append(candidates[0][1])

            if len(selected) >= 3:
                break

        # Disponibiliza no máximo uma referência nominal ao redator. As demais
        # comparações continuam completas, mas são explicadas sem atribuir o
        # veredito da HÍBRIA a veículos externos.
        for pair in selected[1:]:
            pair.pop("referencia_externa_opcional", None)
            pair.pop("titulo_da_referencia", None)

        return selected

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
        text = " ".join(str(value or "").casefold().split())
        if not text:
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
    def _score_description(value: Any) -> str:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return "não disponível"

        if score > 1.0:
            score /= 100.0
        if score >= 0.75:
            return "forte"
        if score >= 0.50:
            return "moderada"
        if score > 0:
            return "fraca"
        return "nenhuma"

    @staticmethod
    def _text_signal_description(value: Any) -> str:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return "não disponível"

        if score > 1.0:
            score /= 100.0
        if score >= 0.70:
            return "favorável"
        if score >= 0.45:
            return "inconclusivo"
        return "desfavorável"

    @staticmethod
    def _stance_description(value: Any) -> str:
        if not isinstance(value, dict):
            return "não disponível"

        supports = int(value.get("support", 0) or 0)
        contradictions = int(value.get("contradict", 0) or 0)
        neutral = int(value.get("neutral", 0) or 0)

        if contradictions and supports:
            return "foram encontrados apoios e informações divergentes"
        if contradictions:
            return "foram encontradas informações divergentes"
        if supports:
            return "foram encontrados apoios, sem divergências identificadas"
        if neutral:
            return "houve apenas contexto, sem confirmação direta"
        return "não houve comparações conclusivas"

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

Explique o resultado abaixo para uma pessoa comum.

Na explicação, identifique o assunto concreto da notícia e resuma por que a
classificação foi atribuída. Nos detalhes, apresente três motivos diferentes:
o resultado das verificações, quanto do conteúdo pôde ser verificado e como
isso afetou a classificação.

Não copie os nomes dos campos. Não use marcadores, numeração, nomes de sites
ou linguagem técnica. Retorne somente o objeto JSON solicitado.

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
