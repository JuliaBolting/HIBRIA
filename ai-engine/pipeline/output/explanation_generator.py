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
            report = cls._parse_report(
                raw_content,
                fallback_details=classification_details,
            )

            if report is None:
                logger.warning(
                    "[explanation_generator] Qwen retornou saída inválida. "
                    f"Conteúdo recebido: {str(raw_content)[:1500]}"
                )
                return None

            # Os detalhes explicam somente os fatores da classificação. O Qwen
            # redige a explicação curta, mas não transforma os tópicos em um
            # resumo das informações encontradas durante a busca.
            report["details"] = classification_details

            # Modelos pequenos às vezes criam uma oposição inexistente, como
            # "a informação foi encontrada, mas há apoio", ou devolvem termos
            # internos que não ajudam uma pessoa leiga. Nesses casos, preserva
            # o resultado da análise e usa uma explicação simples e estável.
            if not cls._is_plain_explanation(report["explanation"]):
                report["explanation"] = cls._deterministic_explanation(result)

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
- Evidência insuficiente não torna uma afirmação falsa ou inválida; significa
  apenas que o sistema não encontrou confirmação externa bastante.
- Só diga que há contradição quando as evidências fornecidas realmente
  indicarem contradição.
- Se um par indicar apoio externo, diga que aquela informação recebeu apoio.
  Não chame essa evidência isolada de insuficiente apenas porque a classificação
  global é "evidência insuficiente".
- Quando a informação principal tiver apoio externo, nunca diga depois que ela
  ficou sem confirmação. Nesse caso, a cobertura baixa se refere às demais
  informações verificáveis da notícia.
- Explique por que a classificação apresentada foi atribuída. O resultado
  pertence ao sistema; nenhuma fonte externa deve aparecer como autora do
  veredito.
- Na resposta ao usuário, use frases diretas como "a informação recebeu
  confirmação", "foi possível verificar" e "o resultado foi". Nunca escreva
  "a HÍBRIA encontrou", "a HÍBRIA classificou" ou "a HÍBRIA atribuiu".
- O leitor não conhece as claims internas. Nunca escreva "claim", "primeira
  afirmação", "segunda afirmação" ou "terceira afirmação". Reescreva o fato
  específico em linguagem comum, por exemplo: "A escalação do ator recebeu
  confirmação externa".
- Explique primeiro o que foi encontrado sobre a informação principal da
  notícia, depois o que ficou sem confirmação e como isso afetou o resultado.
- Normalmente não cite nomes de sites ou veículos. Quando for realmente
  necessário para compreender um fato concreto, cite no máximo uma referência
  em toda a resposta e trate-a apenas como material consultado.
- Não diga que uma fonte provou que a notícia é verdadeira ou falsa. Em caso de
  divergência, diga que a análise encontrou informações divergentes.
- Não use as expressões "polaridade", "sobreposição", "similaridade",
  "base de dados", "componentes" ou "a conclusão não é válida".
- Não escreva frases vagas como "a evidência é relevante" ou "há dados
  suficientes" sem dizer qual informação foi encontrada.
- Escreva para uma pessoa comum, sem linguagem técnica.
- Não use as palavras "cobertura", "comparações", "apoio externo" ou
  "evidências externas" na resposta. Prefira "foi possível verificar",
  "recebeu confirmação" e "não coincidiu com o que foi encontrado".
- Não escreva construções contraditórias como "a informação foi encontrada,
  mas há apoio". Confirmação não é oposição. Use "a informação recebeu
  confirmação; porém, outras partes não puderam ser verificadas".
- Não use Markdown.
- Não explique seu raciocínio.
- Responda em português do Brasil.
- Termine a explicação e cada detalhe com uma frase completa e pontuação final.
- Nunca corte uma palavra nem termine com abreviação causada por corte de texto.
- Não numere os detalhes e não coloque marcadores como "1.", "2.", "3." ou
  hífens. A interface adicionará os marcadores automaticamente.
- Os detalhes devem explicar somente os motivos da classificação. Não use os
  detalhes para contar assuntos, pessoas, datas ou curiosidades descobertas
  durante a pesquisa.

A saída deve conter:
- "explanation": uma ou duas frases completas, com no máximo 300 caracteres.
- "details": exatamente três frases concretas, cada uma com no máximo 260
  caracteres.

Cada detalhe deve abordar uma informação ou um fator diferente. Não repita nos
detalhes exatamente a mesma informação da explicação.
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
            "cobertura_das_informacoes": coverage_description,
        }
        if isinstance(breakdown, dict):
            result_factors.update(
                {
                    "forca_da_confirmacao_externa": cls._score_description(
                        breakdown.get("evidence_score")
                    ),
                    "resultado_das_comparacoes": cls._stance_description(
                        breakdown.get("stance_stats")
                    ),
                    "sinal_auxiliar_da_analise_textual": (
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
            "comparacoes_com_evidencias": evidence_pairs,
        }

        prompt = cls._render_prompt(context)

        # Evita cortar JSON no meio.
        # Se ficar grande, reduz progressivamente as amostras.
        if len(prompt) > cls.MAX_INPUT_CHARS:
            context["comparacoes_com_evidencias"] = evidence_pairs[:2]
            context["informacoes_verificaveis"] = claim_texts[:3]

            prompt = cls._render_prompt(context)

        if len(prompt) > cls.MAX_INPUT_CHARS:
            for pair in context["comparacoes_com_evidencias"]:
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
            context["comparacoes_com_evidencias"] = evidence_pairs[:1]
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
                    "resultado_encontrado_pela_hibria": cls.RELATION_LABELS.get(
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
                f"Nas verificações feitas, houve {supports} "
                f"{cls._plural(supports, 'confirmação', 'confirmações')} e "
                f"{contradictions} "
                f"{cls._plural(contradictions, 'resultado que não coincidiu', 'resultados que não coincidiram')} "
                "com a notícia."
            )
        elif contradictions:
            comparison_text = (
                f"Nas verificações feitas, {contradictions} "
                f"{cls._plural(contradictions, 'resultado não coincidiu', 'resultados não coincidiram')} "
                "com a notícia."
            )
        elif supports:
            comparison_text = (
                f"Nas verificações feitas, houve {supports} "
                f"{cls._plural(supports, 'confirmação', 'confirmações')} e nenhuma "
                "diferença importante."
            )
        elif neutral:
            comparison_text = (
                f"Nas verificações feitas, {neutral} "
                f"{cls._plural(neutral, 'resultado ajudou', 'resultados ajudaram')} "
                "a entender o assunto, mas "
                f"{cls._plural(neutral, 'não confirmou', 'não confirmaram')} "
                "diretamente a notícia."
            )
        else:
            comparison_text = (
                "Não foi possível confirmar nem contestar as principais "
                "informações da notícia."
            )

        coverage = cls._coverage_description(breakdown.get("coverage_score"))
        if coverage == "ampla":
            coverage_text = (
                "A maior parte das informações importantes da notícia pôde "
                "ser verificada."
            )
        elif coverage == "parcial":
            coverage_text = (
                "Apenas parte das informações importantes da notícia pôde "
                "ser verificada."
            )
        elif coverage == "baixa":
            coverage_text = (
                "Poucas informações importantes da notícia puderam ser "
                "verificadas."
            )
        else:
            coverage_text = (
                "Não houve informações suficientes para verificar os pontos "
                "principais da notícia."
            )

        label = str(getattr(result, "label_final", None) or "").casefold()
        if label in {"evidência insuficiente", "não verificado"}:
            meaning_text = (
                f'Por isso, a nota ficou mais baixa e o resultado foi "{label}". '
                "Isso não significa que a notícia seja falsa."
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

        if supports and contradictions:
            finding = (
                "Algumas informações foram confirmadas, mas outras não "
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

        explanation = f"{finding}. Como {reach}, o resultado foi \"{label}\"."
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

Escreva a explicação final da análise usando exclusivamente o contexto abaixo.

Retorne somente o objeto JSON solicitado pelo schema.

A "explanation" deve ter no máximo 300 caracteres e explicar somente o principal
motivo da classificação final. Se a informação principal tiver confirmação,
diga isso uma única vez. Depois explique, quando for o caso, que outras partes
não puderam ser verificadas ou não coincidiram com o que foi encontrado.
Use "fatores_que_formaram_o_resultado" para explicar a decisão global. Use as
comparações somente para dar exemplos concretos.

Os três itens de "details" devem explicar, nesta ordem: o que foi confirmado ou
não coincidiu; quanto da notícia pôde ser verificado; e como isso afetou a nota e
a classificação. Não conte fatos descobertos durante a pesquisa.

Não coloque números, hífens ou bolinhas no início dos detalhes. Não enumere
informações como "primeira afirmação". Não copie nomes de campos nem
justificativas técnicas. Não use "cobertura", "comparações", "apoio externo" ou
"evidências externas". Não cite nomes de sites ou veículos.

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
