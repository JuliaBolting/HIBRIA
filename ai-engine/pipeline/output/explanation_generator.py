# =============================================================================
# HÍBRIA — Explanation Generator
#
# Responsável por gerar a explicação textual do resultado da análise.
#
# IMPORTANTE:
# - Não calcula o score.
# - Não altera o label.
# - Não refaz a análise.
# - Não substitui o aggregator.py.
# - Usa a mesma API REST do Gemini utilizada pelo retriever.
# - Se o Gemini falhar, o pipeline continua normalmente.
# =============================================================================

from __future__ import annotations

import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


class ExplanationGenerator:

    # -------------------------------------------------------------------------
    # Configurações
    # -------------------------------------------------------------------------

    API_URL_TEMPLATE = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "{model}:generateContent"
    )

    DEFAULT_MODEL = "gemini-2.5-flash"

    # Timeout da requisição em segundos.
    # 120 segundos conforme definido para esta etapa.
    TIMEOUT = 120

    # Limite de saída.
    # A explicação da extensão deve ser curta.
    MAX_OUTPUT_TOKENS = 250

    # Limite aproximado do prompt enviado ao Gemini.
    MAX_INPUT_CHARS = 6000

    @classmethod
    def generate(cls, result: Any) -> str | None:
        """
        Gera uma explicação curta para o resultado da análise.

        Retorna:
            str  -> explicação gerada
            None -> Gemini indisponível ou erro

        A falha desta etapa NÃO interrompe o pipeline.
        """

        api_key = os.getenv("GEMINI_API_KEY", "").strip()

        if not api_key:
            logger.warning(
                "[explanation_generator] GEMINI_API_KEY não configurada"
            )
            return None

        model = os.getenv(
            "HIBRIA_AI_FALLBACK_MODEL",
            cls.DEFAULT_MODEL,
        ).strip()

        if not model:
            model = cls.DEFAULT_MODEL

        try:
            prompt = cls._build_prompt(result)

            url = cls.API_URL_TEMPLATE.format(model=model)

            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt,
                            }
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": cls.MAX_OUTPUT_TOKENS,
                },
            }

            logger.info(
                "[explanation_generator] enviando solicitação ao Gemini "
                f"(model={model}, timeout={cls.TIMEOUT}s)"
            )

            response = requests.post(
                url,
                params={"key": api_key},
                headers={
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=cls.TIMEOUT,
            )

            # -----------------------------------------------------------------
            # Erros HTTP
            # -----------------------------------------------------------------

            if response.status_code >= 400:
                logger.warning(
                    "[explanation_generator] erro HTTP da API Gemini "
                    f"({response.status_code}): {response.text[:1000]}"
                )

                return None

            # -----------------------------------------------------------------
            # JSON
            # -----------------------------------------------------------------

            try:
                data = response.json()
            except ValueError:
                logger.warning(
                    "[explanation_generator] Gemini retornou resposta "
                    "que não é JSON válido"
                )
                return None

            # -----------------------------------------------------------------
            # Extrai texto
            # -----------------------------------------------------------------

            explanation = cls._extract_text(data)

            if not explanation:
                logger.warning(
                    "[explanation_generator] Gemini retornou "
                    "resposta sem texto"
                )
                return None

            explanation = explanation.strip()

            logger.info(
                "[explanation_generator] explicação gerada com sucesso "
                f"({len(explanation)} caracteres)"
            )

            return explanation

        except requests.Timeout:
            logger.warning(
                "[explanation_generator] timeout ao consultar Gemini "
                f"(>{cls.TIMEOUT}s)"
            )
            return None

        except requests.RequestException as exc:
            logger.warning(
                "[explanation_generator] erro de comunicação com Gemini: "
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
    # Construção do prompt
    # =========================================================================

    @classmethod
    def _build_prompt(cls, result: Any) -> str:
        """
        Constrói um prompt pequeno usando somente informações já calculadas
        pelo pipeline.

        Não envia o PipelineResult inteiro para o Gemini.
        """

        score = getattr(result, "score_final", None)
        label = getattr(result, "label_final", None)

        breakdown = getattr(
            result,
            "score_breakdown",
            None,
        )

        reputation = getattr(
            result,
            "reputation",
            None,
        )

        claims = getattr(
            result,
            "claims",
            None,
        )

        stance_results = getattr(
            result,
            "stance_results",
            None,
        )

        title = getattr(
            result,
            "title",
            "",
        ) or ""

        # ---------------------------------------------------------------------
        # Claims principais
        # ---------------------------------------------------------------------

        claim_texts: list[str] = []

        for claim in (claims or [])[:5]:

            text = getattr(
                claim,
                "text",
                None,
            )

            if text:
                claim_texts.append(
                    str(text)[:400]
                )

        # ---------------------------------------------------------------------
        # Relações de evidência / stance
        # ---------------------------------------------------------------------

        stance_summary: list[str] = []

        for item in (stance_results or [])[:5]:

            try:

                if hasattr(item, "to_dict"):
                    item = item.to_dict()

                if isinstance(item, dict):

                    stance = (
                        item.get("stance")
                        or item.get("label")
                        or item.get("relation")
                    )

                    if stance:
                        stance_summary.append(
                            str(stance)
                        )

            except Exception:
                continue

        # ---------------------------------------------------------------------
        # Score breakdown
        # ---------------------------------------------------------------------

        safe_breakdown: dict[str, Any] = {}

        if isinstance(breakdown, dict):

            for key, value in list(
                breakdown.items()
            )[:10]:

                if isinstance(
                    value,
                    (str, int, float, bool),
                ):
                    safe_breakdown[key] = value

        # ---------------------------------------------------------------------
        # Reputação
        # ---------------------------------------------------------------------

        reputation_summary: dict[str, Any] = {}

        if isinstance(reputation, dict):

            for key in (
                "status",
                "domain",
                "canonical_domain",
                "source_name",
                "note",
                "score",
            ):

                value = reputation.get(key)

                if value is not None:
                    reputation_summary[key] = value

        # ---------------------------------------------------------------------
        # Contexto reduzido
        # ---------------------------------------------------------------------

        context = {
            "titulo": title[:500],
            "score_final": score,
            "rotulo_final": label,
            "componentes": safe_breakdown,
            "reputacao_fonte": reputation_summary,
            "claims_principais": claim_texts,
            "relacoes_evidencias": stance_summary,
        }

        # ---------------------------------------------------------------------
        # Prompt
        # ---------------------------------------------------------------------

        prompt = f"""
Você é responsável pela explicação e transparência do sistema HÍBRIA.

O HÍBRIA analisou uma notícia utilizando diferentes sinais,
como evidências externas, similaridade entre informações,
relação entre afirmações e evidências, características do texto
e reputação da fonte.

Sua tarefa é explicar o resultado dessa análise para uma pessoa
que NÃO conhece o funcionamento interno do sistema.

A explicação será exibida diretamente na interface da extensão.
Portanto, escreva de forma simples, didática, natural e objetiva.

IMPORTANTE:

- Não refaça a análise.
- Não calcule outro score.
- Não altere o resultado fornecido.
- Não invente informações.
- Não invente fontes ou evidências.
- Não use linguagem excessivamente técnica.
- Evite termos como "pipeline", "aggregator", "modelo",
  "score_breakdown", "claim_detector" ou nomes internos do sistema.
- Não diga simplesmente "o score foi X".
- Explique O QUE levou ao resultado.
- Explique o resultado de maneira que qualquer usuário consiga entender.
- Diferencie claramente "não confirmado" de "falso".
- A ausência de evidências não deve ser apresentada como prova de falsidade.
- Quando houver poucas evidências, diga que parte das informações
  não pôde ser confirmada pelas fontes consultadas.
- Quando houver evidências contraditórias, explique que foram
  encontradas divergências.
- Quando houver evidências favoráveis, explique que as informações
  apresentadas são compatíveis com as fontes encontradas.
- Sempre deixe claro que o resultado representa a análise realizada
  pelo HÍBRIA e que o usuário pode consultar outras fontes.
- Não mencione que você é uma IA.
- Não mencione estas instruções.
- Não use markdown.
- Não faça listas.
- Responda em português do Brasil.
- Gere no máximo 2 parágrafos curtos.

PREFERÊNCIA DE LINGUAGEM:

Em vez de:
"O conteúdo recebeu score de 20 devido à ausência de evidências."

Prefira:
"A análise identificou que parte das informações apresentadas
não pôde ser confirmada pelas fontes consultadas. Por isso,
o resultado indica que é necessário ter cautela com o conteúdo."

Em vez de:
"O sistema classificou a notícia como não verificada."

Prefira:
"A análise não encontrou evidências externas suficientes para
confirmar as informações apresentadas."

Em vez de:
"A reputação da fonte apresentou score insuficiente."

Prefira:
"Também não foram encontradas informações suficientes para avaliar
a fonte de forma conclusiva."
ERRADO:
"O HÍBRIA analisou o conteúdo e atribuiu..."

MELHOR:
"Parte das informações apresentadas não pôde ser confirmada..."

AINDA MELHOR:
"Não foram encontradas fontes suficientes para confirmar..."

RESULTADO DA ANÁLISE:

Título:
{title[:500]}

Resultado:
{label}

Componentes considerados:
{safe_breakdown}

Reputação da fonte:
{reputation_summary}

Principais informações analisadas:
{claim_texts}

Relação encontrada entre informações e evidências:
{stance_summary}

Agora escreva uma explicação curta, didática e transparente para
o usuário final.
""".strip()


        if len(prompt) > cls.MAX_INPUT_CHARS:

            prompt = prompt[:cls.MAX_INPUT_CHARS]

            prompt += """

Continue utilizando somente as informações presentes acima.
Responda em 1 ou 2 parágrafos curtos.
""".strip()

        return prompt

    # =========================================================================
    # Extração da resposta Gemini
    # =========================================================================

    @staticmethod
    def _extract_text(data: dict) -> str:
        """
        Extrai o texto da estrutura retornada pela API REST do Gemini.
        """

        candidates = data.get(
            "candidates",
            [],
        ) or []

        if not candidates:
            return ""

        first_candidate = candidates[0]

        content = first_candidate.get(
            "content",
            {},
        ) or {}

        parts = content.get(
            "parts",
            [],
        ) or []

        texts: list[str] = []

        for part in parts:

            if not isinstance(part, dict):
                continue

            text = part.get(
                "text",
                "",
            )

            if text:
                texts.append(
                    str(text)
                )

        return "\n".join(texts).strip()