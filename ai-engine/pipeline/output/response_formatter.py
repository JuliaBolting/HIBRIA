"""
Formatador da resposta final da HÍBRIA.

Responsabilidade:
- Transformar PipelineResult em um JSON simples e estável.
- Preparar os dados que serão consumidos pela API/extensão.
- Não realizar cálculos de confiabilidade.
- Não alterar score ou label.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """
    Organiza o resultado final da análise para consumo externo.
    """

    @staticmethod
    def format(result) -> dict:
        """
        Cria o JSON final que será utilizado pela API/extensão.
        """

        response = {
            "success": True,

            "analysis": {
                "url": result.url,
                "title": result.title,

                "score": result.score_final,
                "label": result.label_final,

                "score_breakdown": result.score_breakdown,
            },

            "explanation": result.explanation,

            "source": {
                "reputation": result.reputation,
            },

            "evidence": {
                "claim_count": result.claim_count,
                "evidence_count": result.evidence_count,
                "claims": [],
            },

            "transparency": {
                "classification": result.classification,
                "text_features": result.text_features,
                "stance_results": [
                    item.to_dict()
                    if hasattr(item, "to_dict")
                    else item
                    for item in (result.stance_results or [])
                ],
            },

            "metadata": {
                "render_method": result.render_method,
                "paywall_detected": result.paywall_detected,
                "warnings": result.warnings,
                "processing_time": result._processing_time,
            },
        }

        # ---------------------------------------------------------------------
        # Claims + evidências
        # ---------------------------------------------------------------------

        for retrieval in result.retrieval_results or []:

            claim = getattr(
                retrieval,
                "claim",
                None,
            )

            claim_data = {
                "claim_id": getattr(
                    claim,
                    "claim_id",
                    None,
                ),
                "text": getattr(
                    claim,
                    "text",
                    "",
                ),
                "evidences": [],
            }

            for evidence in getattr(
                retrieval,
                "evidences",
                [],
            ):

                claim_data["evidences"].append(
                    {
                        "text": getattr(
                            evidence,
                            "text",
                            "",
                        ),
                        "source": getattr(
                            evidence,
                            "source",
                            "",
                        ),
                        "title": getattr(
                            evidence,
                            "title",
                            "",
                        ),
                        "url": getattr(
                            evidence,
                            "url",
                            "",
                        ),
                        "domain": getattr(
                            evidence,
                            "domain",
                            "",
                        ),
                        "similarity": getattr(
                            evidence,
                            "similarity",
                            None,
                        ),
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
                    }
                )

            response["evidence"]["claims"].append(
                claim_data
            )

        logger.info(
            "[response_formatter] resposta final formatada"
        )

        return response

