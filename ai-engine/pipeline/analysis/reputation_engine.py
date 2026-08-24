# =============================================================================
# Fachada de compatibilidade para a etapa de reputação.
#
# Não existe lista fixa de fontes ou notas. O serviço consulta a base persistente
# e, quando a fonte ainda não foi avaliada, pesquisa os critérios na web, calcula
# a nota e salva o resultado para reutilização.
# =============================================================================

from __future__ import annotations

from pipeline.analysis.reputation.service import SourceReputationService


class ReputationEngine:
    @staticmethod
    def evaluate(url: str) -> dict:
        return SourceReputationService().get_or_evaluate(
            url,
            trigger="pipeline",
        ).to_dict()
