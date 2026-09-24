from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedbackRecord:
    rating: int
    category: str
    already_submitted: bool


class FeedbackRepository:
    """Persistência das avaliações anônimas dos resultados da extensão."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = (
            database_url
            if database_url is not None
            else os.getenv("HIBRIA_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or ""
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.database_url)

    @staticmethod
    def category(rating: int) -> str:
        if rating >= 4:
            return "positiva"
        if rating == 3:
            return "neutra"
        return "negativa"

    def save(self, analysis_id: str, evaluator_id: str, rating: int) -> FeedbackRecord:
        if not self.is_configured:
            raise RuntimeError("Banco de dados não configurado para avaliações.")

        query = """
            INSERT INTO avaliacoes_analise (
                analise_id,
                avaliador_id,
                nota
            ) VALUES (
                %(analysis_id)s,
                %(evaluator_id)s,
                %(rating)s
            )
            ON CONFLICT (analise_id, avaliador_id) DO NOTHING
            RETURNING nota, categoria;
        """
        existing_query = """
            SELECT nota, categoria
            FROM avaliacoes_analise
            WHERE analise_id = %(analysis_id)s
              AND avaliador_id = %(evaluator_id)s;
        """

        import psycopg2

        params = {
            "analysis_id": analysis_id,
            "evaluator_id": evaluator_id,
            "rating": rating,
        }

        with psycopg2.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
                already_submitted = row is None

                if row is None:
                    cursor.execute(existing_query, params)
                    row = cursor.fetchone()

                if row is None:
                    raise RuntimeError("A avaliação não pôde ser recuperada.")

        return FeedbackRecord(
            rating=int(row[0]),
            category=str(row[1]),
            already_submitted=already_submitted,
        )

    def summary(self) -> dict[str, Any]:
        empty = {
            "total": 0,
            "average": None,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "positive_percentage": 0.0,
            "neutral_percentage": 0.0,
            "negative_percentage": 0.0,
        }
        if not self.is_configured:
            return empty

        query = """
            SELECT
                total_avaliacoes,
                media_geral,
                positivas,
                neutras,
                negativas,
                percentual_positivas,
                percentual_neutras,
                percentual_negativas
            FROM resumo_avaliacoes_sistema;
        """

        import psycopg2

        with psycopg2.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                row = cursor.fetchone()

        if row is None:
            return empty

        return {
            "total": int(row[0] or 0),
            "average": float(row[1]) if row[1] is not None else None,
            "positive": int(row[2] or 0),
            "neutral": int(row[3] or 0),
            "negative": int(row[4] or 0),
            "positive_percentage": float(row[5] or 0),
            "neutral_percentage": float(row[6] or 0),
            "negative_percentage": float(row[7] or 0),
        }
