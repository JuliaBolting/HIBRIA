from __future__ import annotations

from datetime import datetime, timezone
import sys
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch

from pipeline.persistence.analysis_repository import AnalysisRepository


class AnalysisCacheTests(unittest.TestCase):
    def test_cache_lookup_uses_url_and_pipeline_instead_of_content_hash(self):
        connection_context = MagicMock()
        connection = connection_context.__enter__.return_value
        cursor_context = connection.cursor.return_value
        cursor = cursor_context.__enter__.return_value
        cursor.fetchone.return_value = {
            "id": "analysis-id",
            "resultado_json": {"analysis": {"score": 50, "label": "teste"}},
            "data_analise": datetime.now(timezone.utc),
            "quantidade_consultas": 2,
        }

        psycopg2_module = ModuleType("psycopg2")
        extras_module = ModuleType("psycopg2.extras")
        extras_module.RealDictCursor = object
        psycopg2_module.connect = MagicMock(return_value=connection_context)
        psycopg2_module.extras = extras_module

        with patch.dict(
            sys.modules,
            {
                "psycopg2": psycopg2_module,
                "psycopg2.extras": extras_module,
            },
        ):
            cached = AnalysisRepository("postgresql://test").get(
                url="https://example.com/noticia?utm_source=teste",
                content="conteúdo dinâmico diferente",
            )

        self.assertIsNotNone(cached)
        self.assertEqual(cached.analysis_id, "analysis-id")

        sql, params = cursor.execute.call_args.args
        self.assertIn("hash_url = %(url_hash)s", sql)
        self.assertIn("versao_pipeline = %(pipeline_version)s", sql)
        self.assertNotIn("hash_conteudo =", sql)
        self.assertNotIn("content_hash", params)

    def test_normalized_url_ignores_tracking_parameters(self):
        first = AnalysisRepository.normalize_url(
            "https://G1.GLOBO.COM/noticia/?utm_source=teste&fbclid=abc"
        )
        second = AnalysisRepository.normalize_url(
            "https://g1.globo.com/noticia"
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
