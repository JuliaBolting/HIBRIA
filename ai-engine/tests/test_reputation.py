from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.analysis.reputation.collector import CollectionResult
from pipeline.analysis.reputation.evaluator import ReputationEvaluator
from pipeline.analysis.reputation.fakebr_loader import FakeBrSourceDiscovery
from pipeline.analysis.reputation.models import ReputationEvidence, SourceIdentity
from pipeline.analysis.reputation.repository import SourceReputationRepository
from pipeline.analysis.reputation.service import SourceReputationService


class FakeIdentityResolver:
    def resolve(self, value: str) -> SourceIdentity:
        return SourceIdentity(
            requested_url=value,
            requested_domain="example.com",
            canonical_url="https://example.com/",
            canonical_domain="example.com",
            source_name="Example Notícias",
            aliases=["www.example.com"],
            homepage_accessible=True,
            homepage_status_code=200,
            homepage_title="Example Notícias",
            homepage_text_excerpt="Página inicial jornalística.",
        )


class FakeCollector:
    def __init__(self, result: CollectionResult) -> None:
        self.result = result
        self.calls = 0

    def collect(self, identity: SourceIdentity) -> CollectionResult:
        self.calls += 1
        return self.result


def evidence(key: str, index: int, text: str) -> ReputationEvidence:
    return ReputationEvidence(
        criterion=key,
        provider="test_provider",
        title=text,
        url=f"https://evidence.example/{key}/{index}",
        snippet=text,
        evidence_type="search_result",
        is_same_domain=False,
    )


class ReputationTests(unittest.TestCase):
    def test_import_facade(self):
        from pipeline.analysis.reputation_engine import ReputationEngine
        self.assertTrue(hasattr(ReputationEngine, "evaluate"))

    def test_insufficient_evidence_does_not_become_low_reputation(self):
        identity = FakeIdentityResolver().resolve("https://example.com")
        collection = CollectionResult(
            evidence_by_criterion={},
            providers_attempted=["brave"],
            providers_succeeded=["brave"],
            query_count=8,
        )
        result = ReputationEvaluator().evaluate(identity, collection, origin="test")
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.note)
        self.assertEqual(result.score, 0.5)
        self.assertTrue(result.requires_external_evidence_weight)

    def test_complete_evaluation_uses_weighted_criteria(self):
        identity = FakeIdentityResolver().resolve("https://example.com")
        evidence_by_criterion = {
            "identificacao_institucional": [evidence("identificacao_institucional", 1, "Quem somos e expediente institucional"), evidence("identificacao_institucional", 2, "Quem somos e expediente institucional")],
            "autoria_responsabilidade_editorial": [evidence("autoria_responsabilidade_editorial", 1, "Autores, jornalistas e editoria"), evidence("autoria_responsabilidade_editorial", 2, "Autores, jornalistas e editoria")],
            "politica_correcoes_canal_erros": [evidence("politica_correcoes_canal_erros", 1, "Política de correções e errata"), evidence("politica_correcoes_canal_erros", 2, "Política de correções e errata")],
            "separacao_noticia_opiniao_publicidade": [evidence("separacao_noticia_opiniao_publicidade", 1, "Opinião e publicidade identificadas"), evidence("separacao_noticia_opiniao_publicidade", 2, "Opinião e publicidade identificadas")],
            "participacao_iniciativas_checagem": [evidence("participacao_iniciativas_checagem", 1, "Participação em projeto de checagem"), evidence("participacao_iniciativas_checagem", 2, "Participação em projeto de checagem")],
            "associacao_reconhecimento_jornalistico": [evidence("associacao_reconhecimento_jornalistico", 1, "Prêmio e associação jornalística"), evidence("associacao_reconhecimento_jornalistico", 2, "Prêmio e associação jornalística")],
            "historico_publico_desinformacao": [],
            "adequacao_tecnica_sistema": [evidence("adequacao_tecnica_sistema", 1, "Página acessível")],
        }
        collection = CollectionResult(
            evidence_by_criterion=evidence_by_criterion,
            providers_attempted=["brave"],
            providers_succeeded=["brave"],
            query_count=8,
        )
        result = ReputationEvaluator().evaluate(identity, collection, origin="test")
        self.assertEqual(result.status, "evaluated")
        self.assertIsNotNone(result.note)
        self.assertGreaterEqual(result.note, 80)

    def test_json_repository_persists_source_and_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = SourceReputationRepository(database_url="", json_path=Path(tmp) / "sources.json")
            identity = FakeIdentityResolver().resolve("https://example.com")
            result = ReputationEvaluator().evaluate(
                identity,
                CollectionResult(
                    evidence_by_criterion={},
                    providers_attempted=[],
                    providers_succeeded=[],
                ),
                origin="test",
            )
            repository.save(result)
            self.assertIsNotNone(repository.get_by_domain_or_alias("example.com"))
            self.assertIsNotNone(repository.get_by_domain_or_alias("www.example.com"))

    def test_service_reuses_stored_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = SourceReputationRepository(database_url="", json_path=Path(tmp) / "sources.json")
            collector = FakeCollector(CollectionResult(
                evidence_by_criterion={},
                providers_attempted=["brave"],
                providers_succeeded=["brave"],
            ))
            service = SourceReputationService(
                repository=repository,
                identity_resolver=FakeIdentityResolver(),
                collector=collector,
            )
            first = service.get_or_evaluate("https://example.com", trigger="test")
            second = service.get_or_evaluate("https://example.com", trigger="test")
            self.assertEqual(collector.calls, 1)
            self.assertEqual(first.identity.canonical_domain, second.identity.canonical_domain)

    def test_aggregator_ignores_insufficient_reputation(self):
        from types import SimpleNamespace
        from pipeline.output.aggregator import Aggregator

        result = SimpleNamespace(reputation={
            "status": "insufficient_evidence",
            "note": None,
            "score": 0.5,
        })
        self.assertIsNone(Aggregator._calculate_reputation_score(result))
        weights = Aggregator._normalize_component_weights(0.7, None)
        self.assertEqual(weights["reputation"], 0.0)

    def test_fakebr_reads_only_metadata_url_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "full_texts" / "true-meta-information"
            folder.mkdir(parents=True)
            (folder / "1-meta.txt").write_text(
                "Autor\nhttps://portal.example.com/noticia/1\npolitica\n",
                encoding="utf-8",
            )
            (folder / "2-meta.txt").write_text(
                "Outro autor\nhttps://portal.example.com/noticia/2\neconomia\n",
                encoding="utf-8",
            )
            sources = FakeBrSourceDiscovery(root).discover()
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0].domain, "portal.example.com")
            self.assertEqual(sources[0].occurrences, 2)


if __name__ == "__main__":
    unittest.main()
