from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from pipeline.analysis.reputation.fakebr_loader import FakeBrSourceDiscovery
from pipeline.analysis.reputation.repository import SourceReputationRepository
from pipeline.analysis.reputation.service import SourceReputationService


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Descobre fontes no Fake.Br e avalia cada uma individualmente pelo "
            "mesmo serviço genérico de reputação usado no pipeline da HÍBRIA."
        )
    )
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "data" / "datasets" / "fake_br_corpus"),
        help="Pasta raiz do Fake.Br Corpus.",
    )
    parser.add_argument("--include-fake", action="store_true", help="Inclui também URLs dos metadados de notícias falsas.")
    parser.add_argument("--diagnose", action="store_true", help="Mostra as fontes encontradas sem avaliar.")
    parser.add_argument("--list", action="store_true", help="Lista todas as fontes descobertas com seus índices.")
    parser.add_argument("--source-index", type=int, help="Avalia somente uma fonte pelo índice exibido em --list.")
    parser.add_argument("--domain", help="Avalia somente o domínio/URL informado, sem percorrer o corpus.")
    parser.add_argument("--start-at", type=int, default=1, help="Índice inicial para retomada. O primeiro índice é 1.")
    parser.add_argument("--limit", type=int, default=0, help="Quantidade máxima de fontes processadas nesta execução.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Pausa entre fontes, em segundos.")
    parser.add_argument("--force", "--no-skip-existing", dest="force", action="store_true", help="Refaz a avaliação mesmo que a fonte já esteja salva e válida.")
    parser.add_argument("--dry-run", action="store_true", help="Mostra as fontes selecionadas sem pesquisar nem salvar.")
    parser.add_argument("--init-schema", action="store_true", help="Inicializa o PostgreSQL ou o arquivo JSON persistente.")
    args = parser.parse_args()

    repository = SourceReputationRepository()
    schema_result = repository.init_schema() if args.init_schema else None
    service = SourceReputationService(repository=repository)
    discovery = FakeBrSourceDiscovery(args.dataset)

    if args.diagnose:
        print(json.dumps(discovery.diagnostics(include_fake=args.include_fake), ensure_ascii=False, indent=2))
        return

    sources = discovery.discover(include_fake=args.include_fake)

    if args.list:
        print(json.dumps({
            "dataset_path": str(args.dataset),
            "sources": [
                {"index": index, **source.to_dict()}
                for index, source in enumerate(sources, start=1)
            ],
        }, ensure_ascii=False, indent=2))
        return

    if args.domain:
        targets = [(1, args.domain, args.domain, 1)]
    elif args.source_index is not None:
        if args.source_index < 1 or args.source_index > len(sources):
            raise SystemExit(f"Índice inválido. Use --list; intervalo disponível: 1 a {len(sources)}.")
        source = sources[args.source_index - 1]
        targets = [(args.source_index, source.sample_url, source.domain, source.occurrences)]
    else:
        selected = [
            (index, source)
            for index, source in enumerate(sources, start=1)
            if index >= max(1, args.start_at)
        ]
        if args.limit > 0:
            selected = selected[:args.limit]
        targets = [
            (index, source.sample_url, source.domain, source.occurrences)
            for index, source in selected
        ]

    if args.dry_run:
        print(json.dumps({
            "dataset_path": str(args.dataset),
            "mode": "dry_run_individual_source_evaluation",
            "storage_backend": repository.backend,
            "sources_discovered": len(sources),
            "targets_selected": len(targets),
            "targets": [
                {
                    "index": index,
                    "domain": discovered_domain,
                    "sample_url": url,
                    "occurrences": occurrences,
                }
                for index, url, discovered_domain, occurrences in targets
            ],
        }, ensure_ascii=False, indent=2))
        return

    evaluated = []
    skipped = []
    failed = []

    for position, (index, url, discovered_domain, occurrences) in enumerate(targets, start=1):
        print(
            f"[fakebr] fonte {position}/{len(targets)} · índice={index} · {discovered_domain}",
            file=sys.stderr,
            flush=True,
        )
        try:
            existing = repository.get_by_domain_or_alias(discovered_domain)
            result = service.get_or_evaluate(
                url,
                trigger="fakebr_seed",
                force=args.force,
                metadata={
                    "fakebr_source_index": index,
                    "fakebr_occurrences": occurrences,
                    "fakebr_sample_url": url,
                },
            )
            item = {
                "index": index,
                "discovered_domain": discovered_domain,
                "canonical_domain": result.identity.canonical_domain,
                "source_name": result.identity.source_name,
                "status": result.status,
                "note": result.note,
                "classification": result.classification,
                "storage_backend": repository.backend,
                "storage_hit": bool(result.metadata.get("storage_hit")),
                "evidence_count": result.evidence_count,
                "providers_succeeded": result.providers_succeeded,
            }
            if existing is not None and not args.force and result.metadata.get("storage_hit"):
                skipped.append(item)
                action = "reutilizada"
            else:
                evaluated.append(item)
                action = "avaliada e salva"

            print(
                f"[fakebr] {action} · domínio={item['canonical_domain']} · "
                f"status={item['status']} · nota={item['note']}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            failed.append({
                "index": index,
                "domain": discovered_domain,
                "error": f"{type(exc).__name__}: {exc}",
            })

        # Cada fonte já foi salva antes desta pausa. A execução pode ser
        # interrompida e retomada com --start-at ou simplesmente repetida.
        if args.sleep > 0 and position < len(targets):
            time.sleep(args.sleep)

    print(json.dumps({
        "dataset_path": str(args.dataset),
        "mode": "individual_source_evaluation",
        "storage_backend": repository.backend,
        "storage_location": str(repository.json_path) if repository.backend == "json" else "PostgreSQL",
        "schema": schema_result,
        "sources_discovered": len(sources),
        "targets_selected": len(targets),
        "evaluated_count": len(evaluated),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "evaluated": evaluated,
        "skipped": skipped,
        "failed": failed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
