# =============================================================================
# index_seeds.py
# Popula a base vetorial com documentos de fontes confiáveis.
#
# Executar uma vez antes de usar o sistema:
#   python data/seeds/index_seeds.py
#
# Fontes indexadas:
#   Primária  — Fake.Br Corpus (notícias verdadeiras) — seção 3.3.2 do TCC
#   Secundária — RSS de portais confiáveis (opcional, requer internet)
#
# Uso básico:
#   python index_seeds.py
#   python index_seeds.py --fakebr /caminho/para/Fake.br-Corpus
#   python index_seeds.py --rss --max-per-feed 20
#   python index_seeds.py --fakebr /caminho --rss --max-true 500
# =============================================================================

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# garante que o ai_engine está no path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.retrieval.vector_store import VectorStore, Document
from pipeline.preprocessing.cleaner  import TextCleaner

# =============================================================================
# Configurações padrão
# =============================================================================

# Caminho padrão do Fake.Br — sobrescrito por --fakebr
# Estrutura do projeto:
# ai_engine/data/seeds/index_seeds.py  ← este arquivo
# ai_engine/data/datasets/fake_br_corpus/  ← corpus
# .parent = seeds/, .parent.parent = data/
DEFAULT_FAKEBR_PATH = Path(__file__).parent.parent / "datasets" / "fake_br_corpus"

# Máximo de notícias verdadeiras a indexar
# 300 é suficiente para o TCC — cobre os principais tópicos sem demorar muito
DEFAULT_MAX_TRUE = 300

# RSS feeds de portais confiáveis (seção 3.3.1 do TCC)
# Agência Brasil tem RSS público e sem bloqueio de bot — ideal para TCC
RSS_FEEDS = [
    ("Agência Brasil", "https://agenciabrasil.ebc.com.br/rss/ultimasnoticias/feed.xml"),
    ("G1 Política",    "https://g1.globo.com/rss/g1/politica/"),
    ("G1 Economia",    "https://g1.globo.com/rss/g1/economia/"),
    ("G1 Saúde",       "https://g1.globo.com/rss/g1/saude/"),
]

DEFAULT_MAX_PER_FEED = 15


# =============================================================================
# Fonte 1 — Fake.Br Corpus
# =============================================================================

def _find_true_dir(base: Path) -> Path | None:
    """
    Localiza o diretório de notícias verdadeiras dentro do Fake.Br.

    O corpus tem variações de estrutura dependendo do release:
      full/true/          ← release padrão
      size_normalized/true/
      true/               ← alguns clones simplificados
    """
    candidates = [
        base / "full_texts" / "true",
        base / "size_normalized_texts" / "true",
        base / "true",
    ]
    for path in candidates:
        if path.exists() and any(path.glob("*.txt")):
            return path
    return None


def _load_fakebr_metadata(base: Path) -> dict[str, dict]:
    """
    Carrega metadados do CSV do Fake.Br se disponível.

    O CSV contém: id, label, title, author, published, category, ...
    Retorna dict keyed por filename (ex: "1" → {title, category, ...}).
    Tolerante a ausência do arquivo — metadados são opcionais.
    """
    meta: dict[str, dict] = {}
    for csv_path in base.rglob("*.csv"):
        try:
            import csv
            with open(csv_path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # chave pode ser "id", "ID", ou nome do arquivo sem extensão
                    key = row.get("id") or row.get("ID") or row.get("filename", "")
                    if key:
                        meta[str(key).strip()] = row
            if meta:
                break  # usa o primeiro CSV encontrado
        except Exception:
            continue
    return meta


def index_fakebr(
    store:    VectorStore,
    base:     Path,
    max_true: int = DEFAULT_MAX_TRUE,
) -> tuple[int, int]:
    """
    Indexa notícias verdadeiras do Fake.Br Corpus em batch.

    Retorna (success, errors).

    Por que só as verdadeiras:
      O vector store é a base de referência do RAG — deve conter apenas
      conteúdo confiável para comparar com as claims da notícia analisada.
      Notícias falsas do corpus são usadas para treinar o BERTimbau,
      não para popular o índice de evidências (seção 3.3.2 do TCC).
    """
    true_dir = _find_true_dir(base)
    if true_dir is None:
        print(f"  ✗ diretório 'true' não encontrado em: {base}")
        print(f"    Estrutura esperada: ai_engine/data/datasets/fake_br_corpus/full_texts/true/*.txt")
        return 0, 1

    txt_files = sorted(true_dir.glob("*.txt"))[:max_true]
    if not txt_files:
        print(f"  ✗ nenhum .txt encontrado em: {true_dir}")
        return 0, 1

    print(f"  Encontrados {len(txt_files)} arquivos em {true_dir}")

    # carrega metadados se disponível
    meta = _load_fakebr_metadata(base)
    if meta:
        print(f"  Metadados CSV: {len(meta)} entradas")

    # acumula documentos para indexação em batch
    documents: list[Document] = []
    skipped = 0

    for txt_path in txt_files:
        try:
            text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as e:
            print(f"  ⚠ erro ao ler {txt_path.name}: {e}")
            skipped += 1
            continue

        # limpa o texto usando o mesmo cleaner do pipeline
        clean_blocks = TextCleaner.clean_blocks([text], source="static")
        if not clean_blocks:
            skipped += 1
            continue

        content = "\n\n".join(clean_blocks)

        # mínimo de conteúdo útil — descarta arquivos muito curtos
        if len(content) < 100:
            skipped += 1
            continue

        # metadados do CSV se disponíveis
        file_id  = txt_path.stem  # "1", "2", ...
        file_meta = meta.get(file_id, {})

        documents.append(Document(
            text         = content,
            source       = "Fake.Br/true",
            url          = file_meta.get("url", f"fakebr://true/{txt_path.name}"),
            published_at = file_meta.get("published") or file_meta.get("date"),
            metadata     = {
                "title":    file_meta.get("title", txt_path.stem),
                "category": file_meta.get("category", ""),
                "corpus":   "fake.br",
                "label":    "true",
            },
        ))

    if not documents:
        print(f"  ✗ nenhum documento válido extraído ({skipped} ignorados)")
        return 0, skipped

    print(f"  Indexando {len(documents)} documentos em batch ({skipped} ignorados)...")
    t0 = time.time()

    store.add_documents(documents, show_progress=True)

    elapsed = time.time() - t0
    print(f"  ✓ Fake.Br: {len(documents)} docs indexados em {elapsed:.1f}s")
    return len(documents), skipped


# =============================================================================
# Fonte 2 — RSS feeds (opcional)
# =============================================================================

def index_rss(
    store:        VectorStore,
    max_per_feed: int = DEFAULT_MAX_PER_FEED,
) -> tuple[int, int]:
    """
    Indexa artigos recentes via RSS de portais confiáveis.

    Requer: pip install feedparser
    Usa o extractor.py para baixar e limpar cada artigo.
    Mais lento que o Fake.Br — cada URL é uma requisição HTTP.
    """
    try:
        import feedparser
    except ImportError:
        print("  ✗ feedparser não instalado — execute: pip install feedparser")
        return 0, 1

    from pipeline.preprocessing.extractor import TextExtractor, ExtractionError

    success = 0
    errors  = 0

    for feed_name, feed_url in RSS_FEEDS:
        print(f"\n  ── {feed_name}")
        try:
            feed = feedparser.parse(feed_url)
            entries = feed.entries[:max_per_feed]
            print(f"     {len(entries)} artigos encontrados no feed")
        except Exception as e:
            print(f"     ✗ erro ao parsear feed: {e}")
            errors += 1
            continue

        for entry in entries:
            url = getattr(entry, "link", None)
            if not url:
                continue

            try:
                raw = TextExtractor.extract(url)
                clean_blocks = TextCleaner.clean_blocks(
                    raw["content_blocks"],
                    source=raw["render_method"],
                )

                if not clean_blocks or sum(len(b) for b in clean_blocks) < 200:
                    print(f"     ⚠ conteúdo insuficiente: {url[:60]}")
                    errors += 1
                    continue

                doc = Document(
                    text         = "\n\n".join(clean_blocks),
                    source       = feed_name,
                    url          = url,
                    published_at = getattr(entry, "published", None),
                    metadata     = {
                        "title":   getattr(entry, "title", ""),
                        "summary": getattr(entry, "summary", "")[:200],
                    },
                )
                store.add_document(doc)
                print(f"     ✓ {raw['title'][:55]}...")
                success += 1

            except ExtractionError as e:
                print(f"     ✗ {url[:60]} — {e}")
                errors += 1

    return success, errors


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Popula o vector store do HÍBRIA com documentos confiáveis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python index_seeds.py
  python index_seeds.py --fakebr ~/Downloads/Fake.br-Corpus
  python index_seeds.py --fakebr ~/Downloads/Fake.br-Corpus --max-true 500
  python index_seeds.py --rss --max-per-feed 20
  python index_seeds.py --fakebr ~/Downloads/Fake.br-Corpus --rss
  python index_seeds.py --clear
        """
    )
    parser.add_argument(
        "--fakebr",
        type=Path,
        default=DEFAULT_FAKEBR_PATH,
        metavar="PATH",
        help=f"caminho para o Fake.br-Corpus (padrão: {DEFAULT_FAKEBR_PATH})",
    )
    parser.add_argument(
        "--max-true",
        type=int,
        default=DEFAULT_MAX_TRUE,
        metavar="N",
        help=f"máximo de notícias verdadeiras a indexar (padrão: {DEFAULT_MAX_TRUE})",
    )
    parser.add_argument(
        "--rss",
        action="store_true",
        help="indexar artigos recentes via RSS (requer internet + feedparser)",
    )
    parser.add_argument(
        "--max-per-feed",
        type=int,
        default=DEFAULT_MAX_PER_FEED,
        metavar="N",
        help=f"máximo de artigos por feed RSS (padrão: {DEFAULT_MAX_PER_FEED})",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="limpa o índice existente antes de indexar",
    )
    parser.add_argument(
        "--skip-fakebr",
        action="store_true",
        help="pula o Fake.Br e usa apenas RSS",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  HÍBRIA — Indexação de Seeds")
    print("=" * 60)

    store = VectorStore()

    if args.clear:
        print("\n⚠ Limpando índice existente...")
        # reconstrói índice vazio
        store._create_index()
        store._metadata = []
        store._save()
        print("  Índice limpo.\n")

    stats = store.stats()
    print(f"\nEstado inicial: {stats['total_chunks']} chunks · "
          f"{stats['unique_documents']} documentos\n")

    total_success = 0
    total_errors  = 0

    # ── Fake.Br ───────────────────────────────────────────────────────────────
    if not args.skip_fakebr:
        fakebr_path = args.fakebr

        if not fakebr_path.exists():
            print(f"⚠ Fake.Br não encontrado em: {fakebr_path}")
            print("  Clone o corpus:")
            print("    git clone https://github.com/roneysco/Fake.br-Corpus data/datasets/fake_br_corpus")
            print(f"  Ou especifique o caminho: --fakebr /seu/caminho\n")
        else:
            print(f"── Fake.Br Corpus ({fakebr_path})")
            s, e = index_fakebr(store, fakebr_path, args.max_true)
            total_success += s
            total_errors  += e

    # ── RSS ───────────────────────────────────────────────────────────────────
    if args.rss:
        print(f"\n── RSS Feeds (máx {args.max_per_feed} por feed)")
        s, e = index_rss(store, args.max_per_feed)
        total_success += s
        total_errors  += e

    # ── Resumo ────────────────────────────────────────────────────────────────
    stats = store.stats()
    print(f"\n{'=' * 60}")
    print(f"  Indexação concluída")
    print(f"  Documentos novos:  {total_success}")
    print(f"  Erros/ignorados:   {total_errors}")
    print(f"  Total no índice:   {stats['total_chunks']} chunks · "
          f"{stats['unique_documents']} documentos únicos")
    if stats["sources"]:
        print(f"  Fontes:")
        for source, count in stats["sources"].items():
            print(f"    {source}: {count} chunks")
    print(f"  Tamanho do índice: {stats['index_size_mb']} MB")
    print("=" * 60)

    if total_success == 0:
        print("\n⚠ Nenhum documento indexado.")
        print("  Para começar rapidamente:")
        print("    git clone https://github.com/roneysco/Fake.br-Corpus data/datasets/fake_br_corpus")
        print("    python index_seeds.py --fakebr ./Fake.br-Corpus")
        sys.exit(1)


if __name__ == "__main__":
    main()