# script temporário para alimentar/testar o FAISS com notícias de fontes confiáveis

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from urllib.parse import urlparse
from pipeline.preprocessing.extractor import TextExtractor
from pipeline.preprocessing.cleaner import TextCleaner
from pipeline.retrieval.vector_store import VectorStore, Document

URLS_PATH = Path("data/trusted_sources/urls.txt")
VECTOR_PATH = Path("data/vector_store")

TRUSTED_DOMAINS = {
    "g1.globo.com",
    "oglobo.globo.com",
    "folha.uol.com.br",
    "estadao.com.br",
    "www.estadao.com.br",
    "bbc.com",
    "www.bbc.com",
    "agenciabrasil.ebc.com.br",
}


def extract_domain(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def is_trusted_url(url: str) -> bool:
    domain = extract_domain(url)
    return domain in TRUSTED_DOMAINS


def load_urls() -> list[str]:
    if not URLS_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {URLS_PATH}\n"
            "Crie esse arquivo e coloque uma URL confiável por linha."
        )

    urls = []

    with open(URLS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            url = line.strip()

            if not url or url.startswith("#"):
                continue

            urls.append(url)

    return urls


def main():
    urls = load_urls()
    vector_store = VectorStore(store_path=VECTOR_PATH)

    documents = []

    for url in urls:
        if not is_trusted_url(url):
            print(f"[ignorado] domínio fora da lista confiável: {url}")
            continue

        print(f"[coletando] {url}")

        try:
            raw = TextExtractor.extract(url)
            blocks_clean = TextCleaner.clean_blocks(
                raw.get("content_blocks", []),
                source=raw.get("render_method", ""),
            )

            text = "\n".join(blocks_clean).strip()

            if len(text) < 500:
                print(f"[ignorado] texto muito curto: {url}")
                continue

            domain = extract_domain(url)

            documents.append(
                Document(
                    text=text,
                    source=domain,
                    url=url,
                    published_at=raw.get("published_at"),
                    metadata={
                        "title": raw.get("title", ""),
                        "description": raw.get("description", ""),
                        "domain": domain,
                        "source_type": "trusted_news",
                        "trusted_source": True,
                    },
                )
            )

            print(f"[ok] {raw.get('title', '')}")

        except Exception as exc:
            print(f"[erro] {url}: {exc}")

    if not documents:
        print("Nenhuma notícia foi indexada.")
        return

    ids = vector_store.add_documents(documents)
    print(f"\n{len(ids)} chunks indexados no FAISS.")


if __name__ == "__main__":
    main()
